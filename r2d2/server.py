from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path
import time

# Both adapters resolve local files only. No hub fallback during normal operation.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .backends import GGUFBackend, MLXBackend, GGUF_PATH, GGUF_VARIANTS, MLX_PATH
from .streaming import Stream, RATE, HOP, WINDOW, MAX_HOPS

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("r2d2")


class Engine:
    def __init__(self):
        # All Metal work, including model destruction, stays on one worker thread.
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")
        self.backend = None
        self.name = "gguf"
        self.state = "unloaded"
        self.busy = False
        self.lock = asyncio.Lock()
        self.error = ""

    async def call(self, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(self.executor, fn, *args)

    def _load(self, name):
        if self.backend:
            self.backend.close()
            self.backend = None
        backend = GGUFBackend(name) if name in GGUF_VARIANTS else MLXBackend()
        try:
            backend.load()
            for language in ("Chinese", "English"):
                for size in (5120, 7680, 10240):
                    backend.decode(np.zeros(size, np.float32), "", language, "", 4)
        except Exception:
            backend.close()
            raise
        self.backend = backend

    async def load(self, name):
        if self.backend and self.name == name:
            return
        self.name, self.state, self.error = name, "loading", ""
        try:
            await self.call(self._load, name)
            self.state = "ready"
        except Exception as exc:
            self.state, self.error = "error", str(exc)
            raise

    def status(self):
        return {"backend": self.name, "state": self.state, "busy": self.busy,
                "error": self.error, "chunk_ms": 160, "lookahead_ms": 160,
                "window_seconds": WINDOW // RATE, "sample_rate": RATE,
                "models": {**{name: all((GGUF_PATH / file).is_file() for file in files)
                                      and (MLX_PATH / "tokenizer.json").is_file()
                              for name, files in GGUF_VARIANTS.items()},
                           "mlx": (MLX_PATH / "model.safetensors").is_file()}}


engine = Engine()


@asynccontextmanager
async def lifespan(app):
    yield
    if engine.backend:
        await engine.call(engine.backend.close)
    engine.executor.shutdown(wait=True)


app = FastAPI(title="R2D2 // Listening room", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")


@app.get("/")
async def index():
    return FileResponse(ROOT / "web/index.html")


@app.get("/api/status")
async def status():
    return engine.status()


@app.get("/api/sample")
async def sample():
    return FileResponse(ROOT / "tests/fixtures/official-test.wav", media_type="audio/wav")


class Selection(BaseModel):
    backend: str = Field(pattern="^(gguf|gguf_q8|gguf_q4|mlx)$")


@app.post("/api/backend")
async def select(selection: Selection):
    async with engine.lock:
        if engine.busy or engine.state == "loading":
            raise HTTPException(409, "正在识别，请先停止当前录音")
        try:
            await engine.load(selection.backend)
        except Exception as exc:
            raise HTTPException(503, str(exc)) from exc
    return engine.status()


class SessionOptions(BaseModel):
    backend: str = Field(default="gguf", pattern="^(gguf|gguf_q8|gguf_q4|mlx)$")
    # Names must match the model's own config.json support_languages entries;
    # they are written straight into the prompt's language tag.
    language: str = Field(
        default="Chinese",
        pattern="^(Chinese|English|Japanese|Korean|Spanish|auto)$",
    )
    # Upstream caps the prompt-borne hint at MAX_SYSTEM_PROMPT_CHARS.
    context: str = Field(default="", max_length=4000)


@app.websocket("/api/stream")
async def stream_socket(ws: WebSocket):
    await ws.accept()
    owned = False
    receive_task = None
    try:
        options = SessionOptions.model_validate_json(await asyncio.wait_for(ws.receive_text(), 15))
        async with engine.lock:
            if engine.busy:
                await ws.send_json({"type": "error", "message": "已有录音正在识别，请稍后重试"})
                await ws.close(code=1013)
                return
            engine.busy = owned = True
            await ws.send_json({"type": "loading", "backend": options.backend})
            await engine.load(options.backend)
        session = Stream(engine.backend, None if options.language == "auto" else options.language,
                         options.context)
        await ws.send_json({"type": "ready", "backend": options.backend, "sample_rate": RATE,
                            "chunk_ms": 160})
        queue = asyncio.Queue(maxsize=64)
        received = 0

        async def receive():
            nonlocal received
            try:
                while True:
                    packet = await asyncio.wait_for(ws.receive(), timeout=30)
                    if packet["type"] == "websocket.disconnect":
                        raise WebSocketDisconnect()
                    if packet.get("bytes") is not None:
                        data = packet["bytes"]
                        if not data or len(data) % 2 or len(data) > HOP * 2:
                            raise ValueError("音频必须为最多 160 ms 的 16 kHz 单声道 PCM16")
                        received += len(data) // 2
                        if received > RATE * 300:
                            raise ValueError("单次录音上限为 5 分钟，请停止后开启新录音")
                        try:
                            queue.put_nowait(data)
                        except asyncio.QueueFull:
                            raise ValueError("识别积压超过 10 秒，已停止；请查看延迟后重试")
                    elif packet.get("text") == "stop":
                        await queue.put(None)
                        return
                    else:
                        raise ValueError("未知的音频消息")
            except BaseException as exc:
                # Signal failure without waiting behind a full audio queue.
                while not queue.empty():
                    queue.get_nowait()
                queue.put_nowait(exc)

        receive_task = asyncio.create_task(receive())
        started = time.perf_counter()
        first_text_ms = None
        while True:
            item = await queue.get()
            if isinstance(item, BaseException):
                raise item
            if item is None:
                break
            # Hand the stream everything already waiting. Whatever queued up while
            # the last decode ran is exactly the backlog Stream.feed merges into
            # fewer, longer steps; one packet at a time would hide it in the queue.
            chunks, stopped = [item], False
            while len(chunks) < MAX_HOPS and not queue.empty():
                nxt = queue.get_nowait()
                if isinstance(nxt, BaseException):
                    raise nxt
                if nxt is None:
                    stopped = True
                    break
                chunks.append(nxt)
            pcm = np.frombuffer(b"".join(chunks), dtype="<i2").astype(np.float32) / 32768
            updates = await engine.call(session.feed, pcm)
            for update in updates:
                if update["text"] and first_text_ms is None:
                    first_text_ms = round((time.perf_counter() - started) * 1000)
                update["backlog_ms"] = round(max(0, received - session.processed) / RATE * 1000)
                update["first_text_ms"] = first_text_ms
                await ws.send_json(update)
            if stopped:
                break
        final = await engine.call(session.finish)
        if final["text"] and first_text_ms is None:
            first_text_ms = round((time.perf_counter() - started) * 1000)
        final["backlog_ms"] = 0
        final["first_text_ms"] = first_text_ms
        await ws.send_json(final)
        await ws.send_json({"type": "done", "backend": options.backend,
                            "audio_ms": round(received / RATE * 1000)})
        await ws.close()
    except WebSocketDisconnect:
        pass
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        log.exception("Streaming session failed")
        try:
            await ws.send_json({"type": "error", "message": str(exc)})
            await ws.close(code=1011)
        except Exception:
            pass
    finally:
        if receive_task:
            receive_task.cancel()
            await asyncio.gather(receive_task, return_exceptions=True)
        if owned:
            # Drain any worker call left running by a disconnected client before unlocking.
            await engine.call(lambda: None)
            engine.busy = False


def main():
    import uvicorn
    uvicorn.run("r2d2.server:app", host=os.environ.get("HOST", "0.0.0.0"),
                port=int(os.environ.get("PORT", "8765")), ws_max_size=16384)


if __name__ == "__main__":
    main()
