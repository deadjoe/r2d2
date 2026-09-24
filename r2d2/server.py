from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
import time
import tomllib

# Both adapters resolve local files only. No hub fallback during normal operation.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .backends import GGUFBackend, MLXBackend, GGUF_PATH, GGUF_VARIANTS, MLX_PATH, MLX_SUPPORTED
from .streaming import Stream, RATE, HOP, WINDOW, MAX_HOPS
from .translate import LiveTranslation, Translator, MT_FILE, DEVICE as MT_DEVICE

ROOT = Path(__file__).resolve().parents[1]
VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
log = logging.getLogger("r2d2")


class Engine:
    def __init__(self):
        # All GPU work (Metal or CUDA), including model destruction, stays on one worker thread.
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")
        self.backend = None
        # The chosen default after the far-field comparison; see docs/validation.md.
        self.name = "gguf_q8"
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
                           "mlx": MLX_SUPPORTED and (MLX_PATH / "model.safetensors").is_file()}}


class TranslationEngine:
    """Hy-MT2 on the CPU, loaded on first use and kept: about 1.2 GB of RAM and no
    GPU. Its own worker thread, so a translation never queues behind a decode."""

    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="translate")
        self.translator = None
        self.state = "unloaded"
        self.error = ""
        self.lock = asyncio.Lock()

    async def call(self, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(self.executor, fn, *args)

    async def load(self):
        async with self.lock:
            if self.translator:
                return
            self.state, self.error = "loading", ""
            translator = Translator()
            try:
                await self.call(translator.load)
            except Exception as exc:
                await self.call(translator.close)
                self.state, self.error = "error", str(exc)
                raise
            self.translator, self.state = translator, "ready"

    async def translate(self, text, abort=None, target="Chinese"):
        return await self.call(self.translator.translate, text, abort, target)

    def status(self):
        return {"model": MT_FILE, "device": MT_DEVICE, "available": Translator.available(), "state": self.state,
                "error": self.error}

    async def close(self):
        if self.translator:
            await self.call(self.translator.close)
            self.translator = None
        self.executor.shutdown(wait=True)


engine = Engine()
translation = TranslationEngine()


@asynccontextmanager
async def lifespan(app):
    yield
    if engine.backend:
        await engine.call(engine.backend.close)
    engine.executor.shutdown(wait=True)
    await translation.close()


app = FastAPI(title="R2D2 // Listening room", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")


@app.middleware("http")
async def revalidate_page(request, call_next):
    # Without it the browser may reuse a cached app.js next to a newer
    # index.html, and the stale script speaks an older session protocol.
    # no-cache still allows the cache; it just asks first (a 304 when unchanged).
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/")
async def index():
    return FileResponse(ROOT / "web/index.html")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    # For clients that ask for the conventional path instead of the <link>.
    return FileResponse(ROOT / "web/favicon.svg", media_type="image/svg+xml")


@app.get("/api/status")
async def status():
    return {**engine.status(), "translation": translation.status(), "version": VERSION}


@app.get("/api/sample")
async def sample():
    return FileResponse(ROOT / "tests/fixtures/official-test.wav", media_type="audio/wav")


class Selection(BaseModel):
    backend: str = Field(pattern="^(gguf|gguf_q8|gguf_q4|mlx)$")


@app.post("/api/backend")
async def select(selection: Selection):
    async with engine.lock:
        if engine.busy or engine.state == "loading":
            raise HTTPException(409, "Recognition in progress; stop the current recording first")
        try:
            await engine.load(selection.backend)
        except Exception as exc:
            raise HTTPException(503, str(exc)) from exc
    return engine.status()


class SessionOptions(BaseModel):
    backend: str = Field(default="gguf_q8", pattern="^(gguf|gguf_q8|gguf_q4|mlx)$")
    # Names must match the model's own config.json support_languages entries;
    # they are written straight into the prompt's language tag.
    language: str = Field(
        default="Chinese",
        pattern="^(Chinese|English|Japanese|Korean|Spanish|auto)$",
    )
    # Upstream caps the prompt-borne hint at MAX_SYSTEM_PROMPT_CHARS.
    context: str = Field(default="", max_length=4000)
    # Live translation target, or "off". A target equal to the spoken language
    # means recognition only.
    translate: str = Field(default="off", pattern="^(off|Chinese|English|Japanese|Korean|Spanish)$")


@app.websocket("/api/stream")
async def stream_socket(ws: WebSocket):
    await ws.accept()
    owned = False
    receive_task = None
    live = None
    received = 0  # samples
    last_packet = time.perf_counter()
    send_lock = asyncio.Lock()

    async def send(message):
        # Transcript and translation updates come from two tasks; one frame at a time.
        async with send_lock:
            await ws.send_json(message)
    try:
        options = SessionOptions.model_validate_json(await asyncio.wait_for(ws.receive_text(), 15))
        async with engine.lock:
            if engine.busy:
                await ws.send_json({"type": "error", "message": "Another recording is already being recognized; try again shortly"})
                await ws.close(code=1013)
                return
            engine.busy = owned = True
            await ws.send_json({"type": "loading", "backend": options.backend})
            await engine.load(options.backend)
        session = Stream(engine.backend, None if options.language == "auto" else options.language,
                         options.context)
        target = options.translate
        if target not in ("off", options.language):
            try:
                await translation.load()
                live = LiveTranslation(
                    lambda text, abort=None: translation.translate(text, abort, target), send, target)
            except Exception as exc:
                # Recognition does not depend on translation; say so and carry on.
                await ws.send_json({"type": "translation_error", "message": f"Translation model unavailable: {exc}"})
        await ws.send_json({"type": "ready", "backend": options.backend, "sample_rate": RATE,
                            "chunk_ms": 160, "translate": live is not None,
                            "target": target if live else None})
        queue = asyncio.Queue(maxsize=64)

        async def receive():
            nonlocal received, last_packet
            try:
                while True:
                    packet = await asyncio.wait_for(ws.receive(), timeout=30)
                    if packet["type"] == "websocket.disconnect":
                        raise WebSocketDisconnect(packet.get("code", 1000), packet.get("reason"))
                    last_packet = time.perf_counter()
                    if packet.get("bytes") is not None:
                        data = packet["bytes"]
                        if not data or len(data) % 2 or len(data) > HOP * 2:
                            raise ValueError("Audio must be 16 kHz mono PCM16 packets of at most 160 ms")
                        received += len(data) // 2
                        if received > RATE * 300:
                            raise ValueError("A single recording is limited to 5 minutes; stop and start a new one")
                        try:
                            queue.put_nowait(data)
                        except asyncio.QueueFull:
                            raise ValueError("Recognition fell more than 10 s behind and stopped; check the latency and retry")
                    elif packet.get("text") == "stop":
                        await queue.put(None)
                        return
                    else:
                        raise ValueError("Unknown audio message")
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
                await send(update)
                if live:
                    live.update(update["text"], update["draft"], language=update["language"])
            if stopped:
                break
        final = await engine.call(session.finish)
        if final["text"] and first_text_ms is None:
            first_text_ms = round((time.perf_counter() - started) * 1000)
        final["backlog_ms"] = 0
        final["first_text_ms"] = first_text_ms
        await send(final)
        if live:
            live.update(final["text"], final=True, language=final["language"])
            await live.finish()
            await send(live.snapshot())
        await send({"type": "done", "backend": options.backend,
                    "audio_ms": round(received / RATE * 1000)})
        await ws.close()
    except WebSocketDisconnect as exc:
        # Nothing to tell a client that is gone, but the close code says who closed
        # it: 1000/1001 the page, 1006 the network path, 1011 a missed keepalive.
        log.warning("Client disconnected mid-session: code %s%s, %.1f s of audio received, "
                    "last packet %.1f s before", exc.code, f" ({exc.reason})" if exc.reason else "",
                    received / RATE, time.perf_counter() - last_packet)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        log.exception("Streaming session failed")
        try:
            await send({"type": "error", "message": str(exc)})
            await ws.close(code=1011)
        except Exception:
            pass
    finally:
        if live:
            await live.close()
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
