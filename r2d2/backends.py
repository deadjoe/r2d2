"""Local, greedy decoding adapters. Audio streaming policy lives in streaming.py."""
from __future__ import annotations

import base64
import io
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
import wave

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = Path(os.environ.get("R2D2_MODELS", ROOT / "models"))
MLX_PATH = MODELS / "Confucius4-R2T2-MLX-BF16"
GGUF_PATH = MODELS / "Confucius4-R2T2-GGUF"
GGUF_VARIANTS = {
    "gguf": ("Confucius4-R2T2-f16.gguf", "mmproj-Confucius4-R2T2-f16.gguf"),
    "gguf_q8": ("Confucius4-R2T2-Q8_0.gguf", "mmproj-Confucius4-R2T2-Q8_0.gguf"),
    "gguf_q4": ("Confucius4-R2T2-Q4_K_M.gguf", "mmproj-Confucius4-R2T2-Q8_0.gguf"),
}


def wav_bytes(audio: np.ndarray) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes((np.clip(audio, -1, 32767 / 32768) * 32768).astype("<i2").tobytes())
    return out.getvalue()


def start_llama_server(args, log_name, label, timeout=120):
    """Private llama-server child bound to a random loopback port. The web app
    binds 0.0.0.0; inference processes never do. Returns (process, client, log)."""
    binary = shutil.which("llama-server")
    if not binary:
        raise RuntimeError("缺少 llama-server，请运行 brew install llama.cpp")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    (ROOT / ".runtime").mkdir(exist_ok=True)
    log = (ROOT / ".runtime" / log_name).open("w")
    process = subprocess.Popen(
        [binary, "-m", *args, "--parallel", "1", "--host", "127.0.0.1", "--port", str(port),
         "--no-webui", "--no-warmup", "--log-disable"],
        stdout=log, stderr=log,
    )
    client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=90, trust_env=False)
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"{label} 启动失败，详情见 .runtime/{log_name}")
            try:
                if client.get("/health").status_code == 200:
                    return process, client, log
            except httpx.HTTPError:
                pass
            time.sleep(0.2)
        raise RuntimeError(f"{label} 加载超时")
    except BaseException:
        stop_llama_server(process, client, log)
        raise


def stop_llama_server(process, client, log):
    if client:
        client.close()
    if process and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    if log:
        log.close()


class GGUFBackend:
    name = "gguf"

    def __init__(self, name="gguf"):
        self.model_file, self.projector_file = GGUF_VARIANTS[name]
        self.name = name
        self.process = None
        self.client = None
        self.log = None

    def load(self):
        from transformers import AutoTokenizer

        model = GGUF_PATH / self.model_file
        projector = GGUF_PATH / self.projector_file
        for path in (model, projector, MLX_PATH / "tokenizer.json"):
            if not path.is_file():
                raise RuntimeError(f"缺少本地模型文件：{path}")
        self.tokenizer = AutoTokenizer.from_pretrained(MLX_PATH, local_files_only=True)
        self.process, self.client, self.log = start_llama_server(
            [str(model), "--mmproj", str(projector), "-ngl", "99",
             # Every step carries new audio, so a saved prompt state can never be
             # reused; the default 8 GiB host-RAM prompt cache only fills up.
             # Upstream's native backend clears memory on every call for the same reason.
             "-c", "4096", "--cache-ram", "0"],
            "llama-server.log", "GGUF")
        self.marker = self.client.get("/props").json().get("media_marker", "<__media__>")

    def decode(self, audio, prefix, language, context, max_tokens):
        assistant = (f"language {language}<asr_text>" if language else "") + prefix
        # The model's own chat_template.json closes the system text directly with
        # <|im_end|>; mlx-audio's extra "\n" is its divergence, not the canon.
        prompt = (f"<|im_start|>system\n{context}<|im_end|>\n"
                  f"<|im_start|>user\n{self.marker}<|im_end|>\n"
                  f"<|im_start|>assistant\n{assistant}")
        response = self.client.post("/completion", json={
            "prompt": {"prompt_string": prompt,
                       "multimodal_data": [base64.b64encode(wav_bytes(audio)).decode()]},
            "n_predict": max_tokens, "temperature": 0, "cache_prompt": True,
            "stream": False, "stop": ["<|im_end|>", "<|endoftext|>"],
        })
        response.raise_for_status()
        return response.json()["content"]

    def close(self):
        stop_llama_server(self.process, self.client, self.log)


class MLXBackend:
    name = "mlx"

    def load(self):
        from mlx_audio.stt.utils import load_model

        if not (MLX_PATH / "model.safetensors").is_file():
            raise RuntimeError(f"缺少本地 MLX 模型：{MLX_PATH}")
        loaded = load_model(str(MLX_PATH), strict=True)
        # mlx-audio's dispatcher wraps __call__(*args, **kwargs); generation's
        # signature check needs the actual Qwen3ASRModel underneath it.
        self.model = getattr(loaded, "_model", loaded)
        self.tokenizer = self.model._tokenizer

    def decode(self, audio, prefix, language, context, max_tokens):
        import mlx.core as mx
        from mlx_audio.lm.generate import generate_step

        model = self.model
        features, mask, n_audio = model._preprocess_audio(audio)
        ids = model._build_prompt(n_audio, language, context or None)
        # Continue the assistant's confirmed prefix, never insert a second chat turn.
        if prefix:
            tail = mx.array(self.tokenizer.encode(prefix, add_special_tokens=False))[None, :]
            ids = mx.concatenate([ids, tail], axis=1)
        audio_features = model.get_audio_features(features, mask)
        embeds = model._build_inputs_embeds(ids, audio_features)
        mx.eval(embeds)
        tokens = []
        for token, _ in generate_step(
            prompt=ids[0], input_embeddings=embeds[0], model=model,
            max_tokens=max_tokens, sampler=lambda logits: mx.argmax(logits, axis=-1),
        ):
            value = int(token)
            if value in model._eos_token_ids():
                break
            tokens.append(value)
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def close(self):
        import gc
        import mlx.core as mx

        self.model = None
        self.tokenizer = None
        gc.collect()
        mx.clear_cache()
