# R2D2 // Listening Room

## Overview

A local web app for streaming speech recognition with NetEase Youdao's
Confucius4-R2T2 on Apple Silicon Macs. The browser sends microphone audio to a
local server, which transcribes it in 160 ms steps and shows the text as it is
recognized. Non-Chinese speech can be translated into Chinese live, alongside
the transcript, with Tencent's HY-MT1.5-1.8B.

All inference runs on the Mac. The app is also a test bench: the same
recording can be replayed through different model builds, and each run can be
exported with per-step timings.

## Quick start

```bash
brew install llama.cpp
uv sync --frozen

# Recognition: GGUF Q8_0 (the default), plus the tokenizer files it uses
uv run hf download netease-youdao/Confucius4-R2T2-GGUF \
  Confucius4-R2T2-Q8_0.gguf mmproj-Confucius4-R2T2-Q8_0.gguf \
  --local-dir models/Confucius4-R2T2-GGUF
uv run hf download netease-youdao/Confucius4-R2T2 --exclude "*.safetensors" \
  --local-dir models/Confucius4-R2T2-MLX-BF16

# Translation
uv run hf download tencent/HY-MT1.5-1.8B-GGUF \
  HY-MT1.5-1.8B-Q4_K_M.gguf License.txt --local-dir models/HY-MT1.5-1.8B-GGUF

./server.sh start
```

Open <http://localhost:8765>, choose the spoken language and press
**Start listening**. The model loads on the first start. **Stop and finish**
flushes the remaining audio. The pinned revisions and SHA-256 of all weights
are in [models-manifest.json](models-manifest.json).

Other things the page can do:

- **Try the official sample** or **Import audio** (up to 5 min / 100 MB)
  instead of using the microphone.
- **Replay last take** sends the identical PCM to the currently selected
  model at real-time speed.
- **Expand** fills the window with the transcript and translation. Press
  **Restore** or Esc to go back. Running sessions are not interrupted.
- **Export** saves the text, translation, settings, per-step timings and your
  listening notes as JSON.

Service control:

```bash
./server.sh start | status | log [-f] | stop
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `PORT` / `HOST` | `8765` / `0.0.0.0` | Web server address |
| `R2D2_MODELS` | `./models` | Model root |
| `R2D2_MT_THREADS` | `6` | CPU threads for translation |

The page is reachable over the LAN, but browsers only allow the microphone on
`localhost` or HTTPS pages.

## Requirements

- Apple Silicon Mac. Tested on an M1 Max with 64 GB of memory.
- Python 3.12 and [uv](https://docs.astral.sh/uv/).
- llama.cpp from Homebrew. Verified with 0.4.1 (b29c606e2).
- Chrome or another browser with AudioWorklet support.
- Disk space for the default setup: about 3.3 GB (Q8_0 recognition,
  translation model, tokenizer files).
- Memory at runtime: about 2.7 GB for Q8_0 recognition and 1.2 GB for
  translation.

After download, the app runs offline. Other model builds are optional:

```text
models/
├── Confucius4-R2T2-GGUF/        official GGUF
│   ├── Confucius4-R2T2-Q8_0.gguf + mmproj-Confucius4-R2T2-Q8_0.gguf
│   ├── Confucius4-R2T2-f16.gguf  + mmproj-Confucius4-R2T2-f16.gguf    (optional)
│   └── Confucius4-R2T2-Q4_K_M.gguf, uses the Q8_0 mmproj               (optional)
├── Confucius4-R2T2-MLX-BF16/    tokenizer files (required);
│                                model.safetensors for MLX (optional, local conversion)
└── HY-MT1.5-1.8B-GGUF/
    └── HY-MT1.5-1.8B-Q4_K_M.gguf
```

The MLX BF16 build is a local conversion of the official model with
mlx-audio. It is not published.

## Features

**Streaming recognition.** The app ports the official
`streaming_transcribe_no_reset` policy. Each step:

- re-encodes the current audio window;
- continues from the confirmed text;
- holds back the last token until the next step.

The first step waits for 160 ms of lookahead. After that, one step runs every
160 ms. Confirmed text is shown in white and is only ever appended to. The
unconfirmed tail is shown in grey.

Departures from upstream, each measured on this hardware:

- **Window size.** The rolling window is 8 s and drops 4 s at a time.
  Upstream uses 16 s and 8 s.
- **Catching up.** When recognition falls behind, up to three 160 ms chunks
  are merged into one step instead of queuing.
- **No VAD.** No voice-activity detection is used.
- **Loops.** A repetition loop, or 60 s without a reset, rebuilds the decoder
  state. Confirmed text is never rewritten.

Details are in [docs/streaming.md](docs/streaming.md).

**Engines.** GGUF F16, Q8_0 and Q4_K_M run through llama.cpp on Metal. MLX
BF16 runs through mlx-audio. Q8_0 is the default: in far-field tests it could
not be told apart from F16 and had the best step timing. Q4_K_M was clearly
worse. See [docs/validation.md](docs/validation.md).

**Languages.** Chinese, English, Japanese, Korean, Spanish, or auto-detect.
An optional hotword and topic hint (up to 4000 characters) goes into the
prompt, as upstream's `context` does.

**Live translation into Chinese.** HY-MT1.5-1.8B runs in its own llama.cpp
process on the CPU. On the GPU it slowed recognition past its 160 ms budget;
on the CPU it had no measurable effect.

- A sentence is translated once it is confirmed and closed. That translation
  is final.
- The unfinished sentence is re-translated as a grey draft whenever it
  changes.
- Chinese speech is not translated.
- If the translator fails, recognition continues.

Details and measurements are in [docs/translation.md](docs/translation.md).

**Metrics.** The page shows four numbers:

- **First text:** time to the first confirmed text.
- **Decode / step:** time taken by each recognition step.
- **Audio backlog:** audio received but not yet decoded.
- **Translation lag:** time from a sentence being confirmed to its
  translation being final.

**Limits.**

- One session at a time.
- 5 minutes per session.
- A session stops with an error if recognition falls more than 10 s behind.
- Audio stays in the page's memory and is never written to disk.
- Inference processes listen only on loopback.

## Third-party models and code

| Project | Used for | License |
| --- | --- | --- |
| [Confucius4-R2T2](https://huggingface.co/netease-youdao/Confucius4-R2T2) ([GGUF](https://huggingface.co/netease-youdao/Confucius4-R2T2-GGUF)), NetEase Youdao | Speech recognition model and tokenizer | NetEase model license, shipped with the weights |
| [Confucius4-R2T2 source](https://github.com/netease-youdao/Confucius4-R2T2) | Streaming policy adapted in `r2d2/streaming.py`; sample audio in `tests/fixtures/` | Apache-2.0 |
| [HY-MT1.5-1.8B-GGUF](https://huggingface.co/tencent/HY-MT1.5-1.8B-GGUF), Tencent | Translation model | Tencent HY Community License (excludes the EU, UK and South Korea) |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | GGUF inference (`llama-server`) | MIT |
| [mlx-audio](https://github.com/Blaizzy/mlx-audio) / [MLX](https://github.com/ml-explore/mlx) | MLX inference | MIT |
| [Transformers](https://github.com/huggingface/transformers) | Tokenizer | Apache-2.0 |
| [FastAPI](https://github.com/fastapi/fastapi), [Uvicorn](https://github.com/encode/uvicorn), [HTTPX](https://github.com/encode/httpx) | Web server and internal HTTP | MIT, BSD-3-Clause, BSD-3-Clause |

Adapted code and its provenance are listed in
[third_party/NOTICE.md](third_party/NOTICE.md). Model weights are not part of
this repository and keep their own licenses.

This project is licensed under the [Apache License 2.0](LICENSE).
