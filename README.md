# R2D2 // Listening Room

A local web app for streaming speech recognition with NetEase Youdao's
Confucius4-R2T2, with optional live translation by Tencent's Hy-MT2-1.8B.
The browser streams microphone audio to a local server, which transcribes it
every 160 ms. Everything runs on your machine: an Apple Silicon Mac, or Linux
with an NVIDIA GPU.

It is also a test bench: replay the same recording through another model
build and export each run with per-step timings.

## Requirements

- An Apple Silicon Mac, or Linux with an NVIDIA GPU and the CUDA 12 toolkit.
- Python 3.12 and [uv](https://docs.astral.sh/uv/).
- llama.cpp: from Homebrew on a Mac, built with CUDA on Linux. Verified with
  0.4.1 (b29c606e2).
- A browser with AudioWorklet support, such as Chrome.
- About 3.3 GB of disk for the default models.

Tested on an M1 Max (64 GB), and on an RTX 4000 Ada (20 GB) and an RTX 2000
Ada (16 GB) under Ubuntu 24.04 with CUDA 12.8. All keep up in real time.

## Install

Get the code:

```bash
git clone https://github.com/deadjoe/r2d2 && cd r2d2
```

Install `llama-server`. On a Mac:

```bash
brew install llama.cpp
```

On Linux, build it with CUDA:

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout b29c606e2          # the version this app is verified with
export PATH=/usr/local/cuda/bin:$PATH   # the toolkit often leaves nvcc off PATH
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native \
  -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
# Each CUDA compile job needs 1-2 GB of RAM; an unbounded -j runs out of memory.
cmake --build build -j 8 --target llama-server
export R2D2_LLAMA_SERVER=$PWD/build/bin/llama-server   # or put it on PATH
cd -
```

`R2D2_LLAMA_SERVER` lasts only for this shell. In a new one, export it again (or add it to
your shell profile) before `./server.sh start`, or recognition fails when it starts.

Then install the app and download the models:

```bash
uv sync --frozen

# Recognition: GGUF Q8_0 (the default), plus the tokenizer files it uses
uv run hf download netease-youdao/Confucius4-R2T2-GGUF \
  Confucius4-R2T2-Q8_0.gguf mmproj-Confucius4-R2T2-Q8_0.gguf \
  --local-dir models/Confucius4-R2T2-GGUF
uv run hf download netease-youdao/Confucius4-R2T2 --exclude "*.safetensors" \
  --local-dir models/Confucius4-R2T2-MLX-BF16

# Translation
uv run hf download tencent/Hy-MT2-1.8B-GGUF \
  Hy-MT2-1.8B-Q4_K_M.gguf LICENSE.txt --local-dir models/Hy-MT2-1.8B-GGUF
```

Pinned revisions and SHA-256 of all weights are in
[models-manifest.json](models-manifest.json). After download the app runs
offline. Instead of the three downloads above, `uv run python -m r2d2.fetch`
fetches the same files from those pinned revisions, resumes an interrupted
download and checks every file's SHA-256 (`--check` verifies only;
`--set default,q4` adds Q4_K_M).

Optional models, shown greyed out on the page until present:

- GGUF F16 (`Confucius4-R2T2-f16.gguf` + `mmproj-Confucius4-R2T2-f16.gguf`)
  and Q4_K_M (`Confucius4-R2T2-Q4_K_M.gguf`, uses the Q8_0 mmproj), from the
  same GGUF repository.
- MLX BF16, Mac only: a local conversion with mlx-audio, saved as
  `models/Confucius4-R2T2-MLX-BF16/model.safetensors`. Not published.

## Run

```bash
./server.sh start | status | log [-f] | stop
```

Open <http://localhost:8765>, choose the spoken language and the translation
target, and press **Start listening**. Models load on first use.

Browsers allow the microphone only on `localhost` or HTTPS. For a remote Linux
machine, forward the port and open `localhost`:
`ssh -L 8765:localhost:8765 <host>`. A cloud GPU's HTTPS proxy works as well
(RunPod: `https://<pod>-8765.proxy.runpod.net`); such a URL is public, so set
`R2D2_ACCESS_KEY` there (see below).

| Variable | Default | Purpose |
| --- | --- | --- |
| `PORT` / `HOST` | `8765` / `0.0.0.0` | Web server address |
| `R2D2_MODELS` | `./models` | Model root |
| `R2D2_LLAMA_SERVER` | `llama-server` on `PATH` | llama.cpp server binary |
| `R2D2_MT_DEVICE` | `cpu` on a Mac, `gpu` on Linux | Where translation runs |
| `R2D2_MT_THREADS` | `6` | Translation threads when on the CPU |
| `R2D2_ACCESS_KEY` | unset | 16-128 of `A-Za-z0-9_-`: the page, API and audio socket then need it; open `/?k=<key>` once and a cookie carries it |

## Container (Linux + NVIDIA)

A prebuilt image with the app and a CUDA build of `llama-server` for every GPU
generation from Turing (T4) to Blackwell (RTX 50, RTX PRO). It contains no
weights: on first start it downloads the default set (Q8_0 + Hy-MT2, 3.3 GB) into
`/data`, from the pinned revisions, and verifies each file's SHA-256; with `/data`
on a volume, later starts reuse the files and check them in milliseconds. Once the
app is ready, the F16 recogniser (4.1 GB more) downloads in the background; its
button on the page turns on when both files are in and verified.

```bash
docker run --gpus all -p 8765:8765 -v r2d2-data:/data ghcr.io/deadjoe/r2d2:latest
```

Needs an NVIDIA driver with CUDA 12.8 (R570 or newer) and the NVIDIA Container
Toolkit. Open <http://localhost:8765> once the log says `ready`. The start script
(`deploy/docker/r2d2-start.sh`) checks the GPU, fetches the weights, starts the
app and loads the recogniser, and stays up with a clear error if a step fails.

| Variable | Purpose |
| --- | --- |
| `R2D2_ACCESS_KEY` | Set it whenever the port is reachable from other machines |
| `R2D2_MODEL_SETS` | Sets fetched before the app starts; `default` (Q8_0 + Hy-MT2) |
| `R2D2_EXTRA_SETS` | Sets fetched in the background once the app is ready; `f16`, empty for none, `f16,q4` adds Q4_K_M |
| `R2D2_PROGRESS_URL` / `R2D2_PROGRESS_TOKEN` | Optional: each start step is POSTed there as JSON |
| `PUBLIC_KEY` | Optional: an SSH public key; sshd then runs (RunPod sets it) |

On RunPod: create a pod from the image with `8765/http` exposed, at least 20 GB of
container disk and `R2D2_ACCESS_KEY` set, then open
`https://<pod>-8765.proxy.runpod.net/?k=<key>`. A personal one-tap launcher that
does this and deletes the pod after a time limit is in
[deadjoe/r2d2_pod](https://github.com/deadjoe/r2d2_pod). The image is built by
`.github/workflows/image.yml` on version tags.

## How it works

**Recognition.** A port of the official `streaming_transcribe_no_reset`
policy: each 160 ms step re-encodes the audio window, continues from the
confirmed text and holds back the last token. Confirmed text (white) is only
ever appended to; the unconfirmed tail is grey. Measured departures from
upstream: an 8 s window instead of 16 s, merging up to three chunks when
behind, a doubled token budget for Korean and kana as well as Han, and no
VAD. See [docs/streaming.md](docs/streaming.md).

**Engines.** GGUF F16, Q8_0 and Q4_K_M run through llama.cpp (Metal on a Mac,
CUDA on Linux); MLX BF16 runs through mlx-audio on a Mac. Q8_0 is the
default: in far-field tests it matched F16 and had the best step timing;
Q4_K_M was clearly worse. See [docs/validation.md](docs/validation.md).

**Languages.** Chinese, English, Japanese, Korean, Spanish or auto-detect,
with an optional hotword and topic hint.

**Translation.** Into Chinese (default), English, Japanese, Korean or Spanish.
A closed sentence is translated once and never changes; the open sentence is
re-translated as a grey draft. A target equal to the spoken language means
recognition only. On a Mac, Hy-MT2 runs on the CPU because on the GPU it
slowed recognition; on Linux it runs on the GPU, 5-10 times faster without
affecting recognition accuracy. See [docs/translation.md](docs/translation.md).

## Limits

- One session at a time, up to 5 minutes; imported audio up to 100 MB.
- A session stops if recognition falls more than 10 s behind.
- Audio goes only to the recognition service you run (this machine, or the GPU
  host you deployed it to) and is never written to disk. Inference processes
  listen only on loopback.

## Third-party models and code

| Project | Used for | License |
| --- | --- | --- |
| [Confucius4-R2T2](https://huggingface.co/netease-youdao/Confucius4-R2T2) ([GGUF](https://huggingface.co/netease-youdao/Confucius4-R2T2-GGUF)), NetEase Youdao | Speech recognition model and tokenizer | NetEase model license, shipped with the weights |
| [Confucius4-R2T2 source](https://github.com/netease-youdao/Confucius4-R2T2) | Streaming policy adapted in `r2d2/streaming.py`; sample audio in `tests/fixtures/` | Apache-2.0 |
| [Hy-MT2-1.8B-GGUF](https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF), Tencent | Translation model | Apache-2.0 |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | GGUF inference (`llama-server`) | MIT |
| [mlx-audio](https://github.com/Blaizzy/mlx-audio) / [MLX](https://github.com/ml-explore/mlx) | MLX inference | MIT |
| [Transformers](https://github.com/huggingface/transformers) | Tokenizer | Apache-2.0 |
| [FastAPI](https://github.com/fastapi/fastapi), [Uvicorn](https://github.com/encode/uvicorn), [HTTPX](https://github.com/encode/httpx) | Web server and internal HTTP | MIT, BSD-3-Clause, BSD-3-Clause |

Adapted code and its provenance are in
[third_party/NOTICE.md](third_party/NOTICE.md). Model weights are not part of
this repository and keep their own licenses. This project is licensed under
the [Apache License 2.0](LICENSE).
