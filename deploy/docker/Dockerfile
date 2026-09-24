# R2D2 // Listening Room — Linux + NVIDIA CUDA image.
#
# The app, its Python environment and a CUDA build of llama-server; no model weights.
# The weights keep their own licenses and are downloaded at first start into /data,
# from pinned revisions, and checked against models-manifest.json (python -m r2d2.fetch).
#
#   docker build -f deploy/docker/Dockerfile -t r2d2 .
#   docker run --gpus all -p 8765:8765 -v r2d2-data:/data ghcr.io/deadjoe/r2d2:latest
#
# Needs an NVIDIA driver that supports CUDA 12.8 (R570 or newer). llama-server is built
# for every GPU generation from Turing (T4, RTX 20) to Blackwell (RTX 50, RTX PRO), plus
# PTX so a newer card can still JIT-compile the kernels.

ARG CUDA_VERSION=12.8.1
ARG UBUNTU=ubuntu24.04

# 1. llama-server, built once per llama.cpp commit and cached on its own.
FROM nvidia/cuda:${CUDA_VERSION}-devel-${UBUNTU} AS llama
# The commit this app is verified with (README).
ARG LLAMA_CPP_COMMIT=b29c606e2
# Turing, Ampere (A100 / A-series & RTX 30), Ada (L4, L40S, RTX 40), Hopper, Blackwell
# datacenter and Blackwell RTX, plus PTX for what comes next.
ARG CUDA_ARCHITECTURES="75-real;80-real;86-real;89-real;90-real;100-real;120-real;120-virtual"
RUN apt-get update \
 && apt-get install -y --no-install-recommends git cmake build-essential ca-certificates \
 && rm -rf /var/lib/apt/lists/*
RUN git clone --filter=blob:none https://github.com/ggml-org/llama.cpp /src \
 && git -C /src checkout "${LLAMA_CPP_COMMIT}"
WORKDIR /src
# Static llama/ggml libraries: one binary plus the CUDA runtime libraries copied below.
# GGML_NATIVE=OFF: the build machine's CPU says nothing about the one it will run on.
RUN cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=OFF \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
      -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release \
 && cmake --build build -j"$(nproc)" --target llama-server \
 && mkdir -p /out \
 && cp build/bin/llama-server /out/ \
 && cp -P /usr/local/cuda/lib64/libcudart.so.12* /usr/local/cuda/lib64/libcublas.so.12* \
          /usr/local/cuda/lib64/libcublasLt.so.12* /out/ \
 && git -C /src rev-parse HEAD > /out/LLAMA_CPP_COMMIT

# 2. The app.
FROM nvidia/cuda:${CUDA_VERSION}-base-${UBUNTU}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# openssh-server only runs when a public key is provided (RunPod passes PUBLIC_KEY);
# its host keys are made at start, never baked into a public image.
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates curl \
      libgomp1 tini openssh-server \
 && rm -rf /var/lib/apt/lists/* /etc/ssh/ssh_host_* \
 && mkdir -p /run/sshd \
 && sed -i 's/^#\?PasswordAuthentication .*/PasswordAuthentication no/; s/^#\?PermitRootLogin .*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config

COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /usr/local/bin/uv
COPY --from=llama /out/ /opt/llama/
ENV LD_LIBRARY_PATH=/opt/llama \
    R2D2_LLAMA_SERVER=/opt/llama/llama-server
# Every library resolves except the driver's libcuda.so.1, which the NVIDIA runtime mounts.
RUN missing="$(ldd /opt/llama/llama-server | grep 'not found' | grep -v 'libcuda.so.1' || true)" \
 && [ -z "$missing" ] || { echo "unresolved libraries: $missing"; exit 1; }

WORKDIR /app
# Dependencies first, so a code change does not reinstall them.
COPY pyproject.toml uv.lock /app/
RUN uv sync --frozen --no-dev --no-install-project --python /usr/bin/python3

COPY r2d2/ /app/r2d2/
COPY web/ /app/web/
COPY tests/fixtures/official-test.wav /app/tests/fixtures/official-test.wav
COPY models-manifest.json LICENSE README.md /app/
COPY third_party/NOTICE.md /app/third_party/NOTICE.md
COPY deploy/docker/r2d2-start.sh /usr/local/bin/r2d2-start
RUN chmod +x /usr/local/bin/r2d2-start \
 && /app/.venv/bin/python -c "import r2d2.server, r2d2.fetch; print('app ok')"

ENV PATH=/app/.venv/bin:$PATH \
    R2D2_MODELS=/data/models \
    HOST=0.0.0.0 \
    PORT=8765
EXPOSE 8765
VOLUME ["/data"]
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["r2d2-start"]
