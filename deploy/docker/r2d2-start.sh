#!/bin/bash
# Bring R2D2 up inside the container: GPU check → model files → app → recogniser loaded,
# then the optional builds downloaded in the background.
# Each step goes to stdout and, when R2D2_PROGRESS_URL is set, to that URL as JSON
# (bearer R2D2_PROGRESS_TOKEN): {"step": gpu|weights|start|ready|extra, "status": started|done|failed}.
# A failed "extra" leaves the app running; that build just stays unavailable on the page.
set -uo pipefail
cd /app || exit 1
PORT="${PORT:-8765}"
R2D2_PROGRESS_URL="${R2D2_PROGRESS_URL:-}"
R2D2_PROGRESS_TOKEN="${R2D2_PROGRESS_TOKEN:-}"
APP_PID=""

progress() { # step status [message]
  local step="$1" status="$2" message="${3:-}"
  echo "[r2d2-start] $(date -u +%FT%TZ) $step $status $message"
  if [[ -n "$R2D2_PROGRESS_URL" ]]; then
    curl -sS -m 10 -o /dev/null -X POST "$R2D2_PROGRESS_URL" \
      -H "content-type: application/json" \
      ${R2D2_PROGRESS_TOKEN:+-H "authorization: Bearer $R2D2_PROGRESS_TOKEN"} \
      --data "$(python3 -c 'import json,sys; print(json.dumps(dict(zip(("step","status","message","ts"), sys.argv[1:]))))' \
        "$step" "$status" "$message" "$(date -u +%FT%TZ)")" || true
  fi
}

fail() { # step message
  progress "$1" failed "$2"
  [[ -n "$APP_PID" ]] && kill "$APP_PID" 2>/dev/null
  # On RunPod an exited container is restarted in a loop; stay up (with sshd) so the
  # logs can be read, and let whoever created the pod delete it. Elsewhere, exit.
  if [[ -n "${RUNPOD_POD_ID:-}" ]]; then sleep infinity; fi
  exit 1
}

trap '[[ -n "$APP_PID" ]] && kill -TERM "$APP_PID" 2>/dev/null; exit 0' TERM INT

# SSH, only when a key is given (RunPod sets PUBLIC_KEY from the account's keys).
if [[ -n "${PUBLIC_KEY:-}" ]]; then
  mkdir -p /root/.ssh && chmod 700 /root/.ssh
  printf '%s\n' "$PUBLIC_KEY" >> /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys
  ssh-keygen -A >/dev/null && /usr/sbin/sshd && echo "[r2d2-start] sshd started"
fi

# 1. GPU and the CUDA build of llama-server.
progress gpu started
GPU=$(nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader 2>&1) \
  || fail gpu "no NVIDIA GPU visible (docker run --gpus all, NVIDIA Container Toolkit): $GPU"
PROBE=$("$R2D2_LLAMA_SERVER" --list-devices 2>&1)
echo "$PROBE" | grep -q "CUDA0" \
  || fail gpu "llama-server sees no CUDA device (driver too old for CUDA 12.8?): $(echo "$PROBE" | tail -3 | tr '\n' ' ')"
progress gpu "done" "$GPU"

# 2. Model files the app starts with, downloaded once into /data and verified against the manifest.
SETS="${R2D2_MODEL_SETS:-default}"
EXTRA_SETS="${R2D2_EXTRA_SETS-f16}"  # set but empty: none
progress weights started "sets: $SETS"
python -m r2d2.fetch --set "$SETS" 2>&1 | tee /tmp/fetch.log
[[ "${PIPESTATUS[0]}" == 0 ]] || fail weights "$(tail -1 /tmp/fetch.log)"
progress weights "done" "$(tail -1 /tmp/fetch.log | sed 's/^\[fetch\] //')"

# 3. The app, then the recogniser loaded so the first recording does not wait for it.
progress start started "port $PORT"
python -m r2d2.server &
APP_PID=$!
COOKIE=()
[[ -n "${R2D2_ACCESS_KEY:-}" ]] && COOKIE=(-H "cookie: r2d2_key=$R2D2_ACCESS_KEY")
for _ in $(seq 1 60); do
  kill -0 "$APP_PID" 2>/dev/null || fail start "the app exited during startup"
  curl -sf -o /dev/null "${COOKIE[@]}" "http://127.0.0.1:$PORT/api/status" && break
  sleep 1
done
curl -sf -o /dev/null "${COOKIE[@]}" "http://127.0.0.1:$PORT/api/status" \
  || fail start "the app did not answer on port $PORT within 60 s"
LOADED=$(curl -s -m 300 "${COOKIE[@]}" -H "content-type: application/json" \
  -d '{"backend":"gguf_q8"}' "http://127.0.0.1:$PORT/api/backend")
echo "$LOADED" | grep -q '"state": *"ready"' || fail start "the recogniser did not load: ${LOADED:0:300}"

URL="http://localhost:$PORT"
[[ -n "${RUNPOD_POD_ID:-}" ]] && URL="https://${RUNPOD_POD_ID}-${PORT}.proxy.runpod.net"
progress ready "done" "$URL"

# 4. Optional builds while the app runs. r2d2.fetch gives a file its real name only once
# verified, and the page offers a build once all its files exist.
if [[ -n "$EXTRA_SETS" ]]; then
  (
    progress extra started "sets: $EXTRA_SETS"
    python -m r2d2.fetch --set "$EXTRA_SETS" 2>&1 | tee /tmp/fetch-extra.log
    if [[ "${PIPESTATUS[0]}" == 0 ]]; then
      progress extra "done" "$(tail -1 /tmp/fetch-extra.log | sed 's/^\[fetch\] //')"
    else
      progress extra failed "$(tail -1 /tmp/fetch-extra.log)"
    fi
  ) &
fi

wait "$APP_PID"
fail start "the app exited (code $?)"
