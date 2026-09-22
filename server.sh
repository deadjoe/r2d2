#!/bin/bash
# R2D2 listening room service control: start | stop | status | log
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$PWD"

PYTHON="$ROOT/.venv/bin/python"
RUNTIME="$ROOT/.runtime"
PIDFILE="$RUNTIME/server.pid"
PORTFILE="$RUNTIME/server.port"
LOGFILE="$RUNTIME/server.log"
LLAMA_LOG="$RUNTIME/llama-server.log"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8765}"
# Absolute interpreter path so pgrep can tell this checkout apart from another.
MARKER="$PYTHON -m r2d2.server"

hr() { printf '%s\n' "────────────────────────────────────────────────────────────"; }

# Echo the live server PID, or nothing. Never fails, so `set -e` stays happy.
server_pid() {
  local pid
  if [[ -f "$PIDFILE" ]] && pid=$(<"$PIDFILE") && [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    if ps -o command= -p "$pid" 2>/dev/null | grep -qF "r2d2.server"; then
      printf '%s' "$pid"; return
    fi
  fi
  pgrep -f "^$MARKER$" 2>/dev/null | head -1 || true
}

# llama.cpp children are identified by this repo's model path, never by name alone.
llama_pids() { pgrep -f "llama-server.*$ROOT/models" 2>/dev/null || true; }

saved_port() { [[ -f "$PORTFILE" ]] && cat "$PORTFILE" || printf '%s' "$PORT"; }

lan_addrs() { ifconfig 2>/dev/null | awk '/inet /{print $2}' | grep -v '^127\.' || true; }

api() { curl -sf --max-time 3 "http://127.0.0.1:$(saved_port)/api/status" 2>/dev/null || true; }

cmd_start() {
  local pid; pid=$(server_pid)
  if [[ -n "$pid" ]]; then
    echo "服务已在运行（PID ${pid}，端口 $(saved_port)）。要重启请先 ./server.sh stop"
    exit 1
  fi
  [[ -x "$PYTHON" ]] || { echo "缺少 ${PYTHON}，先运行：uv sync"; exit 1; }
  mkdir -p "$RUNTIME"
  { hr; echo "启动 $(date '+%Y-%m-%d %H:%M:%S')  HOST=$HOST PORT=$PORT"; hr; } >> "$LOGFILE"

  HOST="$HOST" PORT="$PORT" nohup $PYTHON -m r2d2.server >> "$LOGFILE" 2>&1 &
  pid=$!
  echo "$pid" > "$PIDFILE"
  echo "$PORT" > "$PORTFILE"

  printf '启动中'
  for _ in $(seq 1 60); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo; echo "启动失败，进程已退出。日志尾部："; hr; tail -20 "$LOGFILE"; rm -f "$PIDFILE"; exit 1
    fi
    if curl -sf --max-time 2 "http://127.0.0.1:$PORT/api/status" >/dev/null 2>&1; then
      echo " 就绪"; break
    fi
    printf '.'; sleep 0.5
  done
  if ! curl -sf --max-time 2 "http://127.0.0.1:$PORT/api/status" >/dev/null 2>&1; then
    echo; echo "启动超时（30 秒内没有响应）。日志尾部："; hr; tail -20 "$LOGFILE"; exit 1
  fi

  local status window
  status=$(api)
  window=$(printf '%s' "$status" | sed -n 's/.*"window_seconds": *\([0-9]*\).*/\1/p')
  echo
  hr
  echo "  R2D2 // LISTENING ROOM        PID $pid"
  hr
  echo
  echo "  本机访问（麦克风可用）"
  echo "      http://localhost:$PORT"
  echo "      http://127.0.0.1:$PORT"
  echo
  if [[ "$HOST" == "0.0.0.0" ]]; then
    local addrs; addrs=$(lan_addrs)
    if [[ -n "$addrs" ]]; then
      echo "  局域网访问（监听 0.0.0.0:${PORT}）"
      while read -r ip; do [[ -n "$ip" ]] && echo "      http://$ip:$PORT"; done <<< "$addrs"
      echo
      echo "      注意：浏览器麦克风只在 localhost 或受信任的 HTTPS 页面可用。"
      echo "      局域网地址能打开页面，但录音会被浏览器拒绝；导入音频和官方示例不受影响。"
    else
      echo "  局域网：监听 0.0.0.0:${PORT}，但没有检测到非回环地址"
    fi
  else
    echo "  仅监听 $HOST:${PORT}，未对局域网开放"
  fi
  echo
  hr
  echo "  流式策略   160 ms 步进 / 160 ms 前瞻 / ${window:-?} s 窗口 / 回退 1 token"
  echo "  模型就绪   $(printf '%s' "$status" | grep -qE '"gguf": *true' && printf 'GGUF F16 ✓' || printf 'GGUF F16 ✗')   $(printf '%s' "$status" | grep -qE '"gguf_q8": *true' && printf 'GGUF Q8 ✓' || printf 'GGUF Q8 ✗')   $(printf '%s' "$status" | grep -qE '"gguf_q4": *true' && printf 'GGUF Q4 ✓' || printf 'GGUF Q4 ✗')   $(printf '%s' "$status" | grep -qE '"mlx": *true' && printf 'MLX ✓' || printf 'MLX ✗')"
  echo "             （模型在按下「开始聆听」时才加载，首次需要等待）"
  echo
  echo "  日志       $LOGFILE"
  echo "  查看日志   ./server.sh log      实时跟随：./server.sh log -f"
  echo "  查看状态   ./server.sh status"
  echo "  停止服务   ./server.sh stop"
  hr
}

cmd_stop() {
  local pid stopped=0
  pid=$(server_pid)
  if [[ -n "$pid" ]]; then
    echo "停止服务 PID $pid …"
    kill -TERM "$pid" 2>/dev/null || true
    # Graceful first: the FastAPI lifespan hook is what terminates llama-server.
    for _ in $(seq 1 30); do
      kill -0 "$pid" 2>/dev/null || { stopped=1; break; }
      sleep 0.5
    done
    if [[ $stopped -eq 0 ]]; then
      echo "  15 秒未退出，强制结束（llama-server 子进程可能残留，下面会清理）"
      kill -KILL "$pid" 2>/dev/null || true
      sleep 1
    else
      echo "  已正常退出"
    fi
  else
    echo "没有找到运行中的服务"
  fi

  # Sweep children that outlived the parent, e.g. after a kill -9 or a crash.
  local orphans; orphans=$(llama_pids)
  if [[ -n "$orphans" ]]; then
    echo "清理残留的 llama-server 子进程：$(printf '%s' "$orphans" | tr '\n' ' ')"
    # shellcheck disable=SC2086
    kill -TERM $orphans 2>/dev/null || true
    sleep 2
    orphans=$(llama_pids)
    if [[ -n "$orphans" ]]; then
      # shellcheck disable=SC2086
      kill -KILL $orphans 2>/dev/null || true
      sleep 1
    fi
  fi

  local port; port=$(saved_port)
  if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "警告：端口 $port 仍被占用："
    lsof -nP -iTCP:"$port" -sTCP:LISTEN | tail -n +2
    exit 1
  fi
  rm -f "$PIDFILE"
  [[ -n "$(llama_pids)" ]] && { echo "警告：仍有 llama-server 残留"; exit 1; } || true
  echo "已停止：端口 $port 已释放，无残留进程"
}

cmd_status() {
  local pid port status
  pid=$(server_pid); port=$(saved_port)
  hr
  if [[ -z "$pid" ]]; then
    echo "  服务状态   未运行"
    local orphans; orphans=$(llama_pids)
    [[ -n "$orphans" ]] && echo "  ⚠ 残留     llama-server $(printf '%s' "$orphans" | tr '\n' ' ')  用 ./server.sh stop 清理"
    hr
    return
  fi
  echo "  服务状态   运行中"
  echo "  PID        $pid   已运行 $(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ')"
  echo "  端口       $port"
  echo "  地址       http://localhost:$port"
  if [[ "$HOST" == "0.0.0.0" ]]; then
    while read -r ip; do [[ -n "$ip" ]] && echo "             http://$ip:$port"; done <<< "$(lan_addrs)"
  fi
  local children; children=$(llama_pids)
  echo "  GGUF 子进程 ${children:-无（未加载或使用 MLX）}"
  status=$(api)
  if [[ -n "$status" ]]; then
    echo "  模型引擎   $(printf '%s' "$status" | sed -n 's/.*"backend": *"\([a-z0-9_]*\)".*/\1/p')"
    echo "  加载状态   $(printf '%s' "$status" | sed -n 's/.*"state": *"\([a-z]*\)".*/\1/p')"
    echo "  是否识别中 $(printf '%s' "$status" | grep -qE '"busy": *true' && echo 是 || echo 否)"
    echo "  音频窗口   $(printf '%s' "$status" | sed -n 's/.*"window_seconds": *\([0-9]*\).*/\1/p') 秒"
  else
    echo "  ⚠ HTTP     进程在，但 /api/status 无响应"
  fi
  echo "  日志       $LOGFILE"
  hr
}

cmd_log() {
  [[ -f "$LOGFILE" ]] || { echo "还没有日志：$LOGFILE"; exit 1; }
  if [[ "${1:-}" == "-f" || "${1:-}" == "follow" ]]; then
    echo "跟随 ${LOGFILE}（Ctrl+C 退出）"; hr
    tail -f "$LOGFILE"
  else
    echo "$LOGFILE 最后 80 行"; hr
    tail -80 "$LOGFILE"
    if [[ -f "$LLAMA_LOG" ]]; then
      echo; hr; echo "$LLAMA_LOG 最后 20 行"; hr; tail -20 "$LLAMA_LOG"
    fi
  fi
}

case "${1:-}" in
  start)  cmd_start ;;
  stop)   cmd_stop ;;
  status) cmd_status ;;
  log)    shift; cmd_log "${1:-}" ;;
  *)
    cat <<USAGE
用法：./server.sh <命令>

  start     后台启动服务，等待就绪后显示访问地址
  stop      停止服务，并清理 llama-server 子进程，确认端口释放
  status    显示运行状态、地址、模型引擎和加载状态
  log       查看日志最后 80 行；./server.sh log -f 实时跟随

环境变量：
  PORT=8766          换端口（默认 8765）
  HOST=127.0.0.1     只监听本机（默认 0.0.0.0，对局域网开放）
  R2D2_MODELS=/path  换模型根目录
USAGE
    exit 1 ;;
esac
