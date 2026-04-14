#!/usr/bin/env bash
set -euo pipefail

BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
START_QGROUNDCONTROL="${START_QGROUNDCONTROL:-1}"
WAIT_TIMEOUT_SEC="${WAIT_TIMEOUT_SEC:-25}"

test_port_open() {
  local host="$1"
  local port="$2"
  timeout 1 bash -c "</dev/tcp/${host}/${port}" >/dev/null 2>&1
}

start_qgroundcontrol() {
  if [[ "$START_QGROUNDCONTROL" != "1" ]]; then
    echo "[start_full_linux] QGroundControl launch skipped (START_QGROUNDCONTROL=0)."
    return
  fi

  if pgrep -f -i "qgroundcontrol" >/dev/null 2>&1; then
    echo "[start_full_linux] QGroundControl already running."
    return
  fi

  local candidates=(
    "/usr/bin/QGroundControl.AppImage"
    "/usr/local/bin/QGroundControl.AppImage"
    "${HOME}/AppImages/QGroundControl.AppImage"
    "${HOME}/Applications/QGroundControl.AppImage"
  )

  for exe in "${candidates[@]}"; do
    if [[ -x "$exe" ]]; then
      nohup "$exe" >/dev/null 2>&1 &
      echo "[start_full_linux] Started QGroundControl: $exe"
      return
    fi
  done

  if command -v QGroundControl >/dev/null 2>&1; then
    nohup QGroundControl >/dev/null 2>&1 &
    echo "[start_full_linux] Started QGroundControl from PATH."
    return
  fi

  if command -v QGroundControl.AppImage >/dev/null 2>&1; then
    nohup QGroundControl.AppImage >/dev/null 2>&1 &
    echo "[start_full_linux] Started QGroundControl.AppImage from PATH."
    return
  fi

  echo "[start_full_linux] QGroundControl not found. Install it or set START_QGROUNDCONTROL=0."
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

VENV_PYTHON="${REPO_ROOT}/.venv-web/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "[start_full_linux] Creating .venv-web..."
  python3 -m venv .venv-web
  "$VENV_PYTHON" -m pip install --upgrade pip
  "$VENV_PYTHON" -m pip install -r backend/requirements.txt
fi

if [[ ! -x "${REPO_ROOT}/frontend/node_modules/.bin/vite" ]]; then
  echo "[start_full_linux] Installing frontend dependencies..."
  npm --prefix frontend install
fi

LOG_DIR="${REPO_ROOT}/.runlogs"
mkdir -p "$LOG_DIR"

if test_port_open "127.0.0.1" "$BACKEND_PORT"; then
  echo "[start_full_linux] Backend already running on :$BACKEND_PORT"
else
  echo "[start_full_linux] Starting backend on :$BACKEND_PORT"
  nohup "$VENV_PYTHON" -m uvicorn backend.app.main:app --host 127.0.0.1 --port "$BACKEND_PORT" \
    >"${LOG_DIR}/backend.out.log" 2>"${LOG_DIR}/backend.err.log" &
fi

if test_port_open "127.0.0.1" "$FRONTEND_PORT"; then
  echo "[start_full_linux] Frontend already running on :$FRONTEND_PORT"
else
  echo "[start_full_linux] Starting frontend on :$FRONTEND_PORT"
  (
    cd "${REPO_ROOT}/frontend"
    nohup npm run dev -- --host 127.0.0.1 --port "$FRONTEND_PORT" \
      >"${LOG_DIR}/frontend.out.log" 2>"${LOG_DIR}/frontend.err.log" &
  )
fi

start_qgroundcontrol

backend_up=0
frontend_up=0
for ((i = 0; i < WAIT_TIMEOUT_SEC; i++)); do
  if test_port_open "127.0.0.1" "$BACKEND_PORT"; then
    backend_up=1
  fi
  if test_port_open "127.0.0.1" "$FRONTEND_PORT"; then
    frontend_up=1
  fi
  if [[ "$backend_up" -eq 1 && "$frontend_up" -eq 1 ]]; then
    break
  fi
  sleep 1
done

echo "[start_full_linux] Backend listening: $([[ "$backend_up" -eq 1 ]] && echo true || echo false) (http://127.0.0.1:${BACKEND_PORT})"
echo "[start_full_linux] Frontend listening: $([[ "$frontend_up" -eq 1 ]] && echo true || echo false) (http://127.0.0.1:${FRONTEND_PORT})"
if [[ "$backend_up" -ne 1 || "$frontend_up" -ne 1 ]]; then
  echo "[start_full_linux] One or more services failed to start." >&2
  exit 1
fi
