#!/usr/bin/env bash
set -euo pipefail

MODE_RAW="${1:-${RUNTIME_MODE:-simulation}}"
case "${MODE_RAW}" in
  sim|simulation)
    LAUNCH_MODE="simulation"
    ;;
  real|real_mission|mission)
    LAUNCH_MODE="real_mission"
    ;;
  -h|--help|help)
    cat <<'EOF'
Usage: bash ./scripts/start_full_linux.sh [simulation|real]

simulation: starts backend + frontend + QGroundControl + ArduPilot SITL
real: starts backend + frontend + QGroundControl for a live vehicle
Default SITL path: ~/ardupilot (override with ARDUPILOT_DIR)
EOF
    exit 0
    ;;
  *)
    echo "[start_full_linux] Unknown mode: ${MODE_RAW}" >&2
    echo "[start_full_linux] Use simulation or real." >&2
    exit 2
    ;;
esac

export RUNTIME_MODE="${LAUNCH_MODE}"

BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
WAIT_TIMEOUT_SEC="${WAIT_TIMEOUT_SEC:-25}"
START_QGROUNDCONTROL="${START_QGROUNDCONTROL:-1}"
QGC_APPIMAGE_PATH="${QGC_APPIMAGE_PATH:-${HOME}/Desktop/QGroundControl-x86_64.AppImage}"
CLEAR_VITE_CACHE="${CLEAR_VITE_CACHE:-1}"

if [[ -z "${START_SITL:-}" ]]; then
  if [[ "${LAUNCH_MODE}" == "simulation" ]]; then
    START_SITL=1
  else
    START_SITL=0
  fi
fi

ARDUPILOT_DIR="${ARDUPILOT_DIR:-${HOME}/ardupilot}"
SITL_MODEL="${SITL_MODEL:-ArduCopter}"
SITL_CUSTOM_LOCATION="${SITL_CUSTOM_LOCATION:-32.204407,118.718885,0,0}"
SITL_OUT_PRIMARY="${SITL_OUT_PRIMARY:-udp:127.0.0.1:14550}"
SITL_OUT_SECONDARY="${SITL_OUT_SECONDARY:-udp:127.0.0.1:14551}"
SITL_SESSION_NAME="${SITL_SESSION_NAME:-drone_thesis_sitl}"

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
    "${QGC_APPIMAGE_PATH}"
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

start_sitl() {
  if [[ "$START_SITL" != "1" ]]; then
    echo "[start_full_linux] SITL launch skipped (START_SITL=0)."
    return
  fi

  local sim_vehicle="${ARDUPILOT_DIR}/Tools/autotest/sim_vehicle.py"
  if [[ ! -f "$sim_vehicle" ]]; then
    echo "[start_full_linux] SITL not found at ${sim_vehicle}. Set ARDUPILOT_DIR or START_SITL=0."
    return
  fi

  if screen -ls | grep -q "[.]${SITL_SESSION_NAME}[[:space:]]"; then
    echo "[start_full_linux] SITL already running."
    return
  fi

  echo "[start_full_linux] Starting SITL in screen session: ${SITL_SESSION_NAME}"
  screen -DmS "$SITL_SESSION_NAME" bash -lc "cd \"$ARDUPILOT_DIR\" && Tools/autotest/sim_vehicle.py -v \"$SITL_MODEL\" --console --map --custom-location=\"$SITL_CUSTOM_LOCATION\" --out=\"$SITL_OUT_PRIMARY\" --out=\"$SITL_OUT_SECONDARY\"" \
    >"${LOG_DIR}/sitl.launch.out.log" 2>"${LOG_DIR}/sitl.launch.err.log"
}

ensure_web_venv() {
  local venv_python="${REPO_ROOT}/.venv-web/bin/python"
  if [[ ! -x "$venv_python" ]]; then
    echo "[start_full_linux] Creating .venv-web..."
    python3 -m venv .venv-web
    "$venv_python" -m pip install --upgrade pip
    "$venv_python" -m pip install -r backend/requirements.txt
  fi
}

ensure_frontend_deps() {
  if [[ ! -x "${REPO_ROOT}/frontend/node_modules/.bin/vite" ]]; then
    echo "[start_full_linux] Installing frontend dependencies..."
    npm --prefix frontend install
  fi
}

maybe_clear_vite_cache() {
  if [[ "$CLEAR_VITE_CACHE" != "1" ]]; then
    return
  fi

  local vite_cache_dir="${REPO_ROOT}/frontend/node_modules/.vite"
  if [[ -d "$vite_cache_dir" ]]; then
    echo "[start_full_linux] Clearing stale Vite optimize cache..."
    rm -rf "$vite_cache_dir"
  fi
}

start_backend() {
  local venv_python="${REPO_ROOT}/.venv-web/bin/python"
  if test_port_open "127.0.0.1" "$BACKEND_PORT"; then
    echo "[start_full_linux] Backend already running on :$BACKEND_PORT"
    return
  fi

  echo "[start_full_linux] Starting backend on :$BACKEND_PORT (RUNTIME_MODE=${RUNTIME_MODE})"
  nohup env RUNTIME_MODE="$RUNTIME_MODE" "$venv_python" -m uvicorn backend.app.main:app \
    --host 127.0.0.1 --port "$BACKEND_PORT" \
    >"${LOG_DIR}/backend.out.log" 2>"${LOG_DIR}/backend.err.log" &
}

start_frontend() {
  if test_port_open "127.0.0.1" "$FRONTEND_PORT"; then
    echo "[start_full_linux] Frontend already running on :$FRONTEND_PORT"
    return
  fi

  echo "[start_full_linux] Starting frontend on :$FRONTEND_PORT"
  (
    cd "${REPO_ROOT}/frontend"
    nohup npm run dev -- --host 127.0.0.1 --port "$FRONTEND_PORT" \
      >"${LOG_DIR}/frontend.out.log" 2>"${LOG_DIR}/frontend.err.log" &
  )
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="${REPO_ROOT}/.runlogs"
mkdir -p "$LOG_DIR"

ensure_web_venv
ensure_frontend_deps
maybe_clear_vite_cache

if [[ "${LAUNCH_MODE}" == "simulation" ]]; then
  start_sitl
fi

start_backend
start_frontend
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

echo "[start_full_linux] Mode: ${LAUNCH_MODE}"
echo "[start_full_linux] Backend listening: $([[ "$backend_up" -eq 1 ]] && echo true || echo false) (http://127.0.0.1:${BACKEND_PORT})"
echo "[start_full_linux] Frontend listening: $([[ "$frontend_up" -eq 1 ]] && echo true || echo false) (http://127.0.0.1:${FRONTEND_PORT})"
if [[ "${LAUNCH_MODE}" == "simulation" ]]; then
  echo "[start_full_linux] SITL requested: $([[ "$START_SITL" == "1" ]] && echo true || echo false)"
fi
if [[ "$backend_up" -ne 1 || "$frontend_up" -ne 1 ]]; then
  echo "[start_full_linux] One or more services failed to start." >&2
  exit 1
fi
