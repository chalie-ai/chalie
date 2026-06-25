#!/usr/bin/env bash
# run.sh — Canonical Chalie launcher (dev, installer, Docker)
#
# Resolves a Python interpreter, syncs core deps, then hands off to run.py.
# Voice and playwright are managed at runtime by RuntimeDepsService.
#
# Usage:
#   ./run.sh                          # start on default port 31025
#   ./run.sh --port=9000              # custom port
#   ./run.sh --host=127.0.0.1         # bind to specific address
#   CHALIE_VENV=~/.chalie/venv ./run.sh   # explicit venv (set by installer CLI)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure ~/.local/bin is in PATH — uv installs there and non-login shells
# (e.g. docker exec) won't have it via .bashrc.
export PATH="$HOME/.local/bin:$PATH"

# ─── Arg Parsing ─────────────────────────────────────────────────────────────
_PORT=31025
_HOST="0.0.0.0"

while [[ $# -gt 0 ]]; do
  _arg="$1"
  case "$_arg" in
    --port=*)   _PORT="${_arg#--port=}"; shift ;;
    --port)     _PORT="$2"; shift 2 ;;
    --host=*)   _HOST="${_arg#--host=}"; shift ;;
    --host)     _HOST="$2"; shift 2 ;;
    *) shift ;;
  esac
done

# ─── Python Resolution ───────────────────────────────────────────────────────
# Priority:
#   1. Already in an activated venv (VIRTUAL_ENV is set)
#   2. CHALIE_VENV env var — set by the installed `chalie` CLI wrapper
#   3. ~/.chalie/venv — installed user running from a source clone
#   4. System python3 — local dev (deps installed via `pip install --user`)

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  PYTHON="$VIRTUAL_ENV/bin/python"
elif [[ -n "${CHALIE_VENV:-}" ]] && [[ -d "$CHALIE_VENV" ]]; then
  PYTHON="$CHALIE_VENV/bin/python"
elif [[ -d "$HOME/.chalie/venv" ]]; then
  PYTHON="$HOME/.chalie/venv/bin/python"
else
  PYTHON="$(command -v python3)"
fi

# ─── Dep Sync ────────────────────────────────────────────────────────────────
# Syncs core deps only. Voice/playwright are runtime-managed (RuntimeDepsService).
# uv is instant (~50ms) when deps are already satisfied.

if command -v uv >/dev/null 2>&1; then
  _install() { uv pip install --python "$PYTHON" "$@"; }
else
  _install() { "$PYTHON" -m pip install --user "$@"; }
fi

_install -e "$SCRIPT_DIR/backend"

# ─── Launch ──────────────────────────────────────────────────────────────────
# Loop: Python exits with code 42 to request a restart (e.g. after in-place update).
# Any other exit code passes through normally.
#
# Signal forwarding: when run as PID 1 (e.g. under Docker), bash does not forward
# SIGTERM to its children, so the graceful-shutdown handlers installed in
# consumer.py would never fire and `docker stop` would SIGKILL after the grace
# period. The trap below propagates SIGTERM to the active Python child so its
# own handler can drain the WAL write-queue and stop workers cleanly.
_CHILD_PID=""
_forward_term() {
  [[ -n "$_CHILD_PID" ]] && kill -TERM "$_CHILD_PID" 2>/dev/null || true
}
trap _forward_term TERM INT

while true; do
  # set -e would kill the script on run.py's non-zero exit before we could
  # read it, defeating the restart loop. Capture via `||` (exempt from errexit).
  _EXIT=0
  "$PYTHON" "$SCRIPT_DIR/backend/run.py" --port="$_PORT" --host="$_HOST" &
  _CHILD_PID=$!
  wait "$_CHILD_PID" || _EXIT=$?
  _CHILD_PID=""
  if [[ "$_EXIT" -ne 42 ]]; then
    exit $_EXIT
  fi
  echo "→ Restart requested (exit 42). Re-syncing deps and relaunching..."
  _install -e "$SCRIPT_DIR/backend"
done
