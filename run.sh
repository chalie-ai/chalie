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
# Supervised restart loop. Exit codes:
#   42  → intentional restart (restart_service.py: snapshot import, in-place
#         update) → re-sync deps and relaunch immediately. Resets the crash
#         counter — a clean restart is never penalised.
#   0   → clean shutdown → exit 0, do not loop.
#   *   → crash / OOM / uncaught exception → relaunch with exponential backoff.
#         After MAX_CRASHES consecutive crashes without a clean run, give up and
#         surface the last exit code (a deterministically-fatal startup loops
#         forever otherwise — log flooding, CPU churn). A run that stays up
#         longer than RUN_HEALTHY_SECS resets the crash counter.
MAX_CRASHES=5
RUN_HEALTHY_SECS=30
_crashes=0
_backoff=1

while true; do
  # set -e would kill the script on run.py's non-zero exit before we could
  # read it, defeating the restart loop. Capture via `||` (exempt from errexit).
  _start=$SECONDS
  _EXIT=0
  "$PYTHON" "$SCRIPT_DIR/backend/run.py" --port="$_PORT" --host="$_HOST" || _EXIT=$?

  # Intentional restart: re-sync deps, relaunch immediately, reset crash state.
  if [[ "$_EXIT" -eq 42 ]]; then
    echo "→ Restart requested (exit 42). Re-syncing deps and relaunching..."
    _install -e "$SCRIPT_DIR/backend"
    _crashes=0
    _backoff=1
    continue
  fi

  # Clean shutdown: exit without looping.
  if [[ "$_EXIT" -eq 0 ]]; then
    exit 0
  fi

  # Crash: back off and retry, up to the cap. A run that stayed up long enough
  # to be healthy resets the counter — only rapid repeated crashes escalate.
  if (( SECONDS - _start >= RUN_HEALTHY_SECS )); then
    _crashes=0
    _backoff=1
  fi
  _crashes=$((_crashes + 1))
  if [[ "$_crashes" -gt "$MAX_CRASHES" ]]; then
    echo "→ Process crashed ($MAX_CRASHES consecutive times). Giving up (exit $_EXIT)." >&2
    exit "$_EXIT"
  fi
  echo "→ Process crashed (exit $_EXIT). Retry $_crashes/$MAX_CRASHES in ${_backoff}s..." >&2
  sleep "$_backoff"
  _backoff=$((_backoff * 2))
done
