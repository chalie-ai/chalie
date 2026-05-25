#!/usr/bin/env bash
# run.sh — Canonical Chalie launcher (dev, installer, Docker)
#
# Handles venv resolution, dep sync via uv, then hands off to run.py.
#
# Usage:
#   ./run.sh                          # start on default port 31025
#   ./run.sh --port=9000              # custom port
#   ./run.sh --host=127.0.0.1         # bind to specific address
#   ./run.sh --no-voice               # skip voice dep sync
#   CHALIE_VENV=~/.chalie/venv ./run.sh   # explicit venv (set by installer CLI)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ─── Arg Parsing ─────────────────────────────────────────────────────────────
_PORT=31025
_HOST="0.0.0.0"
_VOICE=true

while [[ $# -gt 0 ]]; do
  _arg="$1"
  case "$_arg" in
    --port=*)   _PORT="${_arg#--port=}"; shift ;;
    --port)     _PORT="$2"; shift 2 ;;
    --host=*)   _HOST="${_arg#--host=}"; shift ;;
    --host)     _HOST="$2"; shift 2 ;;
    --no-voice) _VOICE=false; shift ;;
    *) shift ;;
  esac
done

# ─── Python + Venv Resolution ────────────────────────────────────────────────
# Priority:
#   1. Already in an activated venv (VIRTUAL_ENV is set)
#   2. CHALIE_VENV env var — set by the installed `chalie` CLI wrapper
#   3. .venv/ in repo root — local dev venv (already in .gitignore)
#   4. ~/.chalie/venv — installed user running from a source clone
#   5. None found — create .venv/ in repo root

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  PYTHON="$VIRTUAL_ENV/bin/python"
elif [[ -n "${CHALIE_VENV:-}" ]] && [[ -d "$CHALIE_VENV" ]]; then
  PYTHON="$CHALIE_VENV/bin/python"
elif [[ -d "$SCRIPT_DIR/.venv" ]]; then
  PYTHON="$SCRIPT_DIR/.venv/bin/python"
elif [[ -d "$HOME/.chalie/venv" ]]; then
  PYTHON="$HOME/.chalie/venv/bin/python"
else
  echo "→ No virtual environment found. Creating .venv/ …"
  python3 -m venv "$SCRIPT_DIR/.venv"
  PYTHON="$SCRIPT_DIR/.venv/bin/python"
fi

# ─── Dep Sync ────────────────────────────────────────────────────────────────
# uv is instant (~50ms) when deps are already satisfied — no stamp files needed.
# Falls back to pip if uv isn't installed (slower but functional).

if command -v uv >/dev/null 2>&1; then
  _install() { uv pip install --python "$PYTHON" "$@"; }
else
  _install() { "$PYTHON" -m pip install "$@"; }
fi

_install -e "$SCRIPT_DIR/backend"

if [[ "$_VOICE" == "true" ]]; then
  _install -e "$SCRIPT_DIR/backend[voice]" || \
    echo "  ⚠ Voice dep install failed — voice will be unavailable"
fi

# ─── Launch ──────────────────────────────────────────────────────────────────
# Loop: Python exits with code 42 to request a restart (e.g. after in-place update).
# Any other exit code passes through normally.
while true; do
  "$PYTHON" "$SCRIPT_DIR/backend/run.py" --port="$_PORT" --host="$_HOST"
  _EXIT=$?
  if [[ "$_EXIT" -ne 42 ]]; then
    exit $_EXIT
  fi
  echo "→ Restart requested (exit 42). Re-syncing deps and relaunching..."
  _install -e "$SCRIPT_DIR/backend"
done
