#!/usr/bin/env bash
# run.sh — Canonical Chalie launcher (dev, installer, Docker)
#
# Handles venv resolution, incremental dep sync, then hands off to run.py.
# Any context that already has deps installed (Docker, activated venv) skips
# the sync step automatically.
#
# Usage:
#   ./run.sh                          # start on default port 8081
#   ./run.sh --port=9000              # custom port
#   ./run.sh --host=127.0.0.1         # bind to specific address
#   ./run.sh --no-voice               # skip voice dep sync
#   CHALIE_VENV=~/.chalie/venv ./run.sh   # explicit venv (set by installer CLI)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ─── Arg Parsing ─────────────────────────────────────────────────────────────
_PORT=8081
_HOST="0.0.0.0"
_VOICE=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port=*)   _PORT="${1#--port=}"; shift ;;
    --port)     _PORT="$2"; shift 2 ;;
    --host=*)   _HOST="${1#--host=}"; shift ;;
    --host)     _HOST="$2"; shift 2 ;;
    --no-voice) _VOICE=false; shift ;;
    *) shift ;;
  esac
done

# ─── Python + Pip Resolution ─────────────────────────────────────────────────
# Priority:
#   1. Already in an activated venv (VIRTUAL_ENV is set)
#   2. CHALIE_VENV env var — set by the installed `chalie` CLI wrapper
#   3. Docker (/.dockerenv) — deps baked in at build time, use system Python
#   4. .venv/ in repo root — local dev venv (already in .gitignore)
#   5. ~/.chalie/venv — installed user running from a source clone
#   6. None found — create .venv/ in repo root

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  PYTHON="$VIRTUAL_ENV/bin/python"
  PIP="$VIRTUAL_ENV/bin/pip"
  _IN_DOCKER=false
elif [[ -n "${CHALIE_VENV:-}" ]] && [[ -d "$CHALIE_VENV" ]]; then
  PYTHON="$CHALIE_VENV/bin/python"
  PIP="$CHALIE_VENV/bin/pip"
  _IN_DOCKER=false
elif [[ -f "/.dockerenv" ]]; then
  PYTHON="$(command -v python3)"
  PIP="$(command -v pip3)"
  _IN_DOCKER=true
elif [[ -d "$SCRIPT_DIR/.venv" ]]; then
  PYTHON="$SCRIPT_DIR/.venv/bin/python"
  PIP="$SCRIPT_DIR/.venv/bin/pip"
  _IN_DOCKER=false
elif [[ -d "$HOME/.chalie/venv" ]]; then
  PYTHON="$HOME/.chalie/venv/bin/python"
  PIP="$HOME/.chalie/venv/bin/pip"
  _IN_DOCKER=false
else
  echo "→ No virtual environment found. Creating .venv/ …"
  python3 -m venv "$SCRIPT_DIR/.venv"
  PYTHON="$SCRIPT_DIR/.venv/bin/python"
  PIP="$SCRIPT_DIR/.venv/bin/pip"
  _IN_DOCKER=false
fi

# ─── Incremental Dep Sync ────────────────────────────────────────────────────
# Runs everywhere, including Docker. When backend/ is volume-mounted, the image's
# baked-in packages can drift from the current requirements.txt. The stamp file
# ensures we only run pip when requirements.txt actually changes.

# In Docker the stamp lives in /tmp (writable, ephemeral per container lifecycle)
# so a fresh container always syncs once on first start.
if [[ "$_IN_DOCKER" == "true" ]]; then
  _STAMP_DIR="/tmp"
else
  _STAMP_DIR="$SCRIPT_DIR"
fi

REQ="$SCRIPT_DIR/backend/requirements.txt"
STAMP="$_STAMP_DIR/.deps-installed"

if [[ ! -f "$STAMP" ]] || [[ "$REQ" -nt "$STAMP" ]]; then
  echo "→ Syncing dependencies from requirements.txt …"
  "$PIP" install --quiet -r "$REQ"
  touch "$STAMP"
fi

if [[ "$_VOICE" == "true" ]]; then
  VOICE_REQ="$SCRIPT_DIR/backend/requirements-voice.txt"
  VOICE_STAMP="$_STAMP_DIR/.voice-deps-installed"
  if [[ -f "$VOICE_REQ" ]] && { [[ ! -f "$VOICE_STAMP" ]] || [[ "$VOICE_REQ" -nt "$VOICE_STAMP" ]]; }; then
    echo "→ Syncing voice dependencies …"
    "$PIP" install --quiet -r "$VOICE_REQ" 2>/dev/null \
      || echo "  ⚠ Voice dep install failed — voice will be unavailable"
    touch "$VOICE_STAMP"
  fi
fi

# ─── Launch ──────────────────────────────────────────────────────────────────
exec "$PYTHON" "$SCRIPT_DIR/backend/run.py" --port="$_PORT" --host="$_HOST"
