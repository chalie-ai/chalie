#!/usr/bin/env bash
# Dev/test entrypoint: builds backend + frontend from the local checkout,
# starts the backend, provisions a master account + LLM provider, then runs
# the Playwright interface suite. Args after "--" are passed to playwright.
#
# Run inside the container (WORKDIR=/workspace). The repo is bind-mounted here,
# so local source changes are picked up on every invocation.
set -euo pipefail

PORT=31025
BASE="http://localhost:${PORT}"
SCRIPT_DIR="/workspace"

# ── 1. Backend venv + deps ───────────────────────────────────────────────────
echo "▶ [1/6] Syncing backend deps (uv)…"
export PATH="/usr/local/bin:$HOME/.local/bin:$PATH"
if [[ ! -f "$SCRIPT_DIR/.venv/bin/activate" ]]; then
  uv venv "$SCRIPT_DIR/.venv"
fi
# shellcheck disable=SC1091
source "$SCRIPT_DIR/.venv/bin/activate"
uv pip install -e "$SCRIPT_DIR/backend"

# ── 2. Frontend build (rebuild dist from local source) ───────────────────────
echo "▶ [2/6] Building frontend dist (pnpm)…"
cd "$SCRIPT_DIR/frontend"
pnpm install
pnpm --filter @chalie/interface build

# ── 3. Playwright browser binaries ───────────────────────────────────────────
echo "▶ [3/6] Installing Playwright Chromium…"
cd "$SCRIPT_DIR/frontend/apps/interface"
npx playwright install chromium

# ── 4. Start backend ─────────────────────────────────────────────────────────
echo "▶ [4/6] Starting backend on :${PORT}…"
cd "$SCRIPT_DIR"
python backend/run.py --port="$PORT" --host=127.0.0.1 &
BACKEND_PID=$!

cleanup() {
  echo "▶ Stopping backend (pid $BACKEND_PID)…"
  kill "$BACKEND_PID" 2>/dev/null || true
  wait "$BACKEND_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Wait for /ready (up to 120s — first boot runs schema convergence + migrations).
echo "  waiting for /ready…"
for i in $(seq 1 120); do
  if curl -sf "${BASE}/ready" >/dev/null 2>&1; then
    echo "  backend ready (after ${i}s)"
    break
  fi
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo "✗ Backend process died during startup." >&2
    exit 1
  fi
  sleep 1
  if [[ $i -eq 120 ]]; then
    echo "✗ Backend did not become ready within 120s." >&2
    exit 1
  fi
done

# ── 5. Provision account + provider ──────────────────────────────────────────
echo "▶ [5/6] Provisioning account + provider…"
COOKIE_JAR="$(mktemp)"
USERNAME="${CHALIE_TEST_USERNAME:-admin}"
PASSWORD="${CHALIE_TEST_PASSWORD:?CHALIE_TEST_PASSWORD must be set}"
MM_KEY="${MINIMAX_API_KEY:?MINIMAX_API_KEY must be set}"

# Register (fresh DB) or login (existing account). 201 = registered, 200 = login.
REG_RESP=$(curl -s -w "\n%{http_code}" -c "$COOKIE_JAR" -X POST "${BASE}/auth/register" \
  -H "Content-Type: application/json" \
  -d "{\"username\":\"$USERNAME\",\"password\":\"$PASSWORD\"}")
REG_CODE="$(echo "$REG_RESP" | tail -1)"
if [[ "$REG_CODE" == "409" ]]; then
  echo "  account exists — logging in…"
  curl -sf -b "$COOKIE_JAR" -c "$COOKIE_JAR" -X POST "${BASE}/auth/login" \
    -H "Content-Type: application/json" \
    -d "{\"username\":\"$USERNAME\",\"password\":\"$PASSWORD\"}" >/dev/null
elif [[ "$REG_CODE" != "201" ]]; then
  echo "✗ Register/login failed (HTTP $REG_CODE): $REG_RESP" >&2
  exit 1
fi

# Create the MiniMax provider (idempotent — tolerate 409 name conflict).
# MiniMax IO is NOT a "minimax_io" platform; it's openai_compatible + the host.
PROV_RESP=$(curl -s -w "\n%{http_code}" -b "$COOKIE_JAR" -X POST "${BASE}/providers" \
  -H "Content-Type: application/json" \
  -d "{\"name\":\"MiniMax\",\"platform\":\"openai_compatible\",\"model\":\"MiniMax-M2\",\"host\":\"https://api.minimax.io/v1\",\"api_key\":\"$MM_KEY\"}")
PROV_CODE="$(echo "$PROV_RESP" | tail -1)"
if [[ "$PROV_CODE" == "201" ]]; then
  echo "  provider created + auto-selected."
elif [[ "$PROV_CODE" == "409" ]]; then
  echo "  provider already exists — reusing."
else
  echo "✗ Provider creation failed (HTTP $PROV_CODE): $PROV_RESP" >&2
  exit 1
fi

# ── 6. Run Playwright ────────────────────────────────────────────────────────
echo "▶ [6/6] Running Playwright suite…"
cd "$SCRIPT_DIR/frontend/apps/interface"
export CHALIE_BASE_URL="$BASE"
export CHALIE_TEST_USERNAME="$USERNAME"
export CHALIE_TEST_PASSWORD="$PASSWORD"

# Pass through any args after "--" to playwright; default to the full suite.
if [[ $# -gt 0 ]]; then
  npx playwright test "$@"
else
  npx playwright test
fi
