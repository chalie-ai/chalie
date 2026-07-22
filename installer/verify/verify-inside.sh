#!/usr/bin/env bash
# Runs INSIDE one clean distro container and exercises the real install
# experience end to end for that distro:
#   1. run the branch installer exactly as a user would (curl | bash equivalent),
#   2. audit that every dependency actually resolved (onnxruntime import +
#      execution providers, core modules, installer-managed artifacts on disk),
#   3. optionally boot Chalie and wait for the /ready readiness contract to go
#      green (all preflight components ok).
#
# It never aborts on failure — the whole point is to OBSERVE where an install
# breaks per distro — and it emits machine-parseable `RESULT key=value` lines
# plus a final `VERDICT=<...>` line that the host driver (run-matrix.sh) collects.
#
# Inputs (env):
#   CHALIE_BRANCH  branch ref to install (self-fetched by the installer)
#   ROW_NAME       label for this matrix row (for the log)
#   ROW_EXPECT     pass | refuse-old-python | gap
#   DO_BOOT        1 to boot + poll /ready after the audit, 0 to stop at audit
# Mounted read-only at /verify/install.sh: the branch's installer/install.sh.
set -uo pipefail   # deliberately NOT -e: observe failures, don't abort on them.

BRANCH="${CHALIE_BRANCH:?CHALIE_BRANCH required}"
EXPECT="${ROW_EXPECT:-pass}"
DO_BOOT="${DO_BOOT:-0}"
NAME="${ROW_NAME:-unknown}"
INSTALLER="/verify/install.sh"
PORT="31025"

say()    { printf '\n=== %s ===\n' "$*"; }
result() { printf 'RESULT %s\n' "$*"; }

# The installer checks for Python 3.11+ but deliberately never installs it — a
# real machine is expected to already have python3. Official distro base images
# are more stripped than a real install (Debian/Ubuntu bases ship no python3 at
# all), so reproduce a realistic system by installing the distro's STOCK python3
# (no version bump) before handing off to the installer. The resulting version
# is whatever that distro ships — which is exactly what decides pass vs refuse.
_ensure_stock_python() {
  if command -v python3 >/dev/null 2>&1; then
    echo "stock python3 already present: $(python3 --version 2>&1)"
    return 0
  fi
  echo "installing distro stock python3 (prerequisite the installer does not provide)…"
  if   command -v apt-get >/dev/null 2>&1; then apt-get update -qq && apt-get install -y python3
  elif command -v dnf     >/dev/null 2>&1; then dnf install -y python3
  elif command -v pacman  >/dev/null 2>&1; then pacman -Sy --noconfirm python
  elif command -v zypper  >/dev/null 2>&1; then zypper --non-interactive install python3
  elif command -v apk     >/dev/null 2>&1; then apk add python3
  fi
  if command -v python3 >/dev/null 2>&1; then
    echo "stock python3 now: $(python3 --version 2>&1)"
    return 0
  fi
  echo "stock python3 STILL-ABSENT after install attempt (package manager could not run here)"
  return 1
}

# ── Environment facts ────────────────────────────────────────────────────────
say "environment ($NAME)"
# shellcheck disable=SC1091
. /etc/os-release 2>/dev/null || true
echo "distro-id=${ID:-?} version=${VERSION_ID:-?} id-like=${ID_LIKE:-}"
echo "arch=$(uname -m)"

say "prerequisite: distro stock python3"
# A refuse-old-python row deliberately targets a distro whose stock python is too
# old — establishing that stock python IS the finding. Any other row that cannot
# even establish stock python (e.g. a package manager that will not run under CPU
# emulation) is an environment problem, not an installer finding: report it as
# INCONCLUSIVE so it neither masquerades as a real gap nor fails the run. On a
# native-arch runner the package manager succeeds and this branch never fires.
if ! _ensure_stock_python && [ "$EXPECT" != "refuse-old-python" ]; then
  result "reason=stock-python-prereq-unavailable(environment/emulation, not an installer finding)"
  echo "VERDICT=INCONCLUSIVE $NAME"
  exit 0
fi
py_ver="$(python3 --version 2>&1 || echo none)"
echo "python3=$py_ver  bash=$(command -v bash || echo ABSENT)  curl=$(command -v curl || echo ABSENT)"
result "python=$py_ver"

# ── Run the installer, faithfully and non-interactively ──────────────────────
say "installer (--branch=$BRANCH)"
inst_log="/tmp/install.log"
TO=""; command -v timeout >/dev/null 2>&1 && TO="timeout 1800"
$TO bash "$INSTALLER" --branch="$BRANCH" 2>&1 | tee "$inst_log"
inst_rc="${PIPESTATUS[0]}"
result "installer_rc=$inst_rc"

# ── Expectation: installer should REFUSE (default python < 3.11) ─────────────
if [ "$EXPECT" = "refuse-old-python" ]; then
  if [ "$inst_rc" -ne 0 ] && grep -q "Python 3.11+ is required" "$inst_log"; then
    result "reason=refused-as-expected(old-python)"
    echo "VERDICT=REFUSED $NAME"
  else
    result "reason=expected-python-refusal-did-not-fire"
    echo "VERDICT=FAIL $NAME"
  fi
  exit 0
fi

# ── Dependency audit ─────────────────────────────────────────────────────────
VENV="$HOME/.chalie/venv"
APP="$HOME/.chalie/app"
ort_ok=0
ready=0
if [ -x "$VENV/bin/python" ]; then
  say "dependency audit — python imports"
  "$VENV/bin/python" - <<'PY'
import sys
hard_ok = True
try:
    import onnxruntime as ort
    print("onnxruntime", ort.__version__, ort.get_available_providers())
except Exception as e:
    print("ONNXRUNTIME-IMPORT-FAIL:", repr(e)); hard_ok = False
for m in ("onnx", "numpy", "transformers", "soundfile", "playwright"):
    try:
        __import__(m); print("import-ok", m)
    except Exception as e:
        print("import-FAIL", m, repr(e)); hard_ok = False
# Voice packages: reported, not fatal to the verdict (upstream module names vary);
# the /ready boot check is the real proof the voice stack imports at runtime.
for m in ("kokoro_onnx", "moonshine_onnx"):
    try:
        __import__(m); print("voice-import-ok", m)
    except Exception as e:
        print("voice-import-note", m, repr(e))
sys.exit(0 if hard_ok else 3)
PY
  audit_rc=$?
  [ "$audit_rc" -eq 0 ] && ort_ok=1
  result "dep_audit_rc=$audit_rc"

  say "dependency audit — installer-managed artifacts on disk"
  vm="$APP/resources/voice-models"
  for f in kokoro/kokoro-v1.0.onnx kokoro/voices-v1.0.bin \
           moonshine/base/encoder_model.onnx moonshine/base/decoder_model_merged.onnx \
           silero_vad.onnx; do
    if [ -s "$vm/$f" ]; then
      echo "voice-model-ok   $f ($(stat -c%s "$vm/$f" 2>/dev/null || stat -f%z "$vm/$f" 2>/dev/null) bytes)"
    else
      echo "voice-model-MISS $f"
    fi
  done
  deno_v="$("$HOME/.local/bin/deno" --version 2>/dev/null | head -1 || echo ABSENT)"
  echo "deno=$deno_v"
  [ -d "$HOME/.cache/ms-playwright" ] && echo "playwright-browsers=present" || echo "playwright-browsers=ABSENT"
  cli_v="$("$HOME/.local/bin/chalie" version 2>/dev/null || echo ABSENT)"
  echo "chalie-cli=$cli_v"
else
  say "dependency audit — SKIPPED (no venv; installer did not reach it)"
  result "dep_audit_rc=skipped"
fi

# ── Boot + /ready (only where a green boot is the claim) ─────────────────────
if [ "$DO_BOOT" = "1" ] && [ "$ort_ok" = "1" ]; then
  say "boot + /ready (first boot downloads the embedding model, ~400 MB)"
  "$HOME/.local/bin/chalie" start
  for i in $(seq 1 60); do
    # No -f: let curl report the real HTTP status (503 while preflight warms up)
    # rather than erroring out. -w always prints a code — "000" when the port is
    # not yet listening — so a trailing `|| echo 000` would double-print it.
    code="$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/ready" 2>/dev/null)"
    [ -n "$code" ] || code="000"
    echo "poll $i: /ready -> $code"
    if [ "$code" = "200" ]; then ready=1; break; fi
    sleep 8
  done
  result "ready_http=$([ "$ready" = "1" ] && echo 200 || echo timeout)"
  if [ "$ready" != "1" ]; then
    echo "--- chalie.log (tail) ---"; tail -n 60 "$HOME/.chalie/chalie.log" 2>/dev/null
  fi
fi

# ── Verdict ──────────────────────────────────────────────────────────────────
say "verdict"
if [ "$EXPECT" = "pass" ]; then
  if [ "$inst_rc" = "0" ] && [ "$ort_ok" = "1" ] && { [ "$DO_BOOT" != "1" ] || [ "$ready" = "1" ]; }; then
    echo "VERDICT=PASS $NAME"
  else
    echo "VERDICT=FAIL $NAME"
  fi
else   # EXPECT=gap — an unhandled distro; failure is the documented expectation
  if [ "$inst_rc" = "0" ] && [ "$ort_ok" = "1" ]; then
    echo "VERDICT=UNEXPECTED-PASS $NAME"   # good news: prebuilt wheels covered it
  else
    echo "VERDICT=GAP-CONFIRMED $NAME"
  fi
fi
