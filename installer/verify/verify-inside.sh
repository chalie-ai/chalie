#!/usr/bin/env bash
# Runs INSIDE one clean distro container — or directly on a bare host (the macOS
# CI runner, via INSTALLER_PATH) — and exercises the real install experience end
# to end for that system:
#   1. run the branch installer exactly as a user would (curl | bash equivalent),
#   2. audit that every dependency actually resolved (onnxruntime import +
#      execution providers, core modules, installer-managed artifacts on disk),
#   3. boot Chalie and prove the whole runtime is actually WORKING, not merely
#      started — every subsystem a user touches on day one is exercised for real:
#        readiness  /ready 200 with database, memory_store, embeddings and onnx
#                   each reporting "ok" (embeddings ok == the embedding session
#                   is built; onnx ok == the classifier heads registered)
#        webserver  GET / serves the interface bundle
#        chromium   Playwright drives the live instance and reads its title
#        voice      a real Kokoro synthesis is fed back through Moonshine STT
#                   (which runs Silero VAD on the way) and returns text
#        deno       the bundled runtime executes a script and returns stdout
#      A row is PASS only when every one of those probes passes.
#
# It never aborts on failure — the whole point is to OBSERVE where an install
# breaks per distro — and it emits machine-parseable `RESULT key=value` lines
# plus a final `VERDICT=<...>` line that the host driver (run-matrix.sh) collects.
#
# Inputs (env):
#   CHALIE_BRANCH  branch ref to install (self-fetched by the installer)
#   ROW_NAME       label for this matrix row (for the log)
#   ROW_EXPECT     pass | refuse-old-python
#   INSTALLER_PATH path to the installer (defaults to the container mount at
#                  /verify/install.sh; set it to the checked-out installer on a
#                  bare host where nothing is mounted).
set -uo pipefail   # deliberately NOT -e: observe failures, don't abort on them.

BRANCH="${CHALIE_BRANCH:?CHALIE_BRANCH required}"
EXPECT="${ROW_EXPECT:-pass}"
NAME="${ROW_NAME:-unknown}"
# In a container the installer is mounted at /verify/install.sh; on a bare host
# (the macOS CI runner) point INSTALLER_PATH at the checked-out installer instead.
INSTALLER="${INSTALLER_PATH:-/verify/install.sh}"
PORT="31025"

say()    { printf '\n=== %s ===\n' "$*"; }
result() { printf 'RESULT %s\n' "$*"; }

# The CLI lands in /usr/local/bin on Linux — already on PATH, so `command -v`
# resolving it IS the assertion that the install put it somewhere usable. On
# macOS it lands in ~/.local/bin behind a shell-profile line this script never
# sources, hence the fallback. A Linux run reaching the fallback is a failure,
# which is why the resolved path is printed alongside the version.
_chalie_cli() { command -v chalie 2>/dev/null || echo "$HOME/.local/bin/chalie"; }

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
  # apt and dnf only — the two package managers the installer supports, so the
  # two the matrix has rows for. A distro needing anything else is refused by the
  # installer itself and has no row here.
  if   command -v apt-get >/dev/null 2>&1; then apt-get update -qq && apt-get install -y python3
  elif command -v dnf     >/dev/null 2>&1; then dnf install -y python3
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
# old — establishing that stock python IS the finding. Every row runs on a
# native-arch runner, so any other row that cannot install its distro's stock
# python has a real problem, not an emulation artefact: fail rather than excuse it.
if ! _ensure_stock_python && [ "$EXPECT" != "refuse-old-python" ]; then
  result "reason=stock-python-prereq-unavailable"
  echo "VERDICT=FAIL $NAME"
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
    # Audit the one provider we actually run on. The wheel bundles others
    # (CoreML on macOS) that nothing selects — listing them here would
    # advertise acceleration the runtime never uses.
    if "CPUExecutionProvider" not in ort.get_available_providers():
        print("ONNXRUNTIME-NO-CPU-PROVIDER"); hard_ok = False
    print("onnxruntime", ort.__version__, "CPUExecutionProvider")
except Exception as e:
    print("ONNXRUNTIME-IMPORT-FAIL:", repr(e)); hard_ok = False
for m in ("onnx", "numpy", "transformers", "soundfile", "playwright",
          "kokoro_onnx", "moonshine_onnx"):
    try:
        __import__(m); print("import-ok", m)
    except Exception as e:
        print("import-FAIL", m, repr(e)); hard_ok = False
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
  cli_path="$(_chalie_cli)"
  cli_v="$("$cli_path" version 2>/dev/null || echo ABSENT)"
  echo "chalie-cli=$cli_v (resolved: $cli_path)"

  # Measured, not estimated — this is the number the README's disk requirement
  # has to match, so it is recorded on every run rather than guessed once.
  say "installed footprint on disk"
  du -sh "$HOME/.chalie" "$HOME/.cache/ms-playwright" "$HOME/.local/bin" 2>/dev/null
  du -sc "$HOME/.chalie" "$HOME/.cache/ms-playwright" "$HOME/.local/bin" 2>/dev/null | tail -1
else
  say "dependency audit — SKIPPED (no venv; installer did not reach it)"
  result "dep_audit_rc=skipped"
fi

# ── Boot, then prove every subsystem is actually working ─────────────────────
# Each probe sets its own flag and prints RESULT <probe>=ok|fail. A row passes
# only when all of them are ok — "the process started" is not the claim being
# tested, "a user can use it" is.
web_ok=0; chromium_ok=0; voice_ok=0; deno_ok=0

if [ "$ort_ok" = "1" ]; then
  say "boot"
  "$(_chalie_cli)" start

  # ── readiness: /ready 200 means every preflight component reports ok ───────
  # (database, memory_store, embeddings, onnx — see services/preflight_service.py).
  # The body is printed so a 503 says WHICH component is not ok, and so an "ok"
  # is legible rather than inferred from a bare status code.
  for i in $(seq 1 60); do
    # No -f: let curl report the real HTTP status (503 while preflight warms up)
    # rather than erroring out. -w always prints a code — "000" when the port is
    # not yet listening — so a trailing `|| echo 000` would double-print it.
    body="$(curl -s -w '\n%{http_code}' "http://localhost:$PORT/ready" 2>/dev/null)"
    code="$(printf '%s' "$body" | tail -n1)"
    [ -n "$code" ] || code="000"
    echo "poll $i: /ready -> $code"
    if [ "$code" = "200" ]; then
      ready=1
      echo "  components: $(printf '%s' "$body" | head -n1)"
      break
    fi
    sleep 8
  done
  result "ready_http=$([ "$ready" = "1" ] && echo 200 || echo timeout)"
  if [ "$ready" != "1" ]; then
    echo "--- last /ready body ---"; printf '%s\n' "$body" | head -n1
    echo "--- chalie.log (tail) ---"; tail -n 60 "$HOME/.chalie/chalie.log" 2>/dev/null
  fi
fi

if [ "$ready" = "1" ]; then
  # ── webserver: the interface bundle is actually served, not just the API ───
  say "probe: webserver"
  idx="$(curl -s -w '\n%{http_code}' "http://localhost:$PORT/" 2>/dev/null)"
  idx_code="$(printf '%s' "$idx" | tail -n1)"
  # An index.html that exists but is a stub would still 200, so require the
  # document element too — that is the difference between "a file was returned"
  # and "the built frontend was returned".
  if [ "$idx_code" = "200" ] && printf '%s' "$idx" | grep -qi '<html'; then
    web_ok=1
  else
    echo "GET / -> $idx_code, body head:"; printf '%s' "$idx" | head -c 400
  fi
  result "webserver=$([ "$web_ok" = "1" ] && echo ok || echo fail)"

  # ── chromium: drive the LIVE instance, not an empty browser ───────────────
  # Launching alone proves the shared objects resolved; navigating and reading
  # the title proves the renderer works too, which is what the browsing ability
  # actually needs. --no-sandbox/--disable-dev-shm-usage because this runs as
  # root in a container (Chromium refuses to run as root with the sandbox on).
  say "probe: chromium"
  "$VENV/bin/python" - "$PORT" <<'PY'
import sys
port = sys.argv[1]
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(args=["--no-sandbox", "--disable-dev-shm-usage"])
    page = b.new_page()
    page.goto(f"http://localhost:{port}/", wait_until="domcontentloaded", timeout=60000)
    print("chromium navigated, title=%r" % page.title())
    b.close()
PY
  [ $? -eq 0 ] && chromium_ok=1
  result "chromium=$([ "$chromium_ok" = "1" ] && echo ok || echo fail)"

  # ── voice: a real round trip, not an import check ─────────────────────────
  # Kokoro synthesises speech; that audio is fed straight back through Moonshine
  # STT (which runs Silero VAD on the way). Non-empty text out means the whole
  # on-device voice stack — both models, both ONNX sessions, soundfile, the VAD —
  # is genuinely working. Run from backend/ because it is not an installable
  # package (pyproject sets packages = []); its modules resolve via cwd.
  say "probe: voice (Kokoro TTS → Moonshine STT round trip)"
  ( cd "$APP/backend" && "$VENV/bin/python" - <<'PY'
from services.voice_transcript_service import VoiceTranscriptService
from services.speech_to_text_service import get_service as stt_service

_, sample_rate, wav = VoiceTranscriptService.instance().synthesize(
    "The quick brown fox jumps over the lazy dog."
)
assert wav, "Kokoro produced no audio"
print("kokoro synthesized %d bytes at %d Hz" % (len(wav), sample_rate))

# Same order the transcribe route uses: ensure_loaded() is what reads the model
# files off disk and builds the ONNX session, and it returns False rather than
# raising — so a missing model file must be asserted on, not left to surface as
# a confusing NoneType error inside moonshine.
stt = stt_service()
assert stt.ensure_loaded(), "Moonshine failed to load (missing model files: %r)" % (
    stt.missing_model_files(),
)

text = stt.transcribe(wav)
assert text.strip(), "Moonshine returned empty text for synthesized speech"
print("moonshine transcribed: %r" % text)
PY
  )
  [ $? -eq 0 ] && voice_ok=1
  result "voice=$([ "$voice_ok" = "1" ] && echo ok || echo fail)"

  # ── deno: the bundled runtime executes, which is what code execution needs ─
  say "probe: deno"
  deno_script="$(mktemp -d)/probe.ts"
  echo 'console.log("deno-probe-ok");' > "$deno_script"
  if "$HOME/.local/bin/deno" run --no-config -A "$deno_script" 2>&1 | grep -q 'deno-probe-ok'; then
    deno_ok=1
  fi
  result "deno=$([ "$deno_ok" = "1" ] && echo ok || echo fail)"

  # Measured, not estimated — the counterpart to the disk footprint above, and
  # taken HERE on purpose: by this point the daemon has warmed its voice and
  # embedding models (run.py::_warmup_models) and served real traffic, so this
  # is the steady-state resident set a user's machine actually has to hold.
  # Read /proc rather than ps: official minimal images ship no procps at all
  # (debian:12 has no `ps`), and an absent ps reports zero bytes just as happily
  # as an idle process would — a measurement that cannot tell "nothing running"
  # from "cannot measure" is worse than none, hence the explicit UNAVAILABLE.
  say "resident memory after all probes"
  rss_kb=0; rss_n=0
  if [ -d /proc ]; then
    for _p in /proc/[0-9]*; do
      tr '\0' ' ' < "$_p/cmdline" 2>/dev/null | grep -q "$HOME/.chalie" || continue
      _r="$(awk '/^VmRSS:/{print $2}' "$_p/status" 2>/dev/null)"
      [ -n "$_r" ] && { rss_kb=$((rss_kb + _r)); rss_n=$((rss_n + 1)); }
    done
  else
    # macOS has no /proc, but ps is always present there.
    _ps="$(ps -eo rss,args 2>/dev/null | grep "$HOME/.chalie" | grep -v grep)"
    rss_kb="$(printf '%s' "$_ps" | awk '{s+=$1} END {print s+0}')"
    rss_n="$(printf '%s\n' "$_ps" | grep -c .)"
  fi
  if [ "$rss_n" -gt 0 ]; then
    echo "chalie RSS total: $((rss_kb / 1024)) MB across $rss_n process(es)"
  else
    echo "chalie RSS UNAVAILABLE — no process matched $HOME/.chalie"
  fi
fi

# ── Verdict ──────────────────────────────────────────────────────────────────
say "verdict"
if [ "$inst_rc" = "0" ] && [ "$ort_ok" = "1" ] && [ "$ready" = "1" ] \
   && [ "$web_ok" = "1" ] && [ "$chromium_ok" = "1" ] \
   && [ "$voice_ok" = "1" ] && [ "$deno_ok" = "1" ]; then
  echo "VERDICT=PASS $NAME"
else
  echo "VERDICT=FAIL $NAME"
fi
