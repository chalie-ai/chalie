#!/usr/bin/env bash
# Chalie Installer
# Usage: curl -fsSL https://chalie.ai/install | bash
# Usage (no voice): curl -fsSL https://chalie.ai/install | bash -s -- --disable-voice
#
# Behaviour:
#   - Idempotent: safe to re-run. Updates, upgrades, and fresh installs all use
#     the same flow. Data at $HOME/.chalie/app/data is never touched.
#   - Upgrades: if source already exists, the installer always downloads the
#     target version, replaces managed source directories, and re-runs pip
#     install.
#   - One-time migration: legacy installs whose DB lives inside the source
#     tree at $HOME/.chalie/app/backend/data are moved up one level to
#     $HOME/.chalie/app/data, where the new paths module looks for it.
set -euo pipefail

CHALIE_HOME="$HOME/.chalie"
CHALIE_BIN="$HOME/.local/bin"
CHALIE_REPO="chalie-ai/chalie"
GITHUB_API="https://api.github.com/repos/$CHALIE_REPO/releases/latest"

# onnxruntime: version must match across the three wheels (CPU, CUDA, ROCm).
# Changing this line is the single source of truth — requirements.txt no longer
# pins it; _install_onnxruntime_variant() picks the right wheel for the host.
ORT_VERSION="1.20.1"

# AMD's public ROCm Python wheel index. Microsoft doesn't ship onnxruntime-rocm
# to PyPI; AMD hosts pre-built wheels here, updated per ROCm release. Override
# via $CHALIE_ROCM_INDEX for air-gapped installs.
ROCM_PIP_INDEX="${CHALIE_ROCM_INDEX:-https://repo.radeon.com/rocm/manylinux/latest/}"

# Installer flags (parsed from args)
_DISABLE_VOICE=false
_BRANCH=""
_TAG=""

# ─── Colours ────────────────────────────────────────────────────────────────
_reset="\033[0m"
_bold="\033[1m"
_violet="\033[35m"
_cyan="\033[36m"
_green="\033[32m"
_yellow="\033[33m"
_red="\033[31m"

_info()    { printf "  ${_cyan}→${_reset}  %s\n" "$*"; }
_ok()      { printf "  ${_green}✓${_reset}  %s\n" "$*"; }
_warn()    { printf "  ${_yellow}⚠${_reset}  %s\n" "$*"; }
_error()   { printf "  ${_red}✗${_reset}  %s\n" "$*" >&2; }
_section() { printf "\n${_bold}${_violet}%s${_reset}\n" "$*"; }
_banner() {
  printf "\n"
  printf "${_violet}  ┌─────────────────────────────────────────────┐${_reset}\n"
  printf "${_violet}  │${_reset}    ${_bold}Chalie Installer${_reset}                            ${_violet}│${_reset}\n"
  printf "${_violet}  │${_reset}    ${_cyan}A personal intelligence layer${_reset}               ${_violet}│${_reset}\n"
  printf "${_violet}  └─────────────────────────────────────────────┘${_reset}\n"
  printf "\n"
}

# ─── Parse Installer Args ──────────────────────────────────────────────────
_parse_args() {
  while [[ $# -gt 0 ]]; do
    local arg="$1"
    case "$arg" in
      --disable-voice)         _DISABLE_VOICE=true; shift ;;
      --branch=*)              _BRANCH="${arg#--branch=}"; shift ;;
      --branch)                _BRANCH="$2"; shift 2 ;;
      --tag=*)                 _TAG="${arg#--tag=}"; shift ;;
      --tag)                   _TAG="$2"; shift 2 ;;
      --disable-default-tools) shift ;; # deprecated, ignored — tools are bundled in the repo
      *) shift ;;
    esac
  done
}

# ─── OS + Arch Detection ────────────────────────────────────────────────────
_detect_os() {
  case "$OSTYPE" in
    darwin*)  echo "darwin" ;;
    linux*)   echo "linux" ;;
    *)
      _error "Unsupported OS: $OSTYPE"
      _error "Chalie supports macOS (Intel/Apple Silicon) and Linux (amd64/arm64)."
      exit 1
      ;;
  esac
}

_detect_arch() {
  local machine
  machine="$(uname -m)"
  case "$machine" in
    x86_64)           echo "amd64" ;;
    arm64|aarch64)    echo "arm64" ;;
    *)
      _error "Unsupported architecture: $machine"
      exit 1
      ;;
  esac
}

_detect_linux_distro() {
  if [[ -f /etc/os-release ]]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    echo "${ID_LIKE:-$ID}"
  else
    echo "unknown"
  fi
}

# Emits "cuda" | "rocm" | "cpu" — nothing else. Only Linux can have GPU wheels;
# macOS hands CoreML to the default onnxruntime wheel, so it always reports cpu.
#
# NVIDIA: nvidia-smi must exist AND succeed (driver loaded). A stale symlink with
# no kernel module returns non-zero and we fall through to cpu — correct, because
# onnxruntime-gpu would fail to initialise CUDA at runtime anyway.
#
# AMD: /dev/kfd is the ROCm kernel compute interface; /sys/module/amdgpu confirms
# the kernel driver is loaded. Both must be present — a Radeon with only the
# display driver (no ROCm stack) would have /sys/module/amdgpu but no /dev/kfd.
_detect_gpu_variant() {
  local os
  os="$(_detect_os)"
  if [[ "$os" != "linux" ]]; then
    echo "cpu"; return
  fi
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    echo "cuda"; return
  fi
  if [[ -e /dev/kfd ]] && [[ -d /sys/module/amdgpu ]]; then
    echo "rocm"; return
  fi
  echo "cpu"
}

# ─── Python 3.11+ Check ─────────────────────────────────────────────────────
_python_version_ok() {
  local py="${1:-python3}"
  if ! command -v "$py" >/dev/null 2>&1; then
    return 1
  fi
  local ver
  ver="$("$py" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')"
  local major minor
  major="$(echo "$ver" | cut -d. -f1)"
  minor="$(echo "$ver" | cut -d. -f2)"
  [[ "$major" -gt 3 ]] || { [[ "$major" -eq 3 ]] && [[ "$minor" -ge 11 ]]; }
}

_needs_sudo() {
  # Running as root → no sudo needed; otherwise require sudo to be available
  if [[ "$(id -u)" -eq 0 ]]; then
    return 1
  fi
  if ! command -v sudo >/dev/null 2>&1; then
    _error "This step requires root privileges but 'sudo' is not available."
    _error "Either run the installer as root or install sudo first."
    exit 1
  fi
  return 0
}

_run_privileged() {
  # Run a command with sudo only when not already root
  if _needs_sudo; then
    sudo "$@"
  else
    "$@"
  fi
}

_check_python() {
  _section "Python"
  if _python_version_ok python3; then
    local ver
    ver="$(python3 --version 2>&1)"
    _ok "Found $ver"
    PYTHON="$(command -v python3)"
    return
  fi

  _error "Python 3.11+ is required but was not found."
  _error "Please install Python 3.11+ and re-run the installer."
  _error ""
  _error "Install options:"
  _error "  • macOS:         brew install python@3.12"
  _error "  • Debian/Ubuntu: sudo apt-get install python3 python3-pip python3-venv"
  _error "  • Fedora/RHEL:   sudo dnf install python3 python3-pip"
  _error "  • Download:      https://www.python.org/downloads/"
  exit 1
}

# ─── System Build Dependencies (Linux) ──────────────────────────────────────
# Needed for native Python wheels (cryptography, pywebpush), sqlite-vec rebuild,
# envsubst (sqlite-vec template), Deno installer (unzip), and curl.
_install_build_deps() {
  local os
  os="$(_detect_os)"
  [[ "$os" != "linux" ]] && return 0

  _section "System Build Dependencies"
  local distro
  distro="$(_detect_linux_distro)"
  case "$distro" in
    *debian*|*ubuntu*)
      _run_privileged apt-get update -qq
      _run_privileged apt-get install -y --no-install-recommends \
        build-essential libffi-dev libsqlite3-dev gettext-base curl unzip \
        python3-venv
      ;;
    *fedora*|*rhel*|*centos*)
      _run_privileged dnf install -y gcc gcc-c++ make libffi-devel sqlite-devel gettext curl unzip
      ;;
    *)
      _warn "Unknown distro '$distro' — assuming build tools are present"
      ;;
  esac
  _ok "Build dependencies ready"
}

# ─── ONNX Runtime (GPU wheel swap) ──────────────────────────────────────────
# requirements.txt pulls the CPU `onnxruntime` wheel in transitively via
# rapidocr-onnxruntime. If the host has a GPU, we swap that wheel for the
# matching accelerator build. On any failure the CPU wheel stays in place,
# so Chalie still boots — just without GPU acceleration. This is the
# fallback guarantee: the baseline wheel never gets removed until the
# replacement is confirmed.
_install_onnxruntime_variant() {
  local variant
  variant="$(_detect_gpu_variant)"
  if [[ "$variant" == "cpu" ]]; then
    return 0
  fi

  _section "ONNX Runtime GPU Wheel ($variant)"
  local venv="$CHALIE_HOME/venv"
  local pip="$venv/bin/pip"

  local wheel extra=()
  case "$variant" in
    cuda)
      wheel="onnxruntime-gpu==$ORT_VERSION"
      ;;
    rocm)
      wheel="onnxruntime-rocm==$ORT_VERSION"
      extra=(--extra-index-url "$ROCM_PIP_INDEX")
      ;;
    *)
      _error "Unknown ONNX Runtime variant: $variant"
      return 1
      ;;
  esac

  _info "Detected $variant — installing $wheel"

  # Swap wheels atomically from Python's point of view: uninstall the CPU wheel
  # only after the accelerator wheel downloads cleanly. --dry-run reserves the
  # package and catches registry/index errors before we touch the working set.
  if ! "$pip" install --dry-run "${extra[@]}" "$wheel" >/dev/null 2>&1; then
    _warn "$wheel not reachable — keeping CPU onnxruntime"
    _warn "Chalie will run on CPU providers. Fix the GPU toolkit and re-run installer."
    return 0
  fi

  "$pip" uninstall -y onnxruntime >/dev/null 2>&1 || true
  if "$pip" install "${extra[@]}" "$wheel"; then
    _ok "ONNX Runtime GPU wheel ready ($wheel)"
  else
    _warn "$wheel install failed post-download — restoring CPU onnxruntime"
    "$pip" install "onnxruntime==$ORT_VERSION"
  fi
}

# ─── Voice Dependencies (native, no Docker) ────────────────────────────────
_install_voice_deps() {
  if [[ "$_DISABLE_VOICE" == "true" ]]; then
    _section "Voice (skipped — --disable-voice)"
    _info "Voice disabled at install time. STT/TTS will not be available."
    _info "Re-run installer without --disable-voice to enable later."
    return
  fi

  _section "Voice Dependencies"
  local os
  os="$(_detect_os)"

  # Install system-level dependencies for soundfile + ffmpeg.
  # kokoro-onnx pulls in espeakng-loader (wheel-shipped espeak-ng binary),
  # so the system-level espeak-ng package is not required.
  if [[ "$os" == "darwin" ]]; then
    if command -v brew >/dev/null 2>&1; then
      _info "Installing libsndfile and ffmpeg via Homebrew…"
      brew install libsndfile ffmpeg 2>/dev/null || true
    else
      _warn "Homebrew not found — voice system deps may need manual install"
      _warn "  brew install libsndfile ffmpeg"
    fi
  else
    local distro
    distro="$(_detect_linux_distro)"
    _info "Installing voice system dependencies…"
    case "$distro" in
      *debian*|*ubuntu*)
        _run_privileged apt-get install -y libsndfile1 ffmpeg 2>/dev/null || true
        ;;
      *fedora*|*rhel*|*centos*)
        _run_privileged dnf install -y libsndfile ffmpeg 2>/dev/null || true
        ;;
      *alpine*)
        _run_privileged apk add --no-cache libsndfile ffmpeg 2>/dev/null || true
        ;;
      *arch*|*manjaro*)
        _run_privileged pacman -S --needed --noconfirm libsndfile ffmpeg 2>/dev/null || true
        ;;
      *opensuse*|*suse*)
        _run_privileged zypper install -y libsndfile ffmpeg 2>/dev/null || true
        ;;
      *)
        _warn "Cannot auto-install voice deps on distro: $distro"
        _warn "Install manually: libsndfile, ffmpeg"
        ;;
    esac
  fi
  _ok "Voice system dependencies ready"
}

# ─── Voice Models (baked into the install) ─────────────────────────────────
#
# Kokoro TTS (~310 MB ONNX + ~28 MB voices) and Moonshine STT (~150 MB ONNX
# encoder + decoder) live under $CHALIE_HOME/app/resources/voice-models/. This
# directory is OUTSIDE the data/ bind mount, so the files travel with the
# image when Docker builds, and survive `chalie update` on native installs.
#
# All downloads are idempotent — if the file already exists at the expected
# path, the curl is skipped. Failures here are non-fatal: voice will return
# 503 at runtime with a clear hint, but the rest of Chalie still boots.
_download_voice_models() {
  if [[ "$_DISABLE_VOICE" == "true" ]]; then
    return 0
  fi

  _section "Voice Models"
  local voice_root="$CHALIE_HOME/app/resources/voice-models"
  local kokoro_dir="$voice_root/kokoro"
  local moonshine_dir="$voice_root/moonshine/base"
  mkdir -p "$kokoro_dir" "$moonshine_dir"

  local kokoro_release="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
  local moonshine_hf="https://huggingface.co/UsefulSensors/moonshine/resolve/main/onnx/merged/base/float"

  local fetched=0
  _fetch_model() {
    local url="$1" dest="$2" label="$3"
    if [[ -f "$dest" ]] && [[ "$(stat -c%s "$dest" 2>/dev/null || stat -f%z "$dest" 2>/dev/null)" -gt 1024 ]]; then
      return 0
    fi
    _info "Downloading ${label}…"
    if ! curl -fL --progress-bar -o "$dest" "$url"; then
      _warn "$label download failed — voice will be unavailable until you re-run the installer"
      rm -f "$dest"
      return 1
    fi
    fetched=$((fetched + 1))
  }

  _fetch_model "$kokoro_release/kokoro-v1.0.onnx" "$kokoro_dir/kokoro-v1.0.onnx" "Kokoro TTS model (~310 MB)" || true
  _fetch_model "$kokoro_release/voices-v1.0.bin"   "$kokoro_dir/voices-v1.0.bin"   "Kokoro voices (~28 MB)"     || true
  _fetch_model "$moonshine_hf/encoder_model.onnx"        "$moonshine_dir/encoder_model.onnx"        "Moonshine encoder" || true
  _fetch_model "$moonshine_hf/decoder_model_merged.onnx" "$moonshine_dir/decoder_model_merged.onnx" "Moonshine decoder" || true

  if [[ "$fetched" -eq 0 ]]; then
    local missing=()
    [[ -f "$kokoro_dir/kokoro-v1.0.onnx" ]]             || missing+=("resources/voice-models/kokoro/kokoro-v1.0.onnx")
    [[ -f "$kokoro_dir/voices-v1.0.bin" ]]               || missing+=("resources/voice-models/kokoro/voices-v1.0.bin")
    [[ -f "$moonshine_dir/encoder_model.onnx" ]]         || missing+=("resources/voice-models/moonshine/base/encoder_model.onnx")
    [[ -f "$moonshine_dir/decoder_model_merged.onnx" ]]  || missing+=("resources/voice-models/moonshine/base/decoder_model_merged.onnx")
    if [[ "${#missing[@]}" -eq 0 ]]; then
      _ok "Voice models already present at $voice_root"
    else
      _warn "Voice models missing — no downloads were attempted but the following files are absent:"
      for f in "${missing[@]}"; do
        _warn "  $f"
      done
      _warn "Re-run the installer to download them."
    fi
  else
    _ok "Voice models ready at $voice_root"
  fi
}

# ─── Download Latest Release ────────────────────────────────────────────────
_fetch_latest_tag() {
  local tag
  tag="$(curl -fsSL "$GITHUB_API" 2>/dev/null | grep '"tag_name"' | head -1 | cut -d'"' -f4)"
  if [[ -z "$tag" ]]; then
    _error "Could not fetch latest release tag from GitHub."
    _error "Check your internet connection and try again."
    exit 1
  fi
  echo "$tag"
}

_download_release() {
  local is_upgrade=false
  local current_version="unknown"
  if [[ -f "$CHALIE_HOME/app/backend/requirements.txt" ]]; then
    is_upgrade=true
    current_version="$(cat "$CHALIE_HOME/app/VERSION" 2>/dev/null || echo unknown)"
  fi

  if [[ "$is_upgrade" == true ]]; then
    _section "Upgrading Chalie"
    _info "Installed version: $current_version"
  else
    _section "Downloading Chalie"
  fi

  # Priority order:
  #   1. --tag=NAME  → fetch refs/tags/NAME.tar.gz directly (no API lookup).
  #      Used by the Docker workflow on tag pushes — avoids racing with the
  #      release-publication step that gates _fetch_latest_tag.
  #   2. --branch=NAME → fetch refs/heads/NAME.tar.gz (development builds).
  #   3. neither → call the GitHub API for the latest published release tag.
  local ref tarball_url
  if [[ -n "$_TAG" ]]; then
    ref="$_TAG"
    _info "Tag: $ref"
    tarball_url="https://github.com/$CHALIE_REPO/archive/refs/tags/$ref.tar.gz"
  elif [[ -n "$_BRANCH" ]]; then
    ref="$_BRANCH"
    _info "Branch: $ref"
    tarball_url="https://github.com/$CHALIE_REPO/archive/refs/heads/$ref.tar.gz"
  else
    ref="$(_fetch_latest_tag)"
    _info "Latest release: $ref"
    tarball_url="https://github.com/$CHALIE_REPO/archive/refs/tags/$ref.tar.gz"
  fi

  local ref_version="${ref#v}"

  local tmp_dir
  tmp_dir="$(mktemp -d)"
  local tarball="$tmp_dir/chalie.tar.gz"

  _info "Downloading source archive…"
  local http_code
  http_code="$(curl -fSL -w '%{http_code}' --progress-bar "$tarball_url" -o "$tarball" 2>/dev/null)" || true

  if [[ ! -f "$tarball" ]] || [[ "$(stat -c%s "$tarball" 2>/dev/null || stat -f%z "$tarball" 2>/dev/null)" -lt 1024 ]]; then
    _error "Download failed (HTTP $http_code). The release archive could not be fetched."
    _error "URL: $tarball_url"
    _error "This usually means the ref '$ref' does not have a matching archive."
    rm -rf "$tmp_dir"
    exit 1
  fi

  # Remove old managed source directories before extraction so deleted files
  # from previous versions don't linger. data/ is user state — never touched.
  if [[ "$is_upgrade" == true ]]; then
    _info "Removing old source…"
    rm -rf "$CHALIE_HOME/app/backend" \
           "$CHALIE_HOME/app/frontend" \
           "$CHALIE_HOME/app/resources" \
           "$CHALIE_HOME/app/installer" \
           "$CHALIE_HOME/app/docs" \
           "$CHALIE_HOME/app/scripts" \
           "$CHALIE_HOME/app/utils"
  fi

  _info "Extracting to $CHALIE_HOME/app/…"
  mkdir -p "$CHALIE_HOME/app"
  if ! tar -xzf "$tarball" --strip-components=1 -C "$CHALIE_HOME/app"; then
    _error "Extraction failed — downloaded file may be corrupt."
    _error "URL: $tarball_url"
    rm -rf "$tmp_dir"
    exit 1
  fi

  rm -rf "$tmp_dir"
  if [[ "$is_upgrade" == true ]]; then
    _ok "Upgraded: $current_version → $ref_version"
  else
    _ok "Source extracted ($ref)"
  fi
}

# ─── Python Virtualenv + Dependencies ───────────────────────────────────────
_setup_venv() {
  _section "Python Environment"
  local venv="$CHALIE_HOME/venv"

  if [[ ! -d "$venv" ]]; then
    _info "Creating virtual environment…"
    "$PYTHON" -m venv "$venv"
  else
    _info "Reusing existing virtual environment"
  fi

  _info "Upgrading pip…"
  "$venv/bin/pip" install --upgrade pip

  _info "Installing core dependencies (this may take a few minutes)…"
  "$venv/bin/pip" install -r "$CHALIE_HOME/app/backend/requirements.txt"

  # Voice dependencies (separate file, skipped if --disable-voice)
  local voice_req="$CHALIE_HOME/app/backend/requirements-voice.txt"
  if [[ "$_DISABLE_VOICE" != "true" ]] && [[ -f "$voice_req" ]]; then
    _info "Installing voice dependencies (STT/TTS)…"
    "$venv/bin/pip" install -r "$voice_req" 2>/dev/null || {
      _warn "Voice dependency install failed — voice will be unavailable"
      _warn "You can retry later: $venv/bin/pip install -r $voice_req"
    }
  fi

  # Prime run.sh stamp files so the first `chalie start` skips redundant pip sync.
  # Stamp lives in the source tree (or for Docker, /tmp is handled inside run.sh).
  touch "$CHALIE_HOME/app/.deps-installed"
  [[ "$_DISABLE_VOICE" != "true" ]] && touch "$CHALIE_HOME/app/.voice-deps-installed" || true

  _ok "Python environment ready"
  _info "Note: The embedding model (~400 MB) downloads on first 'chalie start', not now"
  _info "Voice models (Kokoro TTS + Moonshine STT) download next, baked into the install"
}

# ─── Playwright Browsers ────────────────────────────────────────────────────
# playwright pip-installs the Python bindings; the Chromium binary is a separate
# download. Without this step the browser tool fails at runtime with
# "Executable doesn't exist at …/chromium_headless_shell/" — the bug this
# installer step exists to prevent. Failure is fatal; re-run the installer.
_install_playwright_browsers() {
  _section "Browser Runtime (Playwright Chromium)"
  local venv="$CHALIE_HOME/venv"
  local os
  os="$(_detect_os)"

  # Linux needs OS-level deps for Chromium (fonts, libnss, libatk, …);
  # --with-deps calls apt-get/dnf internally and requires privileged access.
  local pw_cmd=("$venv/bin/playwright" install chromium)
  if [[ "$os" == "linux" ]]; then
    pw_cmd+=(--with-deps)
    if _needs_sudo; then
      pw_cmd=(sudo "${pw_cmd[@]}")
    fi
  fi

  "${pw_cmd[@]}"
  _ok "Playwright Chromium ready"
  return 0
}

# ─── Deno Runtime (for interface daemons) ──────────────────────────────────
# Chalie runs user-authored TypeScript interface daemons in Deno. Without it,
# those interfaces silently fail to start.
_install_deno() {
  _section "Deno Runtime"
  if command -v deno >/dev/null 2>&1; then
    _ok "Found $(deno --version 2>&1 | head -1)"
    return
  fi
  _info "Installing Deno…"
  # Official installer drops into $HOME/.deno/bin
  if ! curl -fsSL https://deno.land/install.sh | sh >/dev/null 2>&1; then
    _warn "Deno install failed — TypeScript interface daemons will be unavailable"
    _warn "Retry manually: curl -fsSL https://deno.land/install.sh | sh"
    return 0
  fi
  # Add ~/.deno/bin to PATH for future shells
  local deno_path='export PATH="$HOME/.deno/bin:$PATH"'
  for rc in "$HOME/.bashrc" "$HOME/.zshrc"; do
    if [[ -f "$rc" ]] && ! grep -qF '.deno/bin' "$rc" 2>/dev/null; then
      printf '\n# Added by Chalie installer\n%s\n' "$deno_path" >> "$rc"
    fi
  done
  _ok "Deno installed at $HOME/.deno"
}

# ─── Install CLI Wrapper ─────────────────────────────────────────────────────
_install_cli() {
  _section "CLI Wrapper"
  mkdir -p "$CHALIE_BIN"

  cat > "$CHALIE_BIN/chalie" <<'CHALIE_CLI'
#!/usr/bin/env bash
CHALIE_HOME="${CHALIE_HOME:-$HOME/.chalie}"
PID_FILE="$CHALIE_HOME/chalie.pid"
LOG_FILE="$CHALIE_HOME/chalie.log"
# Runtime state lives alongside the source tree so the Python `paths` module
# (which resolves data/ relative to backend/) finds it without env vars.
DATA_DIR="$CHALIE_HOME/app/data"

_is_running() {
  [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

# Parse --port=N, --port N, --host=H from all arguments
_port="31025"
_host="0.0.0.0"
_cmd=""
_args=()
while [[ $# -gt 0 ]]; do
  _arg="$1"
  case "$_arg" in
    --port=*) _port="${_arg#--port=}"; shift ;;
    --port)   _port="$2"; shift 2 ;;
    --host=*) _host="${_arg#--host=}"; shift ;;
    --host)   _host="$2"; shift 2 ;;
    --version|-V) _cmd="version"; shift ;;
    stop|restart|update|status|logs|help|version) _cmd="$_arg"; shift ;;
    *) _args+=("$_arg"); shift ;;
  esac
done
# Default command: start (if no named command given)
_cmd="${_cmd:-start}"

case "$_cmd" in
  start)
    _is_running && echo "Chalie is already running (PID $(cat "$PID_FILE"))" && exit 0
    mkdir -p "$DATA_DIR"
    CHALIE_VENV="$CHALIE_HOME/venv" \
      bash "$CHALIE_HOME/app/run.sh" --port="$_port" --host="$_host" \
      >> "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    echo "Chalie started → http://localhost:$_port"
    ;;
  stop)
    _is_running || { echo "Chalie is not running"; exit 0; }
    kill "$(cat "$PID_FILE")" && rm -f "$PID_FILE" && echo "Chalie stopped"
    ;;
  restart)
    "$0" stop
    sleep 1
    "$0" --port="$_port" --host="$_host"
    ;;
  update)
    _is_running && "$0" stop
    curl -fsSL https://chalie.ai/install | bash
    ;;
  status)
    _is_running && echo "Running (PID $(cat "$PID_FILE"))" || echo "Not running"
    ;;
  logs)
    if [[ ! -f "$LOG_FILE" ]]; then
      echo "No log file found at $LOG_FILE" >&2
      exit 1
    fi
    # Follow interactively when stdout is a terminal; show last 50 lines and exit when piped
    if [[ -t 1 ]]; then
      tail -f "$LOG_FILE"
    else
      tail -n 50 "$LOG_FILE"
    fi
    ;;
  version)
    _ver=""
    for _vf in "$CHALIE_HOME/app/VERSION" "$CHALIE_HOME/app/backend/consumer.py"; do
      if [[ -f "$_vf" ]]; then
        if [[ "$_vf" == *.py ]]; then
          _ver=$(grep -oE 'APP_VERSION\s*=\s*"[^"]+"' "$_vf" 2>/dev/null | grep -oE '"[^"]+"' | tr -d '"')
        else
          _ver=$(cat "$_vf" | tr -d '[:space:]')
        fi
        [[ -n "$_ver" ]] && break
      fi
    done
    echo "chalie ${_ver:-unknown}"
    ;;
  help|*)
    echo "Usage: chalie [--port=N] [--host=H] [stop|restart|update|status|logs|version]"
    echo ""
    echo "  chalie                   Start on port 31025 (default)"
    echo "  chalie --port=9000       Start on a custom port"
    echo "  chalie --host=127.0.0.1  Bind to specific address"
    echo "  chalie stop              Stop Chalie"
    echo "  chalie restart           Restart Chalie"
    echo "  chalie update            Update to the latest release"
    echo "  chalie status            Check if Chalie is running"
    echo "  chalie logs              Follow the log"
    ;;
esac
CHALIE_CLI

  chmod +x "$CHALIE_BIN/chalie"
  _ok "CLI installed at $CHALIE_BIN/chalie"

  # Ensure ~/.local/bin is in PATH
  local path_line='export PATH="$HOME/.local/bin:$PATH"'
  local added_path=false

  if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    for rc in "$HOME/.bashrc" "$HOME/.zshrc"; do
      if [[ -f "$rc" ]]; then
        if ! grep -qF '.local/bin' "$rc" 2>/dev/null; then
          printf '\n# Added by Chalie installer\n%s\n' "$path_line" >> "$rc"
          added_path=true
        fi
      fi
    done
    if [[ "$added_path" == "true" ]]; then
      _info "Added ~/.local/bin to PATH in shell config"
      _warn "Run 'source ~/.bashrc' (or open a new terminal) to use the chalie command"
    fi
  fi
}

# ─── Success Banner ──────────────────────────────────────────────────────────
_print_success() {
  printf "\n"
  printf "${_green}${_bold}  ┌─────────────────────────────────────────────┐${_reset}\n"
  printf "${_green}${_bold}  │${_reset}  ${_bold}Chalie is installed!${_reset}                        ${_green}${_bold}│${_reset}\n"
  printf "${_green}${_bold}  │${_reset}                                             ${_green}${_bold}│${_reset}\n"
  printf "${_green}${_bold}  │${_reset}    ${_cyan}chalie${_reset}              Start on port 31025${_green}${_bold}│${_reset}\n"
  printf "${_green}${_bold}  │${_reset}    ${_cyan}chalie --port=9000${_reset}  Custom port           ${_green}${_bold}│${_reset}\n"
  printf "${_green}${_bold}  │${_reset}    ${_cyan}chalie stop${_reset}         Stop                  ${_green}${_bold}│${_reset}\n"
  printf "${_green}${_bold}  │${_reset}    ${_cyan}chalie update${_reset}       Update to latest      ${_green}${_bold}│${_reset}\n"
  printf "${_green}${_bold}  │${_reset}    ${_cyan}chalie logs${_reset}         Follow logs           ${_green}${_bold}│${_reset}\n"
  printf "${_green}${_bold}  └─────────────────────────────────────────────┘${_reset}\n"
  printf "\n"
}

# ─── Main ────────────────────────────────────────────────────────────────────
# Single flow — fresh installs, upgrades, and re-runs all take the same path.
# Every step is idempotent. Data at $CHALIE_HOME/app/data is never touched.
main() {
  _parse_args "$@"

  local os arch
  os="$(_detect_os)"
  arch="$(_detect_arch)"

  _banner
  printf "  Platform: %s / %s\n\n" "$os" "$arch"

  _check_python
  _install_build_deps
  _install_voice_deps
  _install_deno
  _download_release
  _setup_venv
  _install_onnxruntime_variant
  _install_playwright_browsers
  _download_voice_models
  _install_cli
  _print_success

  # Only prompt to start when stdin is a TTY — non-interactive invocations
  # (Docker build, piped curl | bash in CI) just finish silently.
  if [[ -t 0 ]]; then
    printf "\n"
    read -r -p "  Start Chalie now? [Y/n] " _start_reply
    printf "\n"
    if [[ "${_start_reply,,}" != "n" ]]; then
      "$CHALIE_BIN/chalie" start
    fi
  fi
}

main "$@"
