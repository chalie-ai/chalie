#!/usr/bin/env bash
# Chalie Installer
# Usage: curl -fsSL https://chalie.ai/install | bash
# Usage (no voice): curl -fsSL https://chalie.ai/install | bash -s -- --disable-voice
#
# Behaviour:
#   - Idempotent: safe to re-run. Updates, upgrades, and fresh installs all use
#     the same flow. Data at $HOME/.chalie/data is never touched.
#   - If the source tree is already present at $HOME/.chalie/app, the GitHub
#     download is skipped (used by the Dockerfile, which COPYs the source in).
set -euo pipefail

CHALIE_HOME="$HOME/.chalie"
CHALIE_BIN="$HOME/.local/bin"
CHALIE_REPO="chalie-ai/chalie"
GITHUB_API="https://api.github.com/repos/$CHALIE_REPO/releases/latest"

# sqlite-vec: PyPI aarch64 wheel ships a broken 32-bit .so (upstream bug).
# Version pinned to match what requirements.txt resolves; bump both together.
SQLITE_VEC_VERSION="0.1.6"

# Installer flags (parsed from args)
_DISABLE_VOICE=false
_BRANCH=""

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
    case "$1" in
      --disable-voice)         _DISABLE_VOICE=true; shift ;;
      --branch=*)              _BRANCH="${1#--branch=}"; shift ;;
      --branch)                _BRANCH="$2"; shift 2 ;;
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

# ─── Python 3.9+ Check ──────────────────────────────────────────────────────
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
  [[ "$major" -gt 3 ]] || { [[ "$major" -eq 3 ]] && [[ "$minor" -ge 9 ]]; }
}

_install_python_macos() {
  if command -v brew >/dev/null 2>&1; then
    _info "Installing Python 3.12 via Homebrew…"
    brew install python@3.12
  else
    _error "Python 3.9+ is required but was not found."
    _error "Install options:"
    _error "  • Homebrew: https://brew.sh  (then: brew install python@3.12)"
    _error "  • Direct download: https://www.python.org/downloads/"
    exit 1
  fi
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

_install_python_linux() {
  local distro
  distro="$(_detect_linux_distro)"
  _info "Installing Python 3 via package manager…"
  case "$distro" in
    *debian*|*ubuntu*)
      _run_privileged apt-get update -qq
      _run_privileged apt-get install -y python3 python3-pip python3-venv
      ;;
    *fedora*|*rhel*|*centos*)
      _run_privileged dnf install -y python3 python3-pip
      ;;
    *)
      _error "Cannot auto-install Python on distro: $distro"
      _error "Please install Python 3.9+ manually and re-run the installer."
      exit 1
      ;;
  esac
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

  _warn "Python 3.9+ not found. Attempting to install…"
  local os
  os="$(_detect_os)"
  if [[ "$os" == "darwin" ]]; then
    _install_python_macos
  else
    _install_python_linux
  fi

  if _python_version_ok python3; then
    PYTHON="$(command -v python3)"
    _ok "Python installed: $(python3 --version 2>&1)"
  else
    _error "Python installation failed. Please install Python 3.9+ and try again."
    exit 1
  fi
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

  # Install system-level dependencies for soundfile/espeak
  if [[ "$os" == "darwin" ]]; then
    if command -v brew >/dev/null 2>&1; then
      _info "Installing libsndfile and espeak-ng via Homebrew…"
      brew install libsndfile espeak-ng ffmpeg 2>/dev/null || true
    else
      _warn "Homebrew not found — voice system deps may need manual install"
      _warn "  brew install libsndfile espeak-ng ffmpeg"
    fi
  else
    local distro
    distro="$(_detect_linux_distro)"
    _info "Installing voice system dependencies…"
    case "$distro" in
      *debian*|*ubuntu*)
        _run_privileged apt-get install -y libsndfile1 espeak-ng ffmpeg 2>/dev/null || true
        ;;
      *fedora*|*rhel*|*centos*)
        _run_privileged dnf install -y libsndfile espeak-ng ffmpeg 2>/dev/null || true
        ;;
      *)
        _warn "Cannot auto-install voice deps on distro: $distro"
        _warn "Install manually: libsndfile, espeak-ng, ffmpeg"
        ;;
    esac
  fi
  _ok "Voice system dependencies ready"
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
  # If the source tree is already present (Dockerfile COPY'd it, or a previous
  # run completed), skip the download. This is how the Dockerfile reuses this
  # script without needing a flag.
  if [[ -f "$CHALIE_HOME/app/backend/requirements.txt" ]]; then
    _section "Using Existing Source"
    _ok "Source present at $CHALIE_HOME/app"
    return
  fi

  _section "Downloading Chalie"
  local ref tarball_url
  if [[ -n "$_BRANCH" ]]; then
    ref="$_BRANCH"
    _info "Branch: $ref"
    tarball_url="https://github.com/$CHALIE_REPO/archive/refs/heads/$ref.tar.gz"
  else
    ref="$(_fetch_latest_tag)"
    _info "Latest release: $ref"
    tarball_url="https://github.com/$CHALIE_REPO/archive/refs/tags/$ref.tar.gz"
  fi
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

  _info "Extracting to $CHALIE_HOME/app/…"
  mkdir -p "$CHALIE_HOME/app"
  if ! tar -xzf "$tarball" --strip-components=1 -C "$CHALIE_HOME/app"; then
    _error "Extraction failed — downloaded file may be corrupt."
    _error "URL: $tarball_url"
    rm -rf "$tmp_dir"
    exit 1
  fi

  rm -rf "$tmp_dir"
  _ok "Source extracted ($ref)"
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
  if [[ "$_DISABLE_VOICE" != "true" ]]; then
    local voice_req="$CHALIE_HOME/app/backend/requirements-voice.txt"
    if [[ -f "$voice_req" ]]; then
      _info "Installing voice dependencies (STT/TTS)…"
      "$venv/bin/pip" install -r "$voice_req" 2>/dev/null || {
        _warn "Voice dependency install failed — voice will be unavailable"
        _warn "You can retry later: $venv/bin/pip install -r $voice_req"
      }
    fi
  fi

  # Prime run.sh stamp files so the first `chalie start` skips redundant pip sync.
  # Stamp lives in the source tree (or for Docker, /tmp is handled inside run.sh).
  touch "$CHALIE_HOME/app/.deps-installed"
  [[ "$_DISABLE_VOICE" != "true" ]] && touch "$CHALIE_HOME/app/.voice-deps-installed" || true

  _ok "Python environment ready"
  _info "Note: The embedding model (~400 MB) downloads on first 'chalie start', not now"
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

# ─── sqlite-vec aarch64 Fix (Linux only) ───────────────────────────────────
# The PyPI sqlite-vec wheel for linux_aarch64 ships a 32-bit ARM .so — a 64-bit
# process cannot load it. Rebuild from source and replace the broken binary.
# Only runs on Linux arm64; other platforms use the working wheel as-is.
_install_sqlite_vec_fix() {
  local os arch
  os="$(_detect_os)"
  arch="$(_detect_arch)"
  if [[ "$os" != "linux" ]] || [[ "$arch" != "arm64" ]]; then
    return 0
  fi

  _section "sqlite-vec (aarch64 wheel patch)"

  local venv="$CHALIE_HOME/venv"
  # If sqlite-vec isn't pip-installed at all, there's nothing to patch —
  # earlier pip install must have failed. Skip with a different warning.
  if ! "$venv/bin/pip" show sqlite-vec >/dev/null 2>&1; then
    _warn "sqlite-vec not pip-installed — skipping patch"
    return 0
  fi

  # Quick sanity: can we even load the existing wheel? If yes, skip.
  if "$venv/bin/python" -c "
import sqlite3, sqlite_vec
c = sqlite3.connect(':memory:')
c.enable_load_extension(True)
sqlite_vec.load(c)
c.execute('CREATE VIRTUAL TABLE t USING vec0(e float[4])')
" 2>/dev/null; then
    _ok "sqlite-vec loads correctly — no patch needed"
    return 0
  fi

  _info "Rebuilding sqlite-vec from source (PyPI aarch64 wheel is broken)…"
  local tmp
  tmp="$(mktemp -d)"
  (
    cd "$tmp"
    curl -sL "https://github.com/asg017/sqlite-vec/archive/refs/tags/v${SQLITE_VEC_VERSION}.tar.gz" | tar xz
    cd "sqlite-vec-${SQLITE_VEC_VERSION}"
    echo "${SQLITE_VEC_VERSION}" > VERSION
    VERSION="${SQLITE_VEC_VERSION}" DATE=installer SOURCE=local \
      VERSION_MAJOR="$(echo "${SQLITE_VEC_VERSION}" | cut -d. -f1)" \
      VERSION_MINOR="$(echo "${SQLITE_VEC_VERSION}" | cut -d. -f2)" \
      VERSION_PATCH="$(echo "${SQLITE_VEC_VERSION}" | cut -d. -f3)" \
      envsubst < sqlite-vec.h.tmpl > sqlite-vec.h
    mkdir -p dist
    cc -fPIC -shared -O3 -lm -I/usr/include sqlite-vec.c -o dist/vec0.so
    site_pkg="$("$venv/bin/python" -c 'import sqlite_vec, os; print(os.path.dirname(sqlite_vec.__file__))')"
    cp dist/vec0.so "$site_pkg/vec0.so"
  ) || {
    rm -rf "$tmp"
    _warn "sqlite-vec rebuild failed — vector search will not work"
    return 0
  }
  rm -rf "$tmp"

  # Verify the patch took.
  if "$venv/bin/python" -c "
import sqlite3, sqlite_vec
c = sqlite3.connect(':memory:')
c.enable_load_extension(True)
sqlite_vec.load(c)
c.execute('CREATE VIRTUAL TABLE t USING vec0(e float[4])')
" 2>/dev/null; then
    _ok "sqlite-vec patched and verified"
  else
    _warn "sqlite-vec still fails to load after rebuild"
  fi
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
DATA_DIR="$CHALIE_HOME/data"

_is_running() {
  [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

# Parse --port=N, --port N, --host=H from all arguments
_port="8081"
_host="0.0.0.0"
_cmd=""
_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port=*) _port="${1#--port=}"; shift ;;
    --port)   _port="$2"; shift 2 ;;
    --host=*) _host="${1#--host=}"; shift ;;
    --host)   _host="$2"; shift 2 ;;
    --version|-V) _cmd="version"; shift ;;
    stop|restart|update|status|logs|help|version) _cmd="$1"; shift ;;
    *) _args+=("$1"); shift ;;
  esac
done
# Default command: start (if no named command given)
_cmd="${_cmd:-start}"

case "$_cmd" in
  start)
    _is_running && echo "Chalie is already running (PID $(cat "$PID_FILE"))" && exit 0
    mkdir -p "$DATA_DIR"
    CHALIE_DATA_DIR="$DATA_DIR" CHALIE_VENV="$CHALIE_HOME/venv" \
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
    echo "  chalie                   Start on port 8081 (default)"
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
  printf "${_green}${_bold}  │${_reset}    ${_cyan}chalie${_reset}              Start on port 8081    ${_green}${_bold}│${_reset}\n"
  printf "${_green}${_bold}  │${_reset}    ${_cyan}chalie --port=9000${_reset}  Custom port           ${_green}${_bold}│${_reset}\n"
  printf "${_green}${_bold}  │${_reset}    ${_cyan}chalie stop${_reset}         Stop                  ${_green}${_bold}│${_reset}\n"
  printf "${_green}${_bold}  │${_reset}    ${_cyan}chalie update${_reset}       Update to latest      ${_green}${_bold}│${_reset}\n"
  printf "${_green}${_bold}  │${_reset}    ${_cyan}chalie logs${_reset}         Follow logs           ${_green}${_bold}│${_reset}\n"
  printf "${_green}${_bold}  └─────────────────────────────────────────────┘${_reset}\n"
  printf "\n"
}

# ─── Main ────────────────────────────────────────────────────────────────────
# Single flow — fresh installs, upgrades, and re-runs all take the same path.
# Every step is idempotent. Data at $CHALIE_HOME/data is never touched.
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
  _install_playwright_browsers
  _install_sqlite_vec_fix
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
