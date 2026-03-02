#!/usr/bin/env bash
# Chalie Installer
# Usage: curl -fsSL https://chalie.ai/install | bash
# Usage (update): curl -fsSL https://chalie.ai/install | CHALIE_UPDATE=1 bash
# Usage (no voice): curl -fsSL https://chalie.ai/install | bash -s -- --disable-voice
set -euo pipefail

CHALIE_HOME="${CHALIE_HOME:-$HOME/.chalie}"
CHALIE_BIN="${CHALIE_BIN:-$HOME/.local/bin}"
CHALIE_REPO="chalie-ai/chalie"
GITHUB_API="https://api.github.com/repos/$CHALIE_REPO/releases/latest"

# Installer flags (parsed from args)
_DISABLE_VOICE=false

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
      --disable-voice) _DISABLE_VOICE=true; shift ;;
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

_install_python_linux() {
  local distro
  distro="$(_detect_linux_distro)"
  _info "Installing Python 3 via package manager…"
  case "$distro" in
    *debian*|*ubuntu*)
      sudo apt-get update -qq
      sudo apt-get install -y python3 python3-pip python3-venv
      ;;
    *fedora*|*rhel*|*centos*)
      sudo dnf install -y python3 python3-pip
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

# ─── Docker Check (non-fatal, sandboxed tools only) ────────────────────────
_check_docker() {
  _section "Docker (optional — sandboxed tools only)"
  if docker info >/dev/null 2>&1; then
    _ok "Docker is available — sandboxed tool execution enabled"
    return
  fi

  local os
  os="$(_detect_os)"
  if [[ "$os" == "darwin" ]]; then
    if command -v docker >/dev/null 2>&1; then
      _warn "Docker is installed but the daemon is not running."
      _warn "Start Docker Desktop to enable sandboxed tool execution."
    else
      _warn "Docker not found."
      _warn "Only needed for sandboxed tool execution (not voice, not core features)."
      _warn "Install Docker Desktop if you want sandboxed tools:"
      _warn "  https://www.docker.com/products/docker-desktop/"
    fi
  else
    # Linux: ask
    printf "\n"
    read -r -p "  Install Docker for sandboxed tool execution? [y/N] " _docker_reply
    printf "\n"
    if [[ "${_docker_reply,,}" == "y" ]]; then
      _info "Installing Docker via get.docker.com…"
      curl -fsSL https://get.docker.com | sudo sh
      if id -nG "$USER" | grep -qw docker; then
        _ok "Already in docker group"
      else
        sudo usermod -aG docker "$USER"
        _warn "Added $USER to docker group. Log out and back in for group membership to take effect."
      fi
      _ok "Docker installed"
    else
      _warn "Skipping Docker. Sandboxed tools will be disabled; trusted tools and core features work fine."
    fi
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
        sudo apt-get install -y libsndfile1 espeak-ng ffmpeg 2>/dev/null || true
        ;;
      *fedora*|*rhel*|*centos*)
        sudo dnf install -y libsndfile espeak-ng ffmpeg 2>/dev/null || true
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
  _section "Downloading Chalie"
  local tag
  tag="$(_fetch_latest_tag)"
  _info "Latest release: $tag"

  local tarball_url="https://github.com/$CHALIE_REPO/archive/refs/tags/$tag.tar.gz"
  local tmp_dir
  tmp_dir="$(mktemp -d)"
  local tarball="$tmp_dir/chalie.tar.gz"

  _info "Downloading source archive…"
  curl -fsSL --progress-bar "$tarball_url" -o "$tarball"

  _info "Extracting to $CHALIE_HOME/app/…"
  mkdir -p "$CHALIE_HOME/app"
  tar -xzf "$tarball" --strip-components=1 -C "$CHALIE_HOME/app"

  rm -rf "$tmp_dir"
  _ok "Source extracted ($tag)"
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
  "$venv/bin/pip" install --upgrade pip --quiet

  _info "Installing core dependencies (this may take a few minutes)…"
  "$venv/bin/pip" install -r "$CHALIE_HOME/app/backend/requirements.txt" --quiet

  # Voice dependencies (separate file, skipped if --disable-voice)
  if [[ "$_DISABLE_VOICE" != "true" ]]; then
    local voice_req="$CHALIE_HOME/app/backend/requirements-voice.txt"
    if [[ -f "$voice_req" ]]; then
      _info "Installing voice dependencies (STT/TTS)…"
      "$venv/bin/pip" install -r "$voice_req" --quiet 2>/dev/null || {
        _warn "Voice dependency install failed — voice will be unavailable"
        _warn "You can retry later: $venv/bin/pip install -r $voice_req"
      }
    fi
  fi

  _ok "Python environment ready"
  _info "Note: The embedding model (~400 MB) downloads on first 'chalie start', not now"
}

# ─── Install CLI Wrapper ─────────────────────────────────────────────────────
_install_cli() {
  _section "CLI Wrapper"
  mkdir -p "$CHALIE_BIN"

  cat > "$CHALIE_BIN/chalie" <<'CHALIE_CLI'
#!/usr/bin/env bash
CHALIE_HOME="${CHALIE_HOME:-$HOME/.chalie}"
VENV="$CHALIE_HOME/venv/bin/python"
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
    stop|restart|update|status|logs|help) _cmd="$1"; shift ;;
    *) _args+=("$1"); shift ;;
  esac
done
# Default command: start (if no named command given)
_cmd="${_cmd:-start}"

case "$_cmd" in
  start)
    _is_running && echo "Chalie is already running (PID $(cat "$PID_FILE"))" && exit 0
    mkdir -p "$DATA_DIR"
    CHALIE_DATA_DIR="$DATA_DIR" "$VENV" "$CHALIE_HOME/app/backend/run.py" \
      --port="$_port" --host="$_host" \
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
    curl -fsSL https://chalie.ai/install | CHALIE_UPDATE=1 bash
    ;;
  status)
    _is_running && echo "Running (PID $(cat "$PID_FILE"))" || echo "Not running"
    ;;
  logs)
    tail -f "$LOG_FILE"
    ;;
  help|*)
    echo "Usage: chalie [--port=N] [--host=H] [stop|restart|update|status|logs]"
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
main() {
  _parse_args "$@"

  local os arch
  os="$(_detect_os)"
  arch="$(_detect_arch)"

  if [[ "${CHALIE_UPDATE:-0}" != "1" ]]; then
    # Fresh install: show banner and interactive steps
    _banner
    printf "  Platform: %s / %s\n\n" "$os" "$arch"

    _check_python
    _check_docker
    _install_voice_deps
    _download_release
    _setup_venv
    _install_cli
    _print_success

    # Ask to start
    printf "\n"
    read -r -p "  Start Chalie now? [Y/n] " _start_reply
    printf "\n"
    if [[ "${_start_reply,,}" != "n" ]]; then
      "$CHALIE_BIN/chalie" start
    fi
  else
    # Update path: skip banner + interactive steps, preserve data
    _section "Updating Chalie"
    _info "Data directory preserved: $CHALIE_HOME/data"
    _download_release
    _setup_venv
    _install_cli
    _ok "Update complete"
    printf "\n  Run '%s start' to start the updated Chalie\n\n" "chalie"
  fi
}

main "$@"
