#!/usr/bin/env bash
# Chalie Installer
# Usage: curl -fsSL https://chalie.ai/install | bash
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

# Installer flags (parsed from args)
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
# Needed for native Python wheels (cryptography), sqlite-vec rebuild,
# envsubst (sqlite-vec template), and curl.
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
  if [[ -f "$CHALIE_HOME/app/backend/pyproject.toml" ]] || [[ -f "$CHALIE_HOME/app/backend/requirements.txt" ]]; then
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
_ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    _ok "uv already installed"
    return
  fi
  _info "Installing uv (fast Python package manager)…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  if ! command -v uv >/dev/null 2>&1; then
    _error "uv installation failed. Falling back to pip."
    return 1
  fi
  _ok "uv installed"
}

_setup_venv() {
  _section "Python Environment"
  local venv="$CHALIE_HOME/venv"
  local use_uv=true

  _ensure_uv || use_uv=false

  if [[ ! -d "$venv" ]]; then
    _info "Creating virtual environment…"
    if [[ "$use_uv" == "true" ]]; then
      uv venv "$venv" --python "$PYTHON"
    else
      "$PYTHON" -m venv "$venv"
    fi
  else
    _info "Reusing existing virtual environment"
  fi

  _info "Installing core dependencies (this may take a few minutes)…"
  if [[ "$use_uv" == "true" ]]; then
    uv pip install --python "$venv/bin/python" -e "$CHALIE_HOME/app/backend"
  else
    "$venv/bin/pip" install --upgrade pip
    "$venv/bin/pip" install -e "$CHALIE_HOME/app/backend"
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
  _ensure_uv
  _download_release
  _setup_venv
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
