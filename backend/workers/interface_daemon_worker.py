"""
Interface daemon watcher — monitors frontend/_interfaces/ for daemon directories,
auto-starts them as background Deno subprocesses, and restarts on crash.

Each subdirectory in frontend/_interfaces/ is expected to contain a daemon.ts
entry point. The watcher assigns each daemon a unique port (starting from 4001),
passes --gateway and --port args, and monitors the subprocess.

New daemons appearing in the folder are auto-started. Removed directories
cause the corresponding subprocess to be stopped.
"""

import logging
import shutil
import subprocess
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

_SCAN_INTERVAL = 5          # seconds between filesystem scans
_RESTART_DELAY = 3          # seconds to wait before restarting a crashed daemon
_PORT_BASE = 4001           # first daemon gets this port; increments per daemon
_MAX_RESTART_BACKOFF = 60   # max restart delay after repeated crashes

# ── State ─────────────────────────────────────────────────────────────────────

_lock = threading.Lock()
# daemon_name → {proc, port, dir, restart_count, last_crash}
_daemons: dict[str, dict] = {}
# Tracks port assignments so they're stable across restarts
_port_assignments: dict[str, int] = {}
_next_port = _PORT_BASE


def _interfaces_dir() -> Path:
    """Return the absolute path to frontend/_interfaces/."""
    backend_dir = Path(__file__).resolve().parent.parent
    return backend_dir.parent / "frontend" / "_interfaces"


def _find_entry_point(daemon_dir: Path) -> Path | None:
    """Find the daemon entry point in a directory."""
    for name in ("daemon.ts", "main.ts", "mod.ts", "index.ts"):
        entry = daemon_dir / name
        if entry.is_file():
            return entry
    return None


def _deno_available() -> bool:
    """Check if deno is on PATH."""
    return shutil.which("deno") is not None


def _assign_port(daemon_name: str) -> int:
    """Get or assign a stable port for a daemon."""
    global _next_port
    if daemon_name in _port_assignments:
        return _port_assignments[daemon_name]
    port = _next_port
    _port_assignments[daemon_name] = port
    _next_port += 1
    return port


def _start_daemon(daemon_name: str, daemon_dir: Path, gateway_url: str) -> dict | None:
    """Start a daemon subprocess. Returns the daemon state dict or None."""
    entry = _find_entry_point(daemon_dir)
    if not entry:
        logger.debug("[DaemonWatcher] No entry point in %s — skipping", daemon_dir)
        return None

    port = _assign_port(daemon_name)
    data_dir = daemon_dir / "data"

    cmd = [
        "deno", "run",
        "--allow-net", "--allow-read", "--allow-write", "--allow-env",
        "--reload=jsr:@chalie",
        str(entry),
        f"--gateway={gateway_url}",
        f"--port={port}",
        f"--data-dir={data_dir}",
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(daemon_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        logger.info(
            "[DaemonWatcher] Started '%s' (pid=%d, port=%d) → %s",
            daemon_name, proc.pid, port, entry.name,
        )

        # Start a log drain thread so stdout doesn't block
        log_thread = threading.Thread(
            target=_drain_output,
            args=(daemon_name, proc),
            daemon=True,
            name=f"daemon-log-{daemon_name}",
        )
        log_thread.start()

        return {
            "proc": proc,
            "port": port,
            "dir": str(daemon_dir),
            "restart_count": 0,
            "last_crash": 0.0,
        }
    except FileNotFoundError:
        logger.error("[DaemonWatcher] 'deno' not found — cannot start daemons")
        return None
    except Exception as e:
        logger.error("[DaemonWatcher] Failed to start '%s': %s", daemon_name, e)
        return None


def _drain_output(daemon_name: str, proc: subprocess.Popen):
    """Read subprocess stdout/stderr and forward to Python logging."""
    try:
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                logger.info("[daemon:%s] %s", daemon_name, line)
    except Exception:
        pass


def _stop_daemon(daemon_name: str):
    """Stop a running daemon subprocess."""
    state = _daemons.get(daemon_name)
    if not state:
        return
    proc = state["proc"]
    if proc.poll() is None:
        logger.info("[DaemonWatcher] Stopping '%s' (pid=%d)", daemon_name, proc.pid)
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)
    del _daemons[daemon_name]


def _restart_delay(state: dict) -> float:
    """Calculate restart delay with exponential backoff."""
    count = state["restart_count"]
    delay = min(_RESTART_DELAY * (2 ** count), _MAX_RESTART_BACKOFF)
    return delay


# ── Main loop ─────────────────────────────────────────────────────────────────

def interface_daemon_worker():
    """Worker function — monitors _interfaces/ and manages daemon lifecycle."""
    import runtime_config

    interfaces_dir = _interfaces_dir()

    if not _deno_available():
        logger.warning("[DaemonWatcher] Deno not found on PATH — daemon watcher disabled")
        # Sleep forever so WorkerManager doesn't keep restarting us
        while True:
            time.sleep(3600)

    # Gateway URL is Chalie's own address (daemons register via POST /register)
    port = runtime_config.get("port", 31025)
    gateway_url = f"http://127.0.0.1:{port}"

    logger.info("[DaemonWatcher] Watching %s (gateway=%s)", interfaces_dir, gateway_url)

    while True:
        try:
            _scan_and_reconcile(interfaces_dir, gateway_url)
        except Exception as e:
            logger.error("[DaemonWatcher] Scan error: %s", e)

        time.sleep(_SCAN_INTERVAL)


def _discover_daemon_dirs(interfaces_dir: Path) -> dict:
    """Return a mapping of daemon name → path for all valid daemon directories."""
    current_dirs: dict[str, Path] = {}
    if interfaces_dir.is_dir():
        for entry in interfaces_dir.iterdir():
            if entry.is_dir() and not entry.name.startswith((".", "_")):
                if _find_entry_point(entry):
                    current_dirs[entry.name] = entry
    return current_dirs


def _health_check_daemon(name: str, state: dict, gateway_url: str) -> None:
    """Check whether a running daemon has crashed and restart it when due.

    Must be called while holding ``_lock``.
    """
    proc = state["proc"]
    if proc.poll() is None:
        return  # still running

    exit_code = proc.returncode
    logger.warning(
        "[DaemonWatcher] '%s' exited (code=%s, restarts=%d)",
        name, exit_code, state["restart_count"],
    )

    delay = _restart_delay(state)
    now = time.time()
    time_since_crash = now - state["last_crash"] if state["last_crash"] else delay

    if time_since_crash < delay:
        return  # Too soon — skip this cycle

    _restart_daemon(name, state, gateway_url, now)


def _restart_daemon(name: str, state: dict, gateway_url: str, now: float) -> None:
    """Restart a crashed daemon.  Must be called while holding ``_lock``."""
    daemon_dir = Path(state["dir"])
    restart_count = state["restart_count"] + 1
    del _daemons[name]

    new_state = _start_daemon(name, daemon_dir, gateway_url)
    if new_state:
        new_state["restart_count"] = restart_count
        new_state["last_crash"] = now
        _daemons[name] = new_state
        logger.info(
            "[DaemonWatcher] Restarted '%s' (attempt #%d, next backoff=%.0fs)",
            name, restart_count, _restart_delay(new_state),
        )


def _scan_and_reconcile(interfaces_dir: Path, gateway_url: str):
    """Scan the directory and start/stop/restart daemons as needed."""
    current_dirs = _discover_daemon_dirs(interfaces_dir)

    with _lock:
        # Stop daemons whose directories were removed
        removed = set(_daemons.keys()) - set(current_dirs.keys())
        for name in removed:
            logger.info("[DaemonWatcher] Directory removed — stopping '%s'", name)
            _stop_daemon(name)

        # Start new daemons
        for name, path in current_dirs.items():
            if name not in _daemons:
                state = _start_daemon(name, path, gateway_url)
                if state:
                    _daemons[name] = state

        # Check running daemons for crashes → restart
        for name in list(_daemons.keys()):
            _health_check_daemon(name, _daemons[name], gateway_url)


def get_daemon_status() -> list[dict]:
    """Return status of all managed daemons (for observability)."""
    with _lock:
        result = []
        for name, state in _daemons.items():
            proc = state["proc"]
            result.append({
                "name": name,
                "port": state["port"],
                "pid": proc.pid,
                "alive": proc.poll() is None,
                "restart_count": state["restart_count"],
            })
        return result
