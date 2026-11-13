# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Headless chromium driven over the Chrome DevTools Protocol.

The browser runs with ``--remote-debugging-pipe``: it reads commands on fd 3
and writes replies and events on fd 4, each message one NUL-terminated JSON
object. One reader thread owns fd 4. It resolves pending replies, dismisses a
JavaScript dialog the moment it opens (an open dialog blocks every other
command on its page), hands a watched session's screencast frames and target
lifecycle events to its listener, and queues each attached session's other
events for the pool thread to drain.

The browser keeps one persistent profile under the data directory, so cookies
and logins outlive both a run and a relaunch.
"""

from __future__ import annotations

import collections
import fcntl
import itertools
import json
import logging
import os
import signal
import threading
import time
from typing import Callable, NamedTuple, TypeVar

from services.file_mapper_service import FileMapperService

logger = logging.getLogger(__name__)

COMMAND_TIMEOUT = 30.0
_STARTUP_TIMEOUT = 30.0
_CLOSE_GRACE = 5.0
_STDERR_TAIL = 1_000
_POSTED_MEMORY = 256       # posted commands remembered by id, to name the one an error reply is for
CLOSED = "Target page, context or browser has been closed"

# The switch set the browser has always run with: the automation defaults
# (no background networking, updaters, throttling or first-run UI; headless
# with hidden scrollbars and muted audio) followed by Chalie's own additions.
_LAUNCH_ARGS = (
    "--disable-field-trial-config",
    "--disable-background-networking",
    "--disable-background-timer-throttling",
    "--disable-backgrounding-occluded-windows",
    "--disable-back-forward-cache",
    "--disable-breakpad",
    "--disable-client-side-phishing-detection",
    "--disable-component-extensions-with-background-pages",
    "--disable-component-update",
    "--no-default-browser-check",
    "--disable-default-apps",
    "--disable-dev-shm-usage",           # Docker /dev/shm is only 64MB
    "--disable-edgeupdater",
    "--disable-extensions",
    "--disable-features=" + ",".join((
        "AvoidUnnecessaryBeforeUnloadCheckSync",
        "BoundaryEventDispatchTracksNodeRemoval",
        "DestroyProfileOnBrowserClose",
        "DialMediaRouteProvider",
        "GlobalMediaControls",
        "HttpsUpgrades",
        "LensOverlay",
        "MediaRouter",
        "PaintHolding",
        "ThirdPartyStoragePartitioning",
        "BlockOriginHeaderModificationOnRedirect",
        "Translate",
        "AutoDeElevate",
        "OptimizationHints",
        "msForceBrowserSignIn",
        "msEdgeUpdateLaunchServicesPreferredVersion",
    )),
    "--enable-features=CDPScreenshotNewSurface",
    "--allow-pre-commit-input",
    "--disable-hang-monitor",
    "--disable-ipc-flooding-protection",
    "--disable-popup-blocking",
    "--disable-prompt-on-repost",
    "--disable-renderer-backgrounding",
    "--disable-updater-scheduler",
    "--force-color-profile=srgb",
    "--metrics-recording-only",
    "--no-first-run",
    "--password-store=basic",
    "--use-mock-keychain",
    "--no-service-autorun",
    "--export-tagged-pdf",
    "--disable-search-engine-choice-screen",
    "--unsafely-disable-devtools-self-xss-warnings",
    "--edge-skip-compat-layer-relaunch",
    "--disable-infobars",
    "--disable-sync",
    "--enable-unsafe-swiftshader",
    "--headless",
    "--hide-scrollbars",
    "--mute-audio",
    "--blink-settings=primaryHoverType=2,availableHoverTypes=2,primaryPointerType=4,availablePointerTypes=4",
    "--no-sandbox",
    "--disable-gpu",
    "--disable-translate",
)

_T = TypeVar("_T", str, int, float, bool)

# What a session's listener receives instead of (frames) or as well as (the
# rest) the pool-thread queue: enough to stream the page, keep its tab's
# url, title and liveness current, and log its navigations.
_WATCHED = frozenset({
    "Page.screencastFrame",
    "Page.domContentEventFired",
    "Page.loadEventFired",
    "Page.frameNavigated",
    "Page.frameRequestedNavigation",
    "Target.targetCreated",
    "Target.targetInfoChanged",
    "Target.targetDestroyed",
    "Target.targetCrashed",
    "Target.detachedFromTarget",
})


class BrowserError(Exception):
    """A browser command or page action failed. The message is shown to the model."""


class Event(NamedTuple):
    """One CDP event, stamped with ``time.monotonic()`` when the reader received it."""

    method: str
    params: dict[str, object]
    at: float


Listener = Callable[[Event], None]


# ── JSON narrowing ────────────────────────────────────────────────────────────

def get(payload: dict[str, object], key: str, kind: type[_T]) -> _T:
    """``payload[key]`` checked to be a *kind* — protocol drift fails here, loudly."""
    value = payload.get(key)
    if not isinstance(value, kind):
        raise BrowserError(f"Protocol error: no {kind.__name__} {key!r} in {str(payload)[:200]}")
    return value


def get_dict(payload: dict[str, object], key: str) -> dict[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise BrowserError(f"Protocol error: no object {key!r} in {str(payload)[:200]}")
    return {str(k): v for k, v in value.items()}


def get_list(payload: dict[str, object], key: str) -> list[object]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise BrowserError(f"Protocol error: no array {key!r} in {str(payload)[:200]}")
    return list(value)


def get_objects(payload: dict[str, object], key: str) -> list[dict[str, object]]:
    items = get_list(payload, key)
    objects = [{str(k): v for k, v in item.items()} for item in items if isinstance(item, dict)]
    if len(objects) != len(items):
        raise BrowserError(f"Protocol error: non-object in {key!r} of {str(payload)[:200]}")
    return objects


# ── Executable discovery ─────────────────────────────────────────────────────

def find_executable() -> str:
    """Chalie's own chrome-headless-shell, unpacked into the install by the installer."""
    executable = FileMapperService.get_browser_executable_path()
    if not os.access(executable, os.X_OK):
        raise RuntimeError(
            f"Headless chromium executable is missing: {executable}. "
            "Re-run the installer to download it."
        )
    return str(executable)


def _fd_above_4(fd: int) -> int:
    """Move *fd* above 4 so the child's dup2 onto fds 3 and 4 can never clobber it."""
    try:
        return fcntl.fcntl(fd, fcntl.F_DUPFD_CLOEXEC, 5)
    finally:
        os.close(fd)


# ── Browser process + pipe transport ─────────────────────────────────────────

class Browser:
    """One headless chromium process and its CDP pipe. Thread-safe."""

    def __init__(self) -> None:
        executable = find_executable()
        profile = FileMapperService.get_browser_profile_path()
        profile.mkdir(mode=0o700, parents=True, exist_ok=True)
        stderr_path = os.path.join(profile, "stderr.log")
        command_read, self._out = os.pipe()
        self._in, event_write = os.pipe()
        child_in, child_out = _fd_above_4(command_read), _fd_above_4(event_write)
        try:
            # posix_spawn, not fork+preexec: forking a multi-threaded server can
            # deadlock the child before exec.
            self._pid = os.posix_spawn(
                executable,
                [executable, *_LAUNCH_ARGS, f"--user-data-dir={profile}",
                 "--remote-debugging-pipe", "--no-startup-window"],
                os.environ,
                file_actions=[
                    (os.POSIX_SPAWN_OPEN, 0, os.devnull, os.O_RDONLY, 0),
                    (os.POSIX_SPAWN_OPEN, 1, os.devnull, os.O_WRONLY, 0),
                    (os.POSIX_SPAWN_OPEN, 2, stderr_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600),
                    (os.POSIX_SPAWN_DUP2, child_in, 3),
                    (os.POSIX_SPAWN_DUP2, child_out, 4),
                ],
                setpgroup=0,  # own process group: a forced kill takes the renderers too
            )
        except OSError as e:
            for fd in (self._in, self._out):
                os.close(fd)
            raise RuntimeError(f"Headless chromium executable could not start: {e}") from e
        finally:
            os.close(child_in)
            os.close(child_out)

        self._ids = itertools.count(1)
        self._write_lock = threading.Lock()
        self._cv = threading.Condition()
        self._waiting: set[int] = set()
        self._replies: dict[int, dict[str, object]] = {}
        self._posted: dict[int, str] = {}        # posted command id -> method, oldest first
        self._queues: dict[str, collections.deque[Event]] = {}
        self._targets: dict[str, str] = {}      # attached target id -> session id
        self._listeners: dict[str, Listener] = {}
        self._connected = True
        self._closed = False
        self._reader = threading.Thread(target=self._read_loop, name="browser-cdp-reader", daemon=True)
        self._reader.start()
        try:
            self.send("Browser.getVersion", timeout=_STARTUP_TIMEOUT)
            self.send("Target.setDiscoverTargets", {"discover": True})
            self.send("Browser.setDownloadBehavior", {"behavior": "deny"})
        except BrowserError as e:
            with open(stderr_path, "rb") as fh:
                tail = fh.read().decode(errors="replace")[-_STDERR_TAIL:].strip()
            self.close()
            raise RuntimeError(f"Headless chromium failed to start: {e}\n{tail}".strip()) from e

    @property
    def connected(self) -> bool:
        return self._connected

    def send(self, method: str, params: dict[str, object] | None = None, *,
             session_id: str | None = None, timeout: float = COMMAND_TIMEOUT) -> dict[str, object]:
        """Send one command and block for its result."""
        with self._cv:
            if not self._connected:
                raise BrowserError(f"{method}: {CLOSED}")
            msg_id = next(self._ids)
            self._waiting.add(msg_id)
        try:
            self._write(msg_id, method, params, session_id)
        except OSError:
            pass  # the pipe is gone; the reader's EOF marks us disconnected below
        with self._cv:
            self._cv.wait_for(lambda: msg_id in self._replies or not self._connected, timeout)
            self._waiting.discard(msg_id)
            reply = self._replies.pop(msg_id, None)
        if reply is None:
            if self._connected:
                raise BrowserError(f"{method}: Timeout {round(timeout * 1000)}ms exceeded.")
            raise BrowserError(f"{method}: {CLOSED}")
        error = reply.get("error")
        if isinstance(error, dict):
            raise BrowserError(f"Protocol error ({method}): {error.get('message')}")
        result = reply.get("result")
        return {str(k): v for k, v in result.items()} if isinstance(result, dict) else {}

    def post(self, method: str, params: dict[str, object] | None = None, *,
             session_id: str | None = None) -> None:
        """Send one command without waiting for its reply; a lost browser drops it, an error reply is logged."""
        with self._cv:
            msg_id = next(self._ids)
            self._posted[msg_id] = method
            if len(self._posted) > _POSTED_MEMORY:
                del self._posted[next(iter(self._posted))]
        try:
            self._write(msg_id, method, params, session_id)
        except OSError:
            pass  # the pipe is gone; the reader's EOF marks us disconnected

    def attach(self, target_id: str, *, queue: bool = True) -> str:
        """Attach to *target_id* as a flat session; *queue* keeps its events for :meth:`events`."""
        session_id = get(self.send("Target.attachToTarget", {"targetId": target_id, "flatten": True}),
                         "sessionId", str)
        with self._cv:
            if queue:
                self._queues[session_id] = collections.deque()
            self._targets[target_id] = session_id
        return session_id

    def watch(self, session_id: str, listener: Listener | None) -> None:
        """Hand *session_id*'s watched events to *listener* on the reader thread; ``None`` stops it.

        A watched session's screencast frames never reach its queue."""
        with self._cv:
            if listener is None:
                self._listeners.pop(session_id, None)
            else:
                self._listeners[session_id] = listener

    def release(self, session_id: str) -> None:
        """Stop queueing and watching *session_id*'s events."""
        with self._cv:
            self._queues.pop(session_id, None)
            self._listeners.pop(session_id, None)
            for target_id in [t for t, s in self._targets.items() if s == session_id]:
                del self._targets[target_id]

    def events(self, session_id: str, timeout: float = 0.0) -> list[Event]:
        """Drain *session_id*'s queued events, first waiting up to *timeout* s for one."""
        with self._cv:
            queue = self._queues.get(session_id)
            if queue is None:
                return []
            if timeout > 0:
                self._cv.wait_for(lambda: bool(queue) or not self._connected, timeout)
            drained = list(queue)
            queue.clear()
            return drained

    def close(self) -> None:
        """Browser.close, then SIGKILL the process group after a grace period."""
        if self._closed:
            return
        self._closed = True
        if self._connected:
            try:
                self.send("Browser.close", timeout=_CLOSE_GRACE)
            except BrowserError:
                pass  # the reply races the pipe closing; the exit wait below is what counts
        if not self._exited_within(_CLOSE_GRACE):
            logger.warning("[BROWSER] chromium ignored Browser.close — killing it")
            try:
                os.killpg(self._pid, signal.SIGKILL)
                os.waitpid(self._pid, 0)
            except (ProcessLookupError, ChildProcessError):
                pass  # exited between the last poll and the kill
        self._reader.join(_CLOSE_GRACE)
        with self._write_lock:
            # A late post must hit a closed descriptor, never a reused one.
            os.close(self._out)
            self._out = -1
        os.close(self._in)

    # ── Internals ────────────────────────────────────────────────────────────

    def _exited_within(self, seconds: float) -> bool:
        deadline = time.monotonic() + seconds
        while True:
            try:
                pid, _ = os.waitpid(self._pid, os.WNOHANG)
            except ChildProcessError:
                return True
            if pid:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)

    def _write(self, msg_id: int, method: str, params: dict[str, object] | None,
               session_id: str | None) -> None:
        message: dict[str, object] = {"id": msg_id, "method": method, "params": params or {}}
        if session_id:
            message["sessionId"] = session_id
        data = memoryview(json.dumps(message).encode() + b"\0")
        with self._write_lock:
            while data:
                data = data[os.write(self._out, data):]

    def _read_loop(self) -> None:
        buf = bytearray()
        try:
            while chunk := os.read(self._in, 1 << 20):
                # Search only the new bytes: a multi-megabyte screenshot reply
                # arrives in many chunks and must not be rescanned each time.
                start = len(buf)
                buf += chunk
                pos = 0
                end = buf.find(b"\0", start)
                while end != -1:
                    self._route(json.loads(buf[pos:end]))
                    pos = end + 1
                    end = buf.find(b"\0", pos)
                del buf[:pos]
        finally:
            with self._cv:
                self._connected = False
                self._cv.notify_all()

    def _route(self, message: object) -> None:
        if not isinstance(message, dict):
            return
        msg_id = message.get("id")
        if isinstance(msg_id, int):
            with self._cv:
                if msg_id in self._waiting:
                    self._replies[msg_id] = message
                    self._cv.notify_all()
                    return
                command = self._posted.pop(msg_id, "command")
            error = message.get("error")
            if isinstance(error, dict):
                logger.warning("[BROWSER] %s (#%d) failed: %s", command, msg_id, error.get("message"))
            return
        method, params, session_id = message.get("method"), message.get("params"), message.get("sessionId")
        if not isinstance(method, str) or not isinstance(params, dict):
            return
        if method == "Page.javascriptDialogOpening" and isinstance(session_id, str):
            try:
                self._write(next(self._ids), "Page.handleJavaScriptDialog", {"accept": False}, session_id)
            except OSError:
                pass  # the browser is gone; the EOF that follows marks us disconnected
        event = Event(method, params, time.monotonic())
        listeners: list[Listener] = []
        with self._cv:
            owner: str | None
            subject: str | None
            if isinstance(session_id, str):
                owner = subject = session_id
            else:
                owner = self._attached(method, params, "openerId")
                subject = self._attached(method, params, "targetId")
            if method in _WATCHED:
                listeners = [self._listeners[s] for s in dict.fromkeys((owner, subject))
                             if s is not None and s in self._listeners]
            queue = self._queues.get(owner) if owner else None
            if queue is not None and not (listeners and method == "Page.screencastFrame"):
                queue.append(event)
                self._cv.notify_all()
        for listener in listeners:
            try:
                listener(event)
            except Exception:
                logger.exception("[BROWSER] %s listener failed", method)

    def _attached(self, method: str, params: dict[str, object], key: str) -> str | None:
        """The attached session a browser-level Target event concerns, if any:
        ``openerId`` names the page that opened it, ``targetId`` the target itself."""
        if method == "Target.detachedFromTarget":
            session_id = params.get("sessionId")
            return session_id if isinstance(session_id, str) else None
        info = params.get("targetInfo")
        target_id = info.get(key) if isinstance(info, dict) else params.get("targetId")
        return self._targets.get(target_id) if isinstance(target_id, str) else None
