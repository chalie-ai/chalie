"""
Browser Pool — dedicated-thread headless chromium lifecycle management.

All browser work runs on a single daemon thread.  External code submits work
callables via submit()/execute() and receives results through Futures.

Why single-thread: page sessions keep per-tab state (event counters, the
dialog and popup they last saw) that is not thread-safe.  One thread owns the
browser and every session; everyone else queues work onto it.

Lifecycle:
  - Browser launches lazily on first work item
  - Idles out after 5 minutes of no activity
  - Re-launches transparently on next request, or after it crashed
  - Clean shutdown via atexit or SIGTERM (daemon thread)
"""

from __future__ import annotations

import atexit
import logging
import queue
import threading
import time
from concurrent.futures import Future
from typing import Callable, cast

from tools.browser.cdp import Browser

logger = logging.getLogger(__name__)

_IDLE_TTL = 300          # 5 min idle → close browser
_QUEUE_POLL = 30         # seconds between idle checks


class BrowserPool:
    """Dedicated-thread browser pool.  Thread-safe facade for the browser."""

    def __init__(self) -> None:
        self._queue: queue.Queue[object] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._running = False
        # Counters for observability
        self._ops_total = 0
        self._ops_failed = 0
        self._browser_launches = 0

    # -- Public API (called from any thread) -----------------------------------

    def submit(self, fn: Callable[..., object], *args: object, **kwargs: object) -> Future[object]:
        """Submit a callable to the browser thread; ``fn`` receives the live Browser first."""
        self._ensure_thread()
        future: Future[object] = Future()
        self._queue.put((fn, args, kwargs, future))
        return future

    def execute(self, fn: Callable[..., object], *args: object, timeout: float = 60, **kwargs: object) -> object:
        """Submit + block until result.  Raises on timeout or exception."""
        return self.submit(fn, *args, **kwargs).result(timeout=timeout)

    def shutdown(self) -> None:
        """Signal the browser thread to stop.  Blocks until drained."""
        self._running = False
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    def status(self) -> dict[str, object]:
        """Return pool health for the /api/system/health endpoint."""
        return {
            "thread_alive": self._thread is not None and self._thread.is_alive(),
            "queue_depth": self._queue.qsize(),
            "ops_total": self._ops_total,
            "ops_failed": self._ops_failed,
            "browser_launches": self._browser_launches,
        }

    # -- Internal --------------------------------------------------------------

    def _ensure_thread(self) -> None:
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._running = True
                self._thread = threading.Thread(
                    target=self._loop,
                    name="browser-pool",
                    daemon=True,
                )
                self._thread.start()
                logger.info("[BROWSER POOL] Dedicated thread started")

    def _loop(self) -> None:
        """Main loop — owns the browser process.  Never called externally."""
        browser: Browser | None = None
        last_used = time.time()

        while self._running:
            # Drain queue with timeout so we can check idle
            try:
                item = self._queue.get(timeout=_QUEUE_POLL)
            except queue.Empty:
                if browser is not None and (time.time() - last_used) > _IDLE_TTL:
                    logger.info("[BROWSER POOL] Idle timeout — closing browser")
                    browser.close()
                    browser = None
                continue

            # Shutdown sentinel
            if item is None:
                break

            fn, args, kwargs, future = cast(
                "tuple[Callable[..., object], tuple[object, ...], dict[str, object], Future[object]]",
                item,
            )

            try:
                if browser is None or not browser.connected:
                    if browser is not None:
                        logger.warning("[BROWSER POOL] Browser disconnected — re-launching")
                        browser.close()  # reaps the dead process and removes its profile
                    browser = Browser()
                    self._browser_launches += 1
                    logger.info("[BROWSER POOL] Chromium launched")

                last_used = time.time()
                self._ops_total += 1
                future.set_result(fn(browser, *args, **kwargs))

            except Exception as e:
                self._ops_failed += 1
                future.set_exception(e)

        # Loop exiting — clean shutdown
        if browser is not None:
            browser.close()
        logger.info("[BROWSER POOL] Thread stopped")


# -- Module-level singleton ----------------------------------------------------

_pool: BrowserPool | None = None
_pool_lock = threading.Lock()


def get_pool() -> BrowserPool:
    """Return the global BrowserPool singleton (lazy-created, atexit-registered)."""
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = BrowserPool()
                atexit.register(_pool.shutdown)
    return _pool


def shutdown() -> None:
    """Module-level shutdown — safe to call even if pool was never created."""
    if _pool is not None:
        _pool.shutdown()
