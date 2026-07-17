"""
Browser Pool — Dedicated-thread Playwright lifecycle management.

All Playwright operations run on a single daemon thread.  External code submits
work callables via submit()/execute() and receives results through Futures.

Why single-thread: Playwright's sync_api binds an event loop to the calling
thread.  Multi-thread access causes deadlocks or cross-thread exceptions.
One thread owns the browser; everyone else queues work onto it.

Lifecycle:
  - Browser launches lazily on first work item
  - Idles out after 5 minutes of no activity
  - Re-launches transparently on next request
  - Clean shutdown via atexit or SIGTERM (daemon thread)
"""

from __future__ import annotations

import atexit
import logging
import queue
import threading
import time
from concurrent.futures import Future
from typing import TYPE_CHECKING, Callable, cast

if TYPE_CHECKING:
    from typing import Protocol

    class _BrowserLike(Protocol):
        def close(self) -> None: ...
        def is_connected(self) -> bool: ...

    class _ChromiumLike(Protocol):
        def launch(self, *, headless: bool, args: list[str]) -> object: ...

    class _PlaywrightLike(Protocol):
        chromium: _ChromiumLike
        def stop(self) -> None: ...

logger = logging.getLogger(__name__)

_IDLE_TTL = 300          # 5 min idle → close browser
_QUEUE_POLL = 30         # seconds between idle checks
_LAUNCH_ARGS = [
    "--disable-gpu",
    "--disable-dev-shm-usage",       # Docker /dev/shm is only 64MB
    "--disable-extensions",
    "--no-first-run",
    "--disable-background-networking",
    "--disable-default-apps",
    "--disable-sync",
    "--disable-translate",
    "--metrics-recording-only",
    "--no-default-browser-check",
]


class BrowserPool:
    """Dedicated-thread browser pool.  Thread-safe facade for Playwright."""

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
        """Submit a callable to the browser thread.

        ``fn`` receives the live Playwright Browser as its first argument::

            def work(browser, url):
                ctx = browser.new_context()
                page = ctx.new_page()
                page.goto(url)
                text = page.inner_text("body")
                page.close()
                ctx.close()
                return text

            future = pool.submit(work, "https://example.com")
            result = future.result(timeout=60)
        """
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
        """Main loop — owns the Playwright runtime.  Never called externally."""
        pw = None
        browser = None
        last_used = time.time()

        while self._running:
            # Drain queue with timeout so we can check idle
            try:
                item = self._queue.get(timeout=_QUEUE_POLL)
            except queue.Empty:
                if browser and (time.time() - last_used) > _IDLE_TTL:
                    logger.info("[BROWSER POOL] Idle timeout — closing browser")
                    browser, pw = self._close(browser, pw)
                continue

            # Shutdown sentinel
            if item is None:
                break

            browser, pw, last_used = self._run_queued_item(item, browser, pw, last_used)

        # Loop exiting — clean shutdown
        self._close(browser, pw)
        logger.info("[BROWSER POOL] Thread stopped")

    def _run_queued_item(
        self, item: object, browser: object, pw: object, last_used: float
    ) -> tuple[object, object, float]:
        """Execute one queued work item; lazy-launches Playwright/Chromium as needed.

        Returns the (possibly updated) ``(browser, pw, last_used)`` triple.
        """
        fn, args, kwargs, future = cast(
            "tuple[Callable[..., object], tuple[object, ...], dict[str, object], Future[object]]",
            item,
        )

        try:
            # Lazy-launch Playwright + Chromium
            if pw is None:
                from playwright.sync_api import sync_playwright
                pw = sync_playwright().start()

            if browser is None or not cast("_BrowserLike", browser).is_connected():
                browser = cast("_PlaywrightLike", pw).chromium.launch(headless=True, args=_LAUNCH_ARGS)
                self._browser_launches += 1
                logger.info("[BROWSER POOL] Chromium launched")

            last_used = time.time()
            self._ops_total += 1
            result = fn(browser, *args, **kwargs)
            future.set_result(result)

        except Exception as e:
            self._ops_failed += 1
            # If browser crashed, discard the reference so next op re-launches
            if browser and not cast("_BrowserLike", browser).is_connected():
                logger.warning("[BROWSER POOL] Browser disconnected — will re-launch")
                browser = None
            future.set_exception(e)

        return browser, pw, last_used

    @staticmethod
    def _close(browser: object, pw: object) -> tuple[None, None]:
        """Close browser and Playwright, swallowing errors."""
        try:
            if browser:
                cast("_BrowserLike", browser).close()
        except Exception:
            pass
        try:
            if pw:
                cast("_PlaywrightLike", pw).stop()
        except Exception:
            pass
        return None, None


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
