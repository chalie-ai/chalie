"""
Browser Security — URL validation, SSRF guards, request interception.

Extends the read skill's SSRF protection with browser-specific additions:
  - Blocks dangerous schemes (file, data, blob, javascript, chrome, about)
  - Request interception on ALL sub-requests (iframes, XHR, fetch, images)
  - Per-page-load DNS cache to avoid 100+ DNS lookups on resource-heavy pages
  - Download blocking, dialog auto-dismiss
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, cast
from urllib.parse import urlparse

# SSRF policy lives in ONE place — services.ssrf. The browser layer re-exports
# the single blocklist and resolver so its request interceptor can never drift
# from what the read/web_fetch abilities enforce.
from services.ssrf import BLOCKED_NETS, resolve_and_check

if TYPE_CHECKING:
    from typing import Protocol

    class _Route(Protocol):
        class _Request(Protocol):
            url: str
        request: _Route._Request
        def abort(self) -> None: ...
        def continue_(self) -> None: ...

    class _Download(Protocol):
        def cancel(self) -> None: ...

    class _Page(Protocol):
        def route(self, pattern: str, handler: object) -> None: ...
        def on(self, event: str, handler: object) -> None: ...

logger = logging.getLogger(__name__)

__all__ = ["BLOCKED_NETS", "resolve_and_check", "DnsCache", "validate_url", "setup_page_security"]

_BLOCKED_SCHEMES = frozenset({
    "file", "ftp", "javascript", "data", "blob",
    "chrome", "chrome-extension", "about", "view-source",
})


# -- DNS cache (per page load, avoids repeated lookups) ------------------------

class DnsCache:
    """Thread-local DNS resolution cache with short TTL.

    Created per page load and discarded when the page closes.
    Prevents the request interceptor from doing 100+ DNS lookups
    on resource-heavy pages.
    """

    def __init__(self, ttl: float = 60.0) -> None:
        self._cache: dict[str, tuple[bool, str, float]] = {}
        self._ttl = ttl
        self._lock = threading.Lock()

    def check(self, hostname: str) -> tuple[bool, str]:
        """Return (ok, reason).  Cached per hostname."""
        now = time.time()
        with self._lock:
            entry = self._cache.get(hostname)
            if entry and (now - entry[2]) < self._ttl:
                return entry[0], entry[1]

        ok, reason = resolve_and_check(hostname)
        with self._lock:
            self._cache[hostname] = (ok, reason, now)
        return ok, reason


# -- URL validation ------------------------------------------------------------

def validate_url(url: str, dns_cache: DnsCache | None = None) -> tuple[bool, str]:
    """Validate a URL is safe for browser navigation.

    Args:
        url: The URL to validate.
        dns_cache: Optional DnsCache for sub-request validation.
                   Pass None for top-level navigation (fresh resolution).

    Returns:
        (ok, reason) tuple.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return False, f"Malformed URL: {url!r}"

    scheme = (parsed.scheme or "").lower()

    if scheme in _BLOCKED_SCHEMES:
        return False, f"Blocked scheme: {scheme}"

    if scheme not in ("http", "https"):
        return False, f"Unsupported scheme: {scheme}"

    hostname = parsed.hostname
    if not hostname:
        return False, "No hostname in URL"

    if dns_cache:
        return dns_cache.check(hostname)
    return resolve_and_check(hostname)


# -- Page-level security setup -------------------------------------------------

def setup_page_security(page: object, dns_cache: DnsCache | None = None) -> None:
    """Install security handlers on a Playwright page.

    Must be called BEFORE any navigation.  Installs:
      - Route interceptor (blocks private IPs, bad schemes on ALL sub-requests)
      - Download blocker (never save files to disk)
      - Dialog auto-dismiss (alert, confirm, prompt, beforeunload)
    """
    cache = dns_cache or DnsCache()

    def _intercept(route: object) -> None:
        req_url = cast("_Route", route).request.url
        ok, reason = validate_url(req_url, dns_cache=cache)
        if not ok:
            logger.debug("[BROWSER SEC] Blocked sub-request: %s (%s)", req_url, reason)
            try:
                cast("_Route", route).abort()
            except Exception:
                pass
            return
        try:
            cast("_Route", route).continue_()
        except Exception:
            pass

    cast("_Page", page).route("**/*", _intercept)

    # Block file downloads — never write to disk
    cast("_Page", page).on("download", lambda dl: cast("_Download", dl).cancel())

    # Dialogs are owned by tools/browser/session.PageSession, which records the
    # message for the action diff and then dismisses. (A second dismiss here
    # would raise on the already-handled dialog.)
