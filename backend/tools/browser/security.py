"""
Browser Security — URL validation, SSRF guards, request interception.

Extends the read skill's SSRF protection with browser-specific additions:
  - Blocks dangerous schemes (file, data, blob, javascript, chrome, about)
  - Request interception on ALL sub-requests (iframes, XHR, fetch, images)
  - Per-page-load DNS cache to avoid 100+ DNS lookups on resource-heavy pages
  - Download blocking, dialog auto-dismiss
"""

import ipaddress
import logging
import socket
import threading
import time
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Same ranges as read_skill.py — single source of truth for SSRF policy
_BLOCKED_NETS = [
    ipaddress.ip_network('127.0.0.0/8'),
    ipaddress.ip_network('10.0.0.0/8'),
    ipaddress.ip_network('172.16.0.0/12'),
    ipaddress.ip_network('192.168.0.0/16'),
    ipaddress.ip_network('169.254.0.0/16'),   # link-local / AWS metadata
    ipaddress.ip_network('::1/128'),           # IPv6 loopback
    ipaddress.ip_network('fc00::/7'),          # IPv6 unique-local
]

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

    def __init__(self, ttl: float = 60.0):
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

        ok, reason = _resolve_and_check(hostname)
        with self._lock:
            self._cache[hostname] = (ok, reason, now)
        return ok, reason


def _resolve_and_check(hostname: str) -> tuple[bool, str]:
    """Resolve hostname and check against blocked networks."""
    if not hostname:
        return False, "Empty hostname"
    try:
        results = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        return False, f"DNS resolution failed: {hostname}"

    for family, _, _, _, addr in results:
        ip = ipaddress.ip_address(addr[0])
        for net in _BLOCKED_NETS:
            if ip in net:
                return False, f"Blocked IP range: {ip} ({hostname})"
    return True, ""


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
    return _resolve_and_check(hostname)


# -- Page-level security setup -------------------------------------------------

def setup_page_security(page, dns_cache: DnsCache | None = None):
    """Install security handlers on a Playwright page.

    Must be called BEFORE any navigation.  Installs:
      - Route interceptor (blocks private IPs, bad schemes on ALL sub-requests)
      - Download blocker (never save files to disk)
      - Dialog auto-dismiss (alert, confirm, prompt, beforeunload)
    """
    cache = dns_cache or DnsCache()

    def _intercept(route):
        req_url = route.request.url
        ok, reason = validate_url(req_url, dns_cache=cache)
        if not ok:
            logger.debug("[BROWSER SEC] Blocked sub-request: %s (%s)", req_url, reason)
            try:
                route.abort()
            except Exception:
                pass
            return
        try:
            route.continue_()
        except Exception:
            pass

    page.route("**/*", _intercept)

    # Block file downloads — never write to disk
    page.on("download", lambda dl: dl.cancel())

    # Dialogs are owned by tools/browser/session.PageSession, which records the
    # message for the action diff and then dismisses. (A second dismiss here
    # would raise on the already-handled dialog.)
