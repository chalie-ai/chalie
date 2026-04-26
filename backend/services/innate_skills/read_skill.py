"""
Read Skill — Unified text reader for URLs and local files.

Fetches and extracts clean text from any source:
  - Web pages and articles (URL)
  - PDFs, DOCX, PPTX, HTML files, Markdown, plain text (filesystem path)

Text extraction delegates to services.text_extractor (shared with DocumentProcessingService).

Security:
  - SSRF guard: blocks requests to private/internal IP ranges (resolved, not string-matched)
  - File guard: blocks reads from system paths (/etc, /proc, /dev, /sys, /var/run)
"""

import ipaddress
import logging
import os
import socket
import urllib3
from urllib.parse import urlparse

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

TOOL_SCHEMA = {
    "name": "read",
    "description": (
        "Read and extract clean text from a URL or local file. Supports web pages, "
        "PDF, DOCX, PPTX, HTML, Markdown, and plain text. Use when user asks to read, "
        "open, fetch, or summarize a URL/webpage/article/file, or when a search returns "
        "a promising URL and the snippet is insufficient. Do NOT use for documents already "
        "in the library (use 'document' skill instead)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": "URL (e.g. 'https://example.com/article') or filesystem path (e.g. '/home/user/doc.pdf').",
            },
            "max_chars": {
                "type": "integer",
                "description": "Max characters to return (default 20000).",
            },
        },
        "required": ["source"],
    },
}

# ─── Constants ────────────────────────────────────────────────────────────────

_URL_FETCH_TIMEOUT = 15  # seconds — real pages take time; humans don't bail at 8s

# Modern browser fingerprint. Chrome 131 / macOS Sonoma — high baseline UA share,
# Sec-Ch-Ua-* + Sec-Fetch-* values match what Chromium actually emits on a top-level
# navigation. Bot-detection vendors (Cloudflare, Akamai, PerimeterX, DataDome) score
# requests on UA/Sec-Ch-Ua/Accept-Language coherence — drift between these fields
# is a strong bot signal, so keep them in sync if you change one.
_BROWSER_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/131.0.0.0 Safari/537.36'
    ),
    'Accept': (
        'text/html,application/xhtml+xml,application/xml;q=0.9,'
        'image/avif,image/webp,image/apng,*/*;q=0.8,'
        'application/signed-exchange;v=b3;q=0.7'
    ),
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Cache-Control': 'max-age=0',
    'Sec-Ch-Ua': '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
    'Sec-Ch-Ua-Mobile': '?0',
    'Sec-Ch-Ua-Platform': '"macOS"',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'none',
    'Sec-Fetch-User': '?1',
    'Upgrade-Insecure-Requests': '1',
}

# Private/internal IP ranges to block (SSRF guard)
_BLOCKED_NETS = [
    ipaddress.ip_network('127.0.0.0/8'),
    ipaddress.ip_network('10.0.0.0/8'),
    ipaddress.ip_network('172.16.0.0/12'),
    ipaddress.ip_network('192.168.0.0/16'),
    ipaddress.ip_network('169.254.0.0/16'),   # link-local / AWS metadata
    ipaddress.ip_network('::1/128'),           # IPv6 loopback
    ipaddress.ip_network('fc00::/7'),          # IPv6 unique-local
]

# System directories never contain user documents
_BLOCKED_PATH_PREFIXES = ('/etc', '/proc', '/dev', '/sys', '/var/run')


# ─── Public handler ───────────────────────────────────────────────────────────

def _tag(source: str, max_chars: int, **extra) -> str:
    """Build the canonical [read(...)] tag — mirrors [memory(radius=X)] form."""
    parts = [f"source={source}", f"max_chars={max_chars}"]
    parts.extend(f"{k}={v}" for k, v in extra.items())
    return f"[read({', '.join(parts)})]"


def handle_read(channel: str, params: dict) -> str:
    """
    Read and extract clean text from a URL or local file.

    Args:
        channel: Current conversation channel (unused but required by skill contract)
        params: {
            source (str, required): URL or filesystem path
            max_chars (int, opt):   Default 20000
        }

    Returns:
        `[read source=..., max_chars=...]\\n<content>` on success,
        `[read error=...]` on failure.
    """
    source = params.get('source', '').strip()
    if not source:
        return "[read(error=source-required)]"

    max_chars = params.get('max_chars', 20000)
    try:
        max_chars = max(100, int(max_chars))
    except (TypeError, ValueError):
        max_chars = 20000

    source_type = _classify_source(source)

    try:
        if source_type == 'url':
            return _read_url(source, max_chars)
        else:
            return _read_file(source, max_chars)
    except Exception as e:
        logger.error(f'[READ SKILL] Unexpected error for source={source!r}: {e}', exc_info=True)
        return f"[read(source={source}, error={str(e)[:200]})]"


# ─── Source classification ────────────────────────────────────────────────────

def _classify_source(source: str) -> str:
    """
    Classify source as 'url' or 'file'.

    Uses urlparse scheme check — simple and unambiguous.
    Anything with an http/https/ftp scheme is a URL; everything else is a file path.
    """
    parsed = urlparse(source)
    if parsed.scheme in ('http', 'https', 'ftp'):
        return 'url'
    return 'file'


# ─── URL reading ──────────────────────────────────────────────────────────────

def _is_private_url(url: str) -> bool:
    """
    Return True if the URL resolves to a private/internal IP address.

    Resolves the hostname before checking — prevents DNS rebinding bypasses.
    Unresolvable hostnames are treated as private (blocked).
    """
    hostname = urlparse(url).hostname
    if not hostname:
        return True
    try:
        for info in socket.getaddrinfo(hostname, None):
            addr = ipaddress.ip_address(info[4][0])
            if any(addr in net for net in _BLOCKED_NETS):
                return True
    except socket.gaierror:
        return True   # unresolvable → block
    return False


def _read_url(url: str, max_chars: int) -> str:
    """Fetch a URL and return extracted text.

    Uses a per-call requests.Session so cookies set by the origin (anti-bot
    challenge cookies, CDN session tokens, consent banners) persist across the
    redirect chain. Without a Session, each redirect hop drops Set-Cookie and
    many sites then bounce the next hop back to a challenge page.
    """
    import requests

    if _is_private_url(url):
        return f"[read(source={url}, error=private-or-internal-url-blocked)]"

    try:
        with requests.Session() as session:
            session.headers.update(_BROWSER_HEADERS)
            response = session.get(
                url,
                timeout=_URL_FETCH_TIMEOUT,
                allow_redirects=True,
                verify=False,
            )
            response.raise_for_status()
            html = response.text
    except requests.RequestException as e:
        return f"[read(source={url}, error=fetch-failed: {str(e)[:150]})]"

    from services.text_extractor import extract_html
    content = extract_html(html, url=url)

    if not content:
        return f"[read(source={url}, error=no-readable-content)]"

    truncated = len(content) > max_chars
    if truncated:
        content = content[:max_chars]

    extras = {'truncated': 'true'} if truncated else {}
    return f"{_tag(url, max_chars, **extras)} {content}"


# ─── File reading ─────────────────────────────────────────────────────────────

def _read_file(file_path: str, max_chars: int) -> str:
    """Read a local file and return extracted text."""
    expanded = os.path.expanduser(file_path)
    resolved = os.path.realpath(expanded)

    # Security: block system paths — check both the raw and resolved path so that
    # symlinks like /etc → /private/etc (macOS) don't bypass the check.
    def _is_blocked(path: str) -> bool:
        return any(path == p or path.startswith(p + '/') for p in _BLOCKED_PATH_PREFIXES)

    if _is_blocked(expanded) or _is_blocked(resolved):
        return f"[read(source={file_path}, error=system-path-blocked)]"

    if not os.path.exists(resolved):
        return f"[read(source={file_path}, error=file-not-found)]"

    if not os.path.isfile(resolved):
        return f"[read(source={file_path}, error=not-a-file)]"

    if not os.access(resolved, os.R_OK):
        return f"[read(source={file_path}, error=no-read-permission)]"

    from services.text_extractor import detect_mime_type, extract_text, normalize_text

    mime_type = detect_mime_type(resolved)
    content = extract_text(resolved, mime_type)

    if not content or not content.strip():
        return f"[read(source={file_path}, error=no-text-content)]"

    content = normalize_text(content)

    truncated = len(content) > max_chars
    if truncated:
        content = content[:max_chars]

    extras = {'mime': mime_type}
    if truncated:
        extras['truncated'] = 'true'
    return f"{_tag(file_path, max_chars, **extras)} {content}"
