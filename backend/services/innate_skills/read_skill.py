"""
Read Skill — Unified text reader for URLs and local files.

Fetches and extracts clean text from any source:
  - Web pages and articles (URL)
  - PDFs, DOCX, PPTX, HTML files, Markdown, plain text (filesystem path)

Text extraction delegates to services.text_extractor (shared with DocumentProcessingService).
URL link extraction is inlined here — it is URL-specific and not useful for file reads.

Security:
  - SSRF guard: blocks requests to private/internal IP ranges (resolved, not string-matched)
  - File guard: blocks reads from system paths (/etc, /proc, /dev, /sys, /var/run)
"""

import ipaddress
import logging
import os
import re
import socket
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

_URL_FETCH_TIMEOUT = 8  # seconds

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

# Link extraction filters
_SKIP_DOMAINS = frozenset((
    'facebook.com', 'twitter.com', 'x.com', 'instagram.com', 'linkedin.com',
    'pinterest.com', 'tiktok.com', 'youtube.com', 'reddit.com',
))
_SKIP_PATH_RE = re.compile(
    r'/(login|signin|signup|register|logout|privacy|terms|cookie|legal|contact'
    r'|about-us|careers|advertise|press|help/?)(\b|$)',
    re.IGNORECASE,
)
_ANCHOR_RE = re.compile(
    r'<a\s[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
    re.DOTALL | re.IGNORECASE,
)
_TAG_RE = re.compile(r'<[^>]+>')
_MAX_LINKS = 15


# ─── Public handler ───────────────────────────────────────────────────────────

def handle_read(topic: str, params: dict) -> str:
    """
    Read and extract clean text from a URL or local file.

    Args:
        topic: Current conversation topic (unused but required by skill contract)
        params: {
            source (str, required): URL or filesystem path
            url    (str, alias):    Accepted as alias for 'source' (backward compat)
            max_chars (int, opt):   Default 4000, max 8000
        }

    Returns:
        Formatted string with [READ] prefix, or [READ] Error on failure.
    """
    source = (params.get('source') or params.get('url') or '').strip()
    if not source:
        return "[READ] Error: 'source' parameter is required (URL or file path)."

    max_chars = params.get('max_chars', 4000)
    try:
        max_chars = max(100, min(8000, int(max_chars)))
    except (TypeError, ValueError):
        max_chars = 4000

    source_type = _classify_source(source)

    try:
        if source_type == 'url':
            return _read_url(source, max_chars)
        else:
            return _read_file(source, max_chars)
    except Exception as e:
        logger.error(f'[READ SKILL] Unexpected error for source={source!r}: {e}', exc_info=True)
        return f"[READ] Error reading '{source}': {str(e)[:200]}"


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
    """Fetch a URL and return extracted text with navigable links."""
    import requests

    if _is_private_url(url):
        return f"[READ] Error: access denied — private/internal URL '{url}'."

    try:
        response = requests.get(
            url,
            timeout=_URL_FETCH_TIMEOUT,
            headers={'User-Agent': 'Mozilla/5.0 (compatible; Chalie/1.0; cognitive-agent)'},
            allow_redirects=True,
        )
        response.raise_for_status()
        html = response.text
    except requests.RequestException as e:
        return f"[READ] Fetch failed for {url}: {str(e)[:150]}"

    from services.text_extractor import extract_html
    content = extract_html(html, url=url)
    links = _extract_links(html, url)

    if not content:
        links_str = _format_links(links)
        return f"[READ] No readable content extracted from {url}.{links_str}"

    truncated = len(content) > max_chars
    if truncated:
        content = content[:max_chars]

    parts = [
        f"[READ] {url} ({len(content)} chars{', truncated' if truncated else ''}):",
        content,
    ]
    links_str = _format_links(links)
    if links_str:
        parts.append(links_str)

    return '\n'.join(parts)


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
        return f"[READ] Error: access denied — system path '{file_path}'."

    if not os.path.exists(resolved):
        return f"[READ] Error: file not found: '{file_path}'."

    if not os.path.isfile(resolved):
        return f"[READ] Error: '{file_path}' is not a file (directory or special file)."

    if not os.access(resolved, os.R_OK):
        return f"[READ] Error: no read permission for '{file_path}'."

    from services.text_extractor import detect_mime_type, extract_text, normalize_text

    mime_type = detect_mime_type(resolved)
    content = extract_text(resolved, mime_type)

    if not content or not content.strip():
        return f"[READ] No text content found in '{os.path.basename(file_path)}'."

    content = normalize_text(content)

    truncated = len(content) > max_chars
    if truncated:
        content = content[:max_chars]

    filename = os.path.basename(file_path)
    return (
        f"[READ] {filename} ({mime_type}, {len(content)} chars"
        f"{', truncated' if truncated else ''}):\n{content}"
    )


# ─── Link extraction (URL-only) ───────────────────────────────────────────────

def _extract_links(html: str, base_url: str) -> list:
    """
    Extract navigable page links from raw HTML.

    Returns up to 15 deduplicated links as [{"text": str, "url": str}, ...].
    Filters social media domains, common non-content paths, fragment-only anchors,
    and non-http(s) schemes.
    """
    try:
        links = []
        seen = set()

        for href, raw_text in _ANCHOR_RE.findall(html):
            href = href.strip()
            if not href or href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                continue

            absolute = urljoin(base_url, href)
            parsed = urlparse(absolute)

            if parsed.scheme not in ('http', 'https'):
                continue

            clean_url = parsed._replace(fragment='').geturl()
            if clean_url in seen or clean_url.rstrip('/') == base_url.rstrip('/'):
                continue

            domain = parsed.netloc.lower()
            if any(domain == d or domain.endswith('.' + d) for d in _SKIP_DOMAINS):
                continue

            if _SKIP_PATH_RE.search(parsed.path):
                continue

            seen.add(clean_url)
            text = _TAG_RE.sub('', raw_text).strip()
            text = re.sub(r'\s+', ' ', text)
            if not text or len(text) > 120:
                text = text[:120].strip() if text else parsed.path.rstrip('/').split('/')[-1]

            links.append({'text': text, 'url': clean_url})
            if len(links) >= _MAX_LINKS:
                break

        return links
    except Exception as e:
        logger.debug(f'[READ SKILL] Link extraction failed: {e}')
        return []


def _format_links(links: list) -> str:
    """Format extracted links as a markdown list for ACT loop consumption."""
    if not links:
        return ''
    lines = ['\nPage links:']
    for link in links:
        lines.append(f"  - [{link['text']}]({link['url']})")
    return '\n'.join(lines)
