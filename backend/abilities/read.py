"""
ReadAbility — Unified text reader for URLs and local files.

Fetches and extracts clean text from any source:
  - Web pages and articles (URL)
  - PDFs, DOCX, PPTX, HTML files, Markdown, plain text (filesystem path)

Security:
  - SSRF guard: blocks requests to private/internal IP ranges (resolved, not string-matched)
  - File guard: blocks reads from system paths (/etc, /proc, /dev, /sys, /var/run)
"""

import ipaddress
import logging
import os
import socket
import urllib3
from typing import ClassVar
from urllib.parse import urlparse

import requests

from abilities._base import Ability
from services.innate_skills._tag import tag as _skill_tag

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


class ReadAbility(Ability):
    NAME = "read"
    SUMMARY = "Fetch and extract clean text from any URL or local file — web pages, PDFs, DOCX, PPTX, and plain text."
    EXAMPLES = [
        "can you read this page and tell me what it says? https://example.com",
        "summarise the article at this link",
        "fetch the content of https://bbc.com/news/science",
        "read my PDF at /home/user/report.pdf",
        "what does that URL say",
        "open this link and give me a summary",
        "read the documentation page at https://docs.python.org/3/library/json.html",
        "extract the text from this document",
    ]
    INPUT_SCHEMA = {
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
    }
    ALWAYS_AVAILABLE = True
    TIMEOUT = 10

    _URL_FETCH_TIMEOUT: ClassVar[int] = 15

    _BROWSER_HEADERS: ClassVar[dict] = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,image/apng,*/*;q=0.8,"
            "application/signed-exchange;v=b3;q=0.7"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Cache-Control": "max-age=0",
        "Sec-Ch-Ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"macOS"',
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
    }

    _BLOCKED_NETS: ClassVar[list] = [
        ipaddress.ip_network("127.0.0.0/8"),
        ipaddress.ip_network("10.0.0.0/8"),
        ipaddress.ip_network("172.16.0.0/12"),
        ipaddress.ip_network("192.168.0.0/16"),
        ipaddress.ip_network("169.254.0.0/16"),
        ipaddress.ip_network("::1/128"),
        ipaddress.ip_network("fc00::/7"),
    ]

    _BLOCKED_PATH_PREFIXES: ClassVar[tuple] = ("/etc", "/proc", "/dev", "/sys", "/var/run")

    def execute(self, channel: str, params: dict, telemetry: dict | None) -> dict:
        source = params.get("source", "").strip()
        if not source:
            return {"text": _skill_tag("read", error="source-required")}

        max_chars = params.get("max_chars", 20000)
        try:
            max_chars = max(100, int(max_chars))
        except (TypeError, ValueError):
            max_chars = 20000

        source_type = _classify_source(source)

        try:
            if source_type == "url":
                return {"text": _read_url(source, max_chars)}
            else:
                return {"text": _read_file(source, max_chars)}
        except Exception as e:
            logger.error(f"[READ SKILL] Unexpected error for source={source!r}: {e}", exc_info=True)
            return {"text": _skill_tag("read", source=source, error=str(e)[:200])}


def _classify_source(source: str) -> str:
    parsed = urlparse(source)
    if parsed.scheme in ("http", "https", "ftp"):
        return "url"
    return "file"


def _is_private_url(url: str) -> bool:
    hostname = urlparse(url).hostname
    if not hostname:
        return True
    try:
        for info in socket.getaddrinfo(hostname, None):
            addr = ipaddress.ip_address(info[4][0])
            if any(addr in net for net in ReadAbility._BLOCKED_NETS):
                return True
    except socket.gaierror:
        return True
    return False


def _read_url(url: str, max_chars: int) -> str:
    if _is_private_url(url):
        return _skill_tag("read", source=url, error="private-or-internal-url-blocked")

    try:
        with requests.Session() as session:
            session.headers.update(ReadAbility._BROWSER_HEADERS)
            response = session.get(
                url,
                timeout=ReadAbility._URL_FETCH_TIMEOUT,
                allow_redirects=True,
                verify=False,
            )
            response.raise_for_status()
            html = response.text
    except requests.RequestException as e:
        return _skill_tag("read", source=url, error=f"fetch-failed:{str(e)[:150]}")

    from services.text_extractor import extract_html
    content = extract_html(html, url=url)

    if not content:
        return _skill_tag("read", source=url, error="no-readable-content")

    truncated = len(content) > max_chars
    if truncated:
        content = content[:max_chars]

    kwargs: dict = {"source": url, "max_chars": max_chars}
    if truncated:
        kwargs["truncated"] = "true"
    return _skill_tag("read", content, **kwargs)


def _read_file(file_path: str, max_chars: int) -> str:
    expanded = os.path.expanduser(file_path)
    resolved = os.path.realpath(expanded)

    def _is_blocked(path: str) -> bool:
        return any(
            path == p or path.startswith(p + "/")
            for p in ReadAbility._BLOCKED_PATH_PREFIXES
        )

    if _is_blocked(expanded) or _is_blocked(resolved):
        return _skill_tag("read", source=file_path, error="system-path-blocked")

    if not os.path.exists(resolved):
        return _skill_tag("read", source=file_path, error="file-not-found")

    if not os.path.isfile(resolved):
        return _skill_tag("read", source=file_path, error="not-a-file")

    if not os.access(resolved, os.R_OK):
        return _skill_tag("read", source=file_path, error="no-read-permission")

    from services.text_extractor import detect_mime_type, extract_text, normalize_text

    mime_type = detect_mime_type(resolved)
    content = extract_text(resolved, mime_type)

    if not content or not content.strip():
        return _skill_tag("read", source=file_path, error="no-text-content")

    content = normalize_text(content)

    truncated = len(content) > max_chars
    if truncated:
        content = content[:max_chars]

    kwargs: dict = {"source": file_path, "max_chars": max_chars, "mime": mime_type}
    if truncated:
        kwargs["truncated"] = "true"
    return _skill_tag("read", content, **kwargs)
