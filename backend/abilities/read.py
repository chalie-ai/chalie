"""
ReadAbility — Unified text reader for URLs and local files.

Fetches and extracts clean text from any source:
  - Web pages and articles (URL)
  - PDFs, DOCX, PPTX, HTML files, Markdown, plain text (filesystem path)

Security:
  - SSRF guard: blocks requests to private/internal IP ranges (resolved, not string-matched)
  - File guard: blocks reads from system paths (/etc, /proc, /dev, /sys, /var/run)
"""

import logging
import os
import urllib3
from typing import ClassVar
from urllib.parse import urlparse

import requests

from abilities._ability import Ability
from abilities._result import ToolResult
from services.ssrf import is_private_url
from services.web_fetch import BROWSER, fetch_text

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


class ReadAbility(Ability):
    def get_name(self) -> str:
        return "read"

    def get_summary(self) -> str:
        return "Fetch and extract clean text from any URL or local file — web pages, PDFs, DOCX, PPTX, and plain text."

    def get_examples(self) -> list[str]:
        return [
            "can you read this page and tell me what it says? https://example.com",
            "summarise the article at this link",
            "fetch the content of https://bbc.com/news/science",
            "read my PDF at /home/user/report.pdf",
            "what does that URL say",
            "open this link and give me a summary",
            "read the documentation page at https://docs.python.org/3/library/json.html",
            "extract the text from this document",
        ]

    def get_search_tooltip(self) -> str:
        return "fetch URL or file contents"

    _PARAMETERS: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": "URL (e.g. 'https://example.com/article') or filesystem path (e.g. '/home/user/doc.pdf'). Aliases 'url' and 'path' are also accepted.",
            },
            "max_chars": {
                "type": "integer",
                "description": "Max characters to return (default 20000).",
            },
        },
        "required": ["source"],
    }

    def get_parameters(self) -> dict:
        return self._PARAMETERS

    _URL_FETCH_TIMEOUT: ClassVar[int] = 15

    _BLOCKED_PATH_PREFIXES: ClassVar[tuple] = ("/etc", "/proc", "/dev", "/sys", "/var/run")

    def run(self, params: dict) -> ToolResult:
        source = _first_source(params)
        if not source:
            # No usable target under any accepted key. Echo what the model DID
            # send so it self-corrects instead of looping on the same opaque
            # error (models pass url=/path= and never learn why).
            received = "|".join(params.keys()) or "none"
            return ToolResult.err(
                "source-required",
                code="source-required",
                hint="pass the URL or file path as 'source'",
                received_keys=received,
            )

        max_chars = params.get("max_chars", 20000)
        try:
            max_chars = max(100, int(max_chars))
        except (TypeError, ValueError):
            max_chars = 20000

        source_type = _classify_source(source)

        try:
            if source_type == "url":
                return _read_url(source, max_chars)
            return _read_file(source, max_chars)
        except Exception as e:
            logger.exception(f"[READ SKILL] Unexpected error for source={source!r}: {e}")
            return ToolResult.err(str(e)[:200], code="error", source=source)


#: The canonical key plus the keys a model naturally emits for the read target.
#: When ``source`` is absent we honour these in order so a call like
#: ``read({"url": …})`` or ``read({"path": …})`` works instead of bouncing on
#: ``source-required``. ``source`` stays first so it always wins.
_SOURCE_KEYS = ("source", "url", "uri", "link", "href", "path", "file", "filepath", "file_path")


def _first_source(params: dict) -> str:
    """Return the first non-empty read target among the accepted keys, stripped.

    Returns "" when no accepted key carries a usable string value.
    """
    for key in _SOURCE_KEYS:
        value = params.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _classify_source(source: str) -> str:
    parsed = urlparse(source)
    if parsed.scheme in ("http", "https", "ftp"):
        return "url"
    return "file"


def _read_url(url: str, max_chars: int) -> ToolResult:
    if is_private_url(url):
        return ToolResult.err(
            "private-or-internal-url-blocked", code="private-or-internal-url-blocked", source=url
        )

    try:
        html = fetch_text(
            url, profile=BROWSER, timeout=ReadAbility._URL_FETCH_TIMEOUT
        )
    except requests.RequestException as e:
        return ToolResult.err(f"fetch-failed:{str(e)[:150]}", code="fetch-failed", source=url)

    from services.text_extractor import extract_html
    content = extract_html(html, url=url)

    if not content:
        return ToolResult.err("no-readable-content", code="no-readable-content", source=url)

    truncated = len(content) > max_chars
    if truncated:
        content = content[:max_chars]

    meta: dict = {"source": url, "max_chars": max_chars}
    if truncated:
        meta["truncated"] = "true"
    return ToolResult.ok(content, **meta)


def _read_file(file_path: str, max_chars: int) -> ToolResult:
    expanded = os.path.expanduser(file_path)
    resolved = os.path.realpath(expanded)

    def _is_blocked(path: str) -> bool:
        return any(
            path == p or path.startswith(p + "/")
            for p in ReadAbility._BLOCKED_PATH_PREFIXES
        )

    if _is_blocked(expanded) or _is_blocked(resolved):
        return ToolResult.err("system-path-blocked", code="system-path-blocked", source=file_path)

    if not os.path.exists(resolved):
        return ToolResult.err("file-not-found", code="file-not-found", source=file_path)

    if not os.path.isfile(resolved):
        return ToolResult.err("not-a-file", code="not-a-file", source=file_path)

    if not os.access(resolved, os.R_OK):
        return ToolResult.err("no-read-permission", code="no-read-permission", source=file_path)

    from services.text_extractor import detect_mime_type, extract_text, normalize_text

    mime_type = detect_mime_type(resolved)
    content = extract_text(resolved, mime_type)

    if not content or not content.strip():
        return ToolResult.err("no-text-content", code="no-text-content", source=file_path)

    content = normalize_text(content)

    truncated = len(content) > max_chars
    if truncated:
        content = content[:max_chars]

    meta: dict = {"source": file_path, "max_chars": max_chars, "mime": mime_type}
    if truncated:
        meta["truncated"] = "true"
    return ToolResult.ok(content, **meta)
