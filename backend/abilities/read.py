"""
ReadAbility — Unified text reader for URLs and local files.

Fetches and extracts clean text from any source:
  - Web pages and articles (URL)
  - PDFs, DOCX, PPTX, HTML files, Markdown, plain text (filesystem path)

Security:
  - SSRF guard: a SINGLE gate in ``services.web_fetch`` blocks requests to
    private/internal IP ranges (resolved, not string-matched) before any socket
    opens. The ability catches :class:`FetchBlocked` and maps it to a stable
    ``code=private-or-internal-url-blocked``.
  - File guard: blocks reads from system paths (/etc, /proc, /dev, /sys,
    /var/run).

Result contract: every return is a :class:`abilities._result.ToolResult` built
only via ``ok()`` / ``err()``; the dispatcher renders the wire envelope. Success
bodies are the extracted text (prose) with ``content_type`` + truncation in the
meta; errors carry a stable kebab-case ``code`` (never the ``code="error"``
placeholder) so a weak model can self-correct without re-reading the schema.

Passthrough: a response whose content type is ``text/*`` (but not ``text/html``),
or a URL/file whose path ends with a known plain-text extension (``.diff``,
``.patch``, ``.txt`` …), is normalised WITHOUT HTML extraction — raw patches and
diffs come back verbatim instead of being stripped to ``no-readable-content``.
"""

import logging
import os
from typing import ClassVar
from urllib.parse import urlparse

from abilities._ability import Ability
from abilities._result import ToolResult, truncate
from services.web_fetch import BROWSER, FetchBlocked, fetch_page

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

    #: The keys a model naturally emits for the read target. ``source`` is the
    #: canonical key; the rest are honoured as aliases via ``Ability.param`` so a
    #: call like ``read({"url": …})`` or ``read({"path": …})`` resolves instead of
    #: bouncing on ``source-required``.
    _SOURCE_ALIASES: ClassVar[tuple[str, ...]] = (
        "url", "uri", "link", "href", "path", "file", "filepath", "file_path",
    )

    _URL_FETCH_TIMEOUT: ClassVar[int] = 15

    _BLOCKED_PATH_PREFIXES: ClassVar[tuple] = ("/etc", "/proc", "/dev", "/sys", "/var/run")

    #: Path suffixes whose bytes are plain text, not markup. A URL/file ending in
    #: one of these is normalised WITHOUT HTML extraction so raw diffs/patches/code
    #: come back verbatim instead of being stripped to ``no-readable-content``.
    _PLAINTEXT_EXTENSIONS: ClassVar[tuple[str, ...]] = (
        ".diff", ".patch", ".txt", ".md", ".rst", ".csv", ".tsv", ".log",
        ".json", ".yaml", ".yml", ".toml", ".ini", ".py", ".js", ".ts", ".sh",
    )

    def run(self, params: dict) -> ToolResult:
        source = self.param(params, "source", aliases=self._SOURCE_ALIASES)
        if not isinstance(source, str) or not source.strip():
            # No usable target under any accepted key. Echo what the model DID send
            # so it self-corrects instead of looping on the same opaque error
            # (models pass url=/path= and never learn why). Keep code=source-required
            # so other surfaces still match on it — param(required=True) would lose
            # the received-keys diagnostic.
            received = "|".join(params.keys()) or "none"
            return ToolResult.err(
                "No URL or file path was provided.",
                code="source-required",
                hint=f"pass the URL or file path as 'source' (received: {received})",
                received_keys=received,
            )
        source = source.strip()

        max_chars = self.param(params, "max_chars", default=20000, clamp=(100, 100000))

        if self._is_url(source):
            return self._read_url(source, max_chars)
        return self._read_file(source, max_chars)

    # ── Classification ─────────────────────────────────────────────────────────

    @staticmethod
    def _is_url(source: str) -> bool:
        return urlparse(source).scheme in ("http", "https", "ftp")

    @classmethod
    def _is_plaintext_path(cls, path: str) -> bool:
        """True when *path* ends with a known plain-text extension (case-folded).

        A bare URL path is fine — the query/fragment never carry the extension we
        gate on, and ``urlparse(...).path`` is already stripped of them.
        """
        return path.lower().endswith(cls._PLAINTEXT_EXTENSIONS)

    # ── URL branch ───────────────────────────────────────────────────────────────

    def _read_url(self, url: str, max_chars: int) -> ToolResult:
        try:
            html, content_type = fetch_page(
                url, profile=BROWSER, timeout=self._URL_FETCH_TIMEOUT
            )
        except FetchBlocked:
            # The single SSRF gate in web_fetch refused this host.
            return ToolResult.err(
                "This URL resolves to a private or internal address and was blocked.",
                code="private-or-internal-url-blocked",
                source=url,
            )
        except Exception as e:  # noqa: BLE001 — surface the fetch failure as a coded result
            return ToolResult.err(
                f"Could not fetch the URL: {str(e)[:150]}",
                code="fetch-failed",
                hint="check the URL is reachable and try again",
                source=url,
            )

        from services.text_extractor import extract_html, normalize_text

        # Plain-text responses (text/* but not text/html) and known plain-text URL
        # paths pass through verbatim — extraction would strip a raw patch/diff to
        # nothing. Everything else goes through HTML extraction as before.
        path = urlparse(url).path
        is_plaintext = (
            (content_type.startswith("text/") and content_type != "text/html")
            or self._is_plaintext_path(path)
        )

        if is_plaintext:
            content = normalize_text(html)
        else:
            content = extract_html(html, url=url)

        if not content or not content.strip():
            return ToolResult.err(
                "The page was fetched but no readable text could be extracted.",
                code="no-readable-content",
                source=url,
            )

        return self._ok_text(content, max_chars, source=url, content_type=content_type or "text/html")

    # ── File branch ────────────────────────────────────────────────────────────

    def _read_file(self, file_path: str, max_chars: int) -> ToolResult:
        expanded = os.path.expanduser(file_path)
        resolved = os.path.realpath(expanded)

        if self._is_blocked_path(expanded) or self._is_blocked_path(resolved):
            return ToolResult.err(
                "Reading this system path is not permitted.",
                code="system-path-blocked",
                source=file_path,
            )

        if not os.path.exists(resolved):
            return ToolResult.err(
                "No file exists at that path.",
                code="file-not-found",
                hint="check the path is correct",
                source=file_path,
            )

        if not os.path.isfile(resolved):
            return ToolResult.err(
                "That path is not a file.",
                code="not-a-file",
                source=file_path,
            )

        if not os.access(resolved, os.R_OK):
            return ToolResult.err(
                "That file exists but cannot be read (no permission).",
                code="no-read-permission",
                source=file_path,
            )

        from services.text_extractor import detect_mime_type, extract_text, normalize_text

        mime_type = detect_mime_type(resolved)
        content = extract_text(resolved, mime_type)

        if not content or not content.strip():
            return ToolResult.err(
                "No text content could be extracted from that file.",
                code="no-text-content",
                source=file_path,
            )

        content = normalize_text(content)
        return self._ok_text(content, max_chars, source=file_path, content_type=mime_type)

    # ── Shared success builder ─────────────────────────────────────────────────

    @staticmethod
    def _ok_text(content: str, max_chars: int, *, source: str, content_type: str) -> ToolResult:
        clipped, was_clipped = truncate(content, max_chars)
        meta: dict = {"source": source, "content_type": content_type}
        if was_clipped:
            meta["truncated"] = True
        return ToolResult.ok(clipped, **meta)

    @classmethod
    def _is_blocked_path(cls, path: str) -> bool:
        return any(
            path == p or path.startswith(p + "/")
            for p in cls._BLOCKED_PATH_PREFIXES
        )
