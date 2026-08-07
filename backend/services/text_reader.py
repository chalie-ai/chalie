"""TextReader — the single place a URL or filesystem path becomes plain text.

Shared by the file-ingest pipeline, the ``read`` tool (file branch only — the
ability rejects URLs before constructing a reader) and the docs pipeline (URL
branch). Returns the FULL text verbatim: no truncation (the caller gates
oversized content behind its own line-window contract) and no whitespace
munging. Every failure is a raise, never an empty string.
"""

from __future__ import annotations

import os
from typing import ClassVar
from urllib.parse import urlparse

from exceptions import (
    NoReadableContent,
    NoTextContent,
    NotAFile,
    SourceIsImage,
    SystemPathBlocked,
)
from services.web_fetch import BROWSER, WebFetch


class TextReader:
    _URL_FETCH_TIMEOUT: ClassVar[int] = 15

    #: Filesystem prefixes a read never descends into — the FILE branch only;
    #: network destinations are never refused.
    _BLOCKED_PATH_PREFIXES: ClassVar[tuple[str, ...]] = ("/etc", "/proc", "/dev", "/sys", "/var/run")

    #: Path suffixes whose bytes are plain text, not markup. A URL/file ending in
    #: one of these skips HTML extraction so raw diffs/patches/code come back
    #: verbatim instead of being stripped to ``no-readable-content``.
    _PLAINTEXT_EXTENSIONS: ClassVar[tuple[str, ...]] = (
        ".diff", ".patch", ".txt", ".md", ".rst", ".csv", ".tsv", ".log",
        ".json", ".yaml", ".yml", ".toml", ".ini", ".py", ".js", ".ts", ".sh",
    )

    def __init__(self, file_path_or_url: str) -> None:
        self._source: str = file_path_or_url.strip()

    def get_value(self) -> str:
        if TextReader.is_url(self._source):
            return self._read_url()
        return self._read_file()

    # ── Classification ─────────────────────────────────────────────────────────

    @staticmethod
    def is_url(source: str) -> bool:
        return urlparse(source).scheme in ("http", "https", "ftp")

    @classmethod
    def _is_plaintext_path(cls, path: str) -> bool:
        return path.lower().endswith(cls._PLAINTEXT_EXTENSIONS)

    # ── URL branch ───────────────────────────────────────────────────────────────

    def _read_url(self) -> str:
        html, content_type = WebFetch.fetch_page(
            self._source, profile=BROWSER, timeout=self._URL_FETCH_TIMEOUT
        )

        from services.text_extractor import TextExtractor

        # Plain-text responses (text/* but not text/html) and known plain-text URL
        # paths pass through verbatim — extraction would strip a raw patch/diff to
        # nothing. Everything else goes through HTML extraction as before.
        path = urlparse(self._source).path
        is_plaintext = (
            (content_type.startswith("text/") and content_type != "text/html")
            or self._is_plaintext_path(path)
        )

        content = html if is_plaintext else TextExtractor.extract_html(html, url=self._source)

        if not content or not content.strip():
            raise NoReadableContent("The page was fetched but no readable text could be extracted.")

        return content

    # ── File branch ────────────────────────────────────────────────────────────

    def _read_file(self) -> str:
        expanded = os.path.expanduser(self._source)
        resolved = os.path.realpath(expanded)

        if self._is_blocked_path(expanded) or self._is_blocked_path(resolved):
            raise SystemPathBlocked("Reading this system path is not permitted.")

        if not os.path.exists(resolved):
            raise FileNotFoundError("No file exists at that path.")

        if not os.path.isfile(resolved):
            raise NotAFile("That path is not a file.")

        if not os.access(resolved, os.R_OK):
            raise PermissionError("That file exists but cannot be read (no permission).")

        from services.text_extractor import TextExtractor

        mime_type = TextExtractor.detect_mime_type(resolved)
        from abilities.read import ReadAbility  # noqa: PLC0415

        # read is a TEXT reader. An image has no text to read — it is described by
        # the vision tool. Rejected here with a route rather than left to
        # TextExtractor.extract_text's raise, so the LLM gets a next step instead of a stack trace.
        # The raise is mapped by the caller (the read ability) to code=not-text.
        if mime_type.startswith("image/"):
            raise SourceIsImage(f"That file is an image, not a text file — {ReadAbility.NAME} cannot see images.")
        content = TextExtractor.extract_text(resolved, mime_type)

        if not content or not content.strip():
            raise NoTextContent("No text content could be extracted from that file.")

        return content

    # ── Shared guard ───────────────────────────────────────────────────────────

    @classmethod
    def _is_blocked_path(cls, path: str) -> bool:
        return any(
            path == p or path.startswith(p + "/")
            for p in cls._BLOCKED_PATH_PREFIXES
        )
