"""WebFetchAbility — fetch a URL and persist it under the data directory.

The single URL-owning tool: HTML pages are converted to markdown (default) or
kept as verbatim HTML under the web-pages directory; every other content type
streams to the downloads directory. Re-fetching the same URL overwrites in
place — the on-disk copy is always the latest fetch, never a cache to consult.

Security:
  - Scheme check: ``file:`` / ``data:`` URLs are refused inline (network-free)
    with ``code=blocked-url``, before any socket opens.
  - SSRF guard: the single gate in ``services.web_fetch`` blocks
    private/internal hosts; :class:`FetchBlocked` maps to ``code=blocked-url``.
  - Size cap: ``stream_to_file`` enforces the 100 MB cap during the stream;
    :class:`DownloadTooLarge` maps to ``code=too-large``.

The 20000-character gate: when the persisted page text fits, it IS the body
(with ``url``/``path`` meta); when it does not, the body is a one-line pointer
at the saved path and the model chunks it back in via ``read``'s line windows.
Non-HTML responses never return file bytes — the body is a per-MIME
instruction telling the model which tool reads the saved path.

Result contract: every return is a :class:`abilities._result.ToolResult` built
via ``ok()`` / ``err()``; the dispatcher renders the wire envelope. Errors
carry a stable kebab-case ``code`` (``blocked-url`` / ``too-large`` /
``download-failed`` / ``no-readable-content``) so a weak model can
self-correct without re-reading the schema.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar
from urllib.parse import urlparse

import requests

from abilities._ability import Ability
from abilities._result import ToolResult
from configs.enums.ability_category import AbilityCategory
from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag
from contracts.params.web_fetch_params_bag import WebFetchParamsBag
from exceptions import DownloadTooLarge, FetchBlocked
from services.file_mapper_service import FileMapperService
from services.url_to_markdown_service import UrlToMarkdownService
from services.web_fetch import BROWSER, stream_to_file

_BLOCKED_SCHEMES = frozenset({"file", "data"})


class WebFetchAbility(Ability[WebFetchParamsBag]):
    #: The fetch size cap (100 MB). Enforced by ``stream_to_file`` during the
    #: stream; an over-cap pull is a ``too-large`` error with the partial file
    #: removed.
    _MAX_FETCH_BYTES: ClassVar[int] = 100 * 1024 * 1024

    #: Persisted page text at or under this many characters is returned as the
    #: body; anything over it returns only the pointer at the saved path — the
    #: same gate ``read`` applies, so the two tools never disagree about what
    #: fits in context.
    _RETURN_CHAR_LIMIT: ClassVar[int] = 20_000

    #: Action-less tool — the ``""`` key drives the dispatcher's ACTION_REQUIRED
    #: pre-gate to reject a missing/blank url with ``code=missing-params``
    #: BEFORE run().
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"": (Keys.url,)}

    # The typed input contract: the dispatch seam builds the bag via
    # WebFetchParamsBag.from_params before run() is called.
    PARAMS: ClassVar[type[ParamBag] | None] = WebFetchParamsBag
    SEARCHABLE_AS: ClassVar[tuple[str, ...]] = (
        "fetch web page",
        "url to markdown",
        "save web page",
        "fetch url",
        "open url",
        "read page",
        "fetch",
        "download",
        "download file",
        "download url",
    )
    NAME: ClassVar[str] = "web_fetch"
    CATEGORY: ClassVar[AbilityCategory] = AbilityCategory.WEB

    def get_summary(self) -> str:
        from abilities.read import ReadAbility  # noqa: PLC0415

        return (
            "Fetch a URL and persist it to disk. Web pages are converted to "
            "markdown and saved under the web-pages directory (pass "
            "convert_to_markdown as false to keep the raw HTML); any other "
            "file type is saved to the downloads directory. Small pages come "
            f"back directly; large ones return the saved path for the "
            f"`{ReadAbility.NAME}` tool to load in chunks."
        )

    def get_examples(self) -> list[str]:
        return [
            "fetch that article at https://example.com/post and convert to markdown",
            "save the documentation page to my web-pages folder",
            "retrieve https://wiki.example.org/some-page as markdown",
            "get the content of this URL into context",
        ]

    def get_search_tooltip(self) -> str:
        return "Fetch and persist web pages"

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.url: {
                "type": "string",
                "description": "URL of the page or file to fetch.",
            },
            Keys.convert_to_markdown: {
                "type": "boolean",
                "default": True,
                "description": (
                    "Convert HTML pages to markdown before persisting "
                    "(default: true). Pass false to keep the HTML verbatim."
                ),
            },
            Keys.timeout: {
                "type": "integer",
                "description": "Request timeout in minutes (default: 15, max: 120).",
            },
        },
        "required": [Keys.url],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: WebFetchParamsBag) -> ToolResult:
        url = params.url

        scheme = urlparse(url).scheme
        if scheme in _BLOCKED_SCHEMES:
            return ToolResult.err(
                f"The '{scheme}:' URL scheme cannot be fetched.",
                code="blocked-url",
                hint="provide an http(s) URL",
                source=url,
            )

        dest = FileMapperService.get_downloads_path(_filename_from_url(url))
        try:
            bytes_written, content_type = stream_to_file(
                url,
                str(dest),
                profile=BROWSER,
                timeout=params.timeout * 60.0,
                max_bytes=self._MAX_FETCH_BYTES,
            )
        except FetchBlocked:
            # The single SSRF gate in services.web_fetch refused this host.
            return ToolResult.err(
                "This URL resolves to a private or internal address and was blocked.",
                code="blocked-url",
                source=url,
            )
        except DownloadTooLarge as e:
            return ToolResult.err(
                f"The resource exceeds the {self._MAX_FETCH_BYTES // (1024 * 1024)} MB "
                "fetch cap and was not saved.",
                code="too-large",
                hint="fetch a smaller resource",
                max_bytes=e.max_bytes,
                source=url,
            )
        except requests.RequestException as e:
            return ToolResult.err(
                f"Could not fetch the URL: {str(e)[:150]}",
                code="download-failed",
                hint="check the URL is reachable and try again",
                source=url,
            )

        if content_type and ("text/html" in content_type or "application/xhtml" in content_type):
            return self._persist_page(url, dest, convert=params.convert_to_markdown)
        return self._describe_download(url, dest, content_type, bytes_written)

    # ── HTML → web-pages directory --------------------------------------

    def _persist_page(self, url: str, dest: Path, *, convert: bool) -> ToolResult:
        """Move the fetched HTML out of downloads into the web-pages directory,
        as markdown or verbatim, and return content or a pointer per the 20k gate."""
        try:
            raw_bytes = dest.read_bytes()
        except OSError as e:
            return ToolResult.err(
                f"Fetched the page but could not read it back from disk: {e}",
                code="download-failed",
                source=url,
            )
        finally:
            # The downloads copy is a fetch buffer for pages, never the home —
            # remove it on every exit so no temp file outlives the call.
            dest.unlink(missing_ok=True)

        converter = UrlToMarkdownService()
        if convert:
            try:
                content = converter.convert(raw_bytes, url)
            except ValueError as e:
                return ToolResult.err(
                    str(e),
                    code="no-readable-content",
                    hint="fetch with convert_to_markdown as false to keep the raw HTML instead",
                    source=url,
                )
            page_path = converter.persist(content, url, "md")
        else:
            page_path = FileMapperService.get_web_pages_path(converter.filename_for(url, "html"))
            page_path.parent.mkdir(parents=True, exist_ok=True)
            page_path.write_bytes(raw_bytes)
            page_path = page_path.absolute()
            content = raw_bytes.decode("utf-8", errors="replace")

        if len(content) <= self._RETURN_CHAR_LIMIT:
            return ToolResult.ok(content, url=url, path=str(page_path))

        from abilities.read import ReadAbility  # noqa: PLC0415

        return ToolResult.ok(
            f"URL fetched and persisted at; `{page_path}`. "
            f"Use the `{ReadAbility.NAME}` tool to load it in context in chunks",
            url=url,
            path=str(page_path),
        )

    # ── non-HTML → downloads directory ----------------------------------

    def _describe_download(
        self, url: str, dest: Path, content_type: str, bytes_written: int
    ) -> ToolResult:
        """The file stays in downloads; the body tells the model which tool
        reads the saved path for this content type."""
        path_str = str(dest)
        mime = content_type or ""

        if mime.startswith("image/"):
            from abilities.vision import VisionAbility  # noqa: PLC0415

            body = (
                f"Image fetched and saved to; `{path_str}`. "
                f"Use the `{VisionAbility.NAME}` tool on this path to see it."
            )
        elif (
            "pdf" in mime
            or "wordprocessingml" in mime
            or "presentationml" in mime
            or mime.startswith("text/")
        ):
            from abilities.read import ReadAbility  # noqa: PLC0415

            body = (
                f"File fetched and saved to; `{path_str}`. "
                f"Use the `{ReadAbility.NAME}` tool to load it in context."
            )
        else:
            body = f"File fetched and saved to; `{path_str}`."

        return ToolResult.ok(
            body, url=url, path=path_str, content_type=mime, bytes=bytes_written
        )


def _filename_from_url(url: str) -> str:
    """Basename of the URL path, falling back to the host — flat in downloads,
    so a re-fetch of the same URL overwrites in place."""
    parsed = urlparse(url)
    return os.path.basename(parsed.path.rstrip("/")) or parsed.hostname or "download"
