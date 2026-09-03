"""Url-to-markdown conversion backed by trafilatura.

Performs no HTTP itself: takes already-fetched HTML, extracts markdown, and
appends an ``## Embedded media`` section listing iframe/video/source URLs
(resolved absolute; omitted when the page has none). Persists under the
web-pages directory.
"""

import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

from services.file_mapper_service import FileMapperService


def url_slug(url: str) -> str:
    """Flat per-URL slug ``<host>--<path-slug>`` (just ``<host>`` on an empty path).

    The one naming convention for every URL-derived filename (web pages AND
    downloads): distinct URLs never collide on disk, a re-fetch maps to the
    same name and overwrites in place, and no separator or ``..`` survives
    into the filename — path runs of non-alphanumerics collapse to a single
    ``-``, capped at 120 characters.
    """
    parsed = urlparse(url.lower())
    host = parsed.hostname or ""
    slug = re.sub(r"[^a-z0-9]+", "-", (parsed.path or "").lstrip("/")).strip("-")[:120]
    return f"{host}--{slug}" if slug else host


class _MediaSrcCollector(HTMLParser):
    """Collect ``src`` attributes from iframe/video/source elements (self-closing
    tags included — HTMLParser's ``handle_startendtag`` delegates here)."""

    _MEDIA_TAGS = frozenset({"iframe", "video", "source"})

    def __init__(self) -> None:
        super().__init__()
        self.media_srcs: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in self._MEDIA_TAGS:
            for name, val in attrs:
                if name == "src" and val and val.strip():
                    self.media_srcs.add(val.strip())


def convert_to_markdown(html: str | bytes, url: str) -> str:
    """Extract markdown and append the embedded-media section.

    Raw ``bytes`` go to trafilatura verbatim so it sniffs the declared
    charset; the media scan decodes them UTF-8-with-replacement, lossless
    for the ASCII ``src`` URLs it looks for. *url* resolves relative links.

    Raises:
        ValueError: trafilatura found nothing extractable.
    """
    import trafilatura

    markdown = trafilatura.extract(
        html,
        url=url,
        output_format="markdown",
        include_images=True,
        include_links=True,
        include_tables=True,
        favor_recall=True,
    )

    if not markdown or not markdown.strip():
        raise ValueError(
            f"trafilatura produced no content for URL '{url}'; "
            "the supplied HTML is empty or contains no extractable body."
        )

    collector = _MediaSrcCollector()
    collector.feed(html if isinstance(html, str) else html.decode("utf-8", "replace"))

    media_srcs = sorted(collector.media_srcs)
    if not media_srcs:
        return markdown

    links_block = "\n".join(
        f"[{urljoin(url, src)}]({urljoin(url, src)})" for src in media_srcs
    )
    section = f"\n## Embedded media\n\n{links_block}\n"
    return markdown + section


def filename_for(url: str, extension: str) -> str:
    """Flat filename ``<host>--<path-slug>.<extension>`` (rules: :func:`url_slug`)."""
    return f"{url_slug(url)}.{extension}"


def persist(content: str, url: str, extension: str) -> Path:
    """Write *content* under the web-pages directory for *url*, overwriting
    any previous fetch, and return the absolute path."""
    fname = filename_for(url, extension)
    target = FileMapperService.get_web_pages_path(fname)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target.absolute()
