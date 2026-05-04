"""OG Image Service — extract og:image / twitter:image from article URLs.

Used as a fallback when a search/news provider doesn't surface a native
image URL (Google News RSS strips them, for example). Fetches the article
HTML in parallel, scans the head for an `og:image` or `twitter:image` meta
tag, and returns the first valid absolute URL.

Public API:
    resolve_og_images(urls, max_workers=3, timeout=3.0) -> dict[str, str]
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable
from urllib.parse import urljoin

import requests

logger = logging.getLogger(__name__)

_FETCH_TIMEOUT = 3.0
_MAX_BYTES = 256 * 1024  # head usually well under this; cap to avoid huge pages
_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# Google's consent gate redirects EU/UK requests to consent.google.com unless
# SOCS is set. Without this the og:image lives on the consent page (not the
# article), so Google News thumbnails never come through.
_DEFAULT_COOKIES = {
    "SOCS": "CAESHAgBEhJnd3NfMjAyNDA1MjItMF9SQzIaAmVuIAEaBgiAxqWyBg",
}

# Match property/name in either order, single or double quotes.
_META_RE = re.compile(
    r"""<meta\s+[^>]*?(?:property|name)\s*=\s*['"](og:image(?::secure_url)?|twitter:image)['"][^>]*?>""",
    re.IGNORECASE | re.DOTALL,
)
_CONTENT_RE = re.compile(
    r"""content\s*=\s*['"]([^'"]+)['"]""",
    re.IGNORECASE,
)


def resolve_og_images(
    urls: Iterable[str], max_workers: int = 3, timeout: float = _FETCH_TIMEOUT
) -> dict[str, str]:
    """Fetch article HTML in parallel and extract og:image for each URL.

    Returns ``{article_url: image_url}`` mapping for URLs that yielded a hit.
    Failures and missing tags are simply omitted — callers should not assume
    every input URL is in the output.
    """
    targets = [u for u in urls if isinstance(u, str) and u.startswith(("http://", "https://"))]
    if not targets:
        return {}

    out: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=min(max_workers, len(targets))) as pool:
        futures = {pool.submit(_extract_one, u, timeout): u for u in targets}
        for future in as_completed(futures):
            url = futures[future]
            try:
                img = future.result()
            except Exception as exc:
                logger.debug("og_image: %s failed: %s", url, exc)
                continue
            if img:
                out[url] = img
    return out


def _extract_one(url: str, timeout: float) -> str:
    try:
        with requests.get(
            url,
            timeout=timeout,
            stream=True,
            allow_redirects=True,
            headers={"User-Agent": _USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
            cookies=_DEFAULT_COOKIES,
            verify=False,
        ) as resp:
            resp.raise_for_status()
            ct = (resp.headers.get("Content-Type") or "").lower()
            if "html" not in ct:
                return ""
            buf = bytearray()
            for chunk in resp.iter_content(chunk_size=16 * 1024, decode_unicode=False):
                if not chunk:
                    continue
                buf.extend(chunk)
                if len(buf) >= _MAX_BYTES:
                    break
                # Early exit once </head> seen
                if b"</head>" in buf or b"</HEAD>" in buf:
                    break
            final_url = str(resp.url)
    except Exception as exc:
        logger.debug("og_image: GET %s failed: %s", url, exc)
        return ""

    text = buf.decode("utf-8", errors="ignore")
    for tag_match in _META_RE.finditer(text):
        tag = tag_match.group(0)
        content_match = _CONTENT_RE.search(tag)
        if not content_match:
            continue
        candidate = content_match.group(1).strip()
        if not candidate:
            continue
        # Resolve relative URLs against the final (post-redirect) page URL
        absolute = urljoin(final_url, candidate)
        if absolute.startswith(("http://", "https://")):
            return absolute
    return ""
