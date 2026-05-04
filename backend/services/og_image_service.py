"""OG Image Service — extract og:image / twitter:image + description from article URLs.

Used as a fallback when a search/news provider doesn't surface a native
image URL (Google News RSS strips them, for example). Fetches the article
HTML in parallel, scans the head for og:image and og:description / og:title
meta tags, and returns structured metadata per URL.

Public API:
    resolve_og_images(urls, max_workers=3, timeout=3.0) -> dict[str, dict]

Each value is ``{"image_url": str, "description": str}``.  Either field may
be an empty string if not found.  URLs that fail entirely are omitted.
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
# Google News article pages bury og:image after ~575KB of inline JSON/scripts,
# so the cap has to sit above ~600KB or news thumbnails never come through.
_MAX_BYTES = 1024 * 1024
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

# Match og:image / twitter:image meta tags — property/name in either order,
# single or double quotes.
_IMAGE_META_RE = re.compile(
    r"""<meta\s+[^>]*?(?:property|name)\s*=\s*['"](og:image(?::secure_url)?|twitter:image)['"][^>]*?>""",
    re.IGNORECASE | re.DOTALL,
)

# Match og:description / og:title meta tags.
_DESC_META_RE = re.compile(
    r"""<meta\s+[^>]*?property\s*=\s*['"](og:description|og:title)['"][^>]*?>""",
    re.IGNORECASE | re.DOTALL,
)

# Match <title>…</title>.
_TITLE_TAG_RE = re.compile(r"<title[^>]*?>([^<]+)</title>", re.IGNORECASE | re.DOTALL)

_CONTENT_RE = re.compile(
    r"""content\s*=\s*['"]([^'"]+)['"]""",
    re.IGNORECASE,
)

_DESC_CAP = 200


def resolve_og_images(
    urls: Iterable[str], max_workers: int = 3, timeout: float = _FETCH_TIMEOUT
) -> dict[str, dict]:
    """Fetch article HTML in parallel and extract og:image + description for each URL.

    Returns ``{article_url: {"image_url": str, "description": str}}`` for URLs
    that yielded at least an image URL.  Failures and pages with no og:image
    are omitted entirely — callers should not assume every input URL is present.

    ``description`` is derived from og:description ≥ 20 chars, falling back to
    og:title, then <title>.  It may be an empty string when none of the above
    are found.
    """
    targets = [u for u in urls if isinstance(u, str) and u.startswith(("http://", "https://"))]
    if not targets:
        return {}

    out: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=min(max_workers, len(targets))) as pool:
        futures = {pool.submit(_extract_one, u, timeout): u for u in targets}
        for future in as_completed(futures):
            url = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                logger.debug("og_image: %s failed: %s", url, exc)
                continue
            if result and result.get("image_url"):
                out[url] = result
    return out


def _extract_one(url: str, timeout: float) -> dict:
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
                return {}
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
        return {}

    text = buf.decode("utf-8", errors="ignore")
    image_url = _extract_image(text, final_url)
    description = _extract_description(text)
    if not image_url:
        return {}
    return {"image_url": image_url, "description": description}


def _extract_image(text: str, final_url: str) -> str:
    for tag_match in _IMAGE_META_RE.finditer(text):
        tag = tag_match.group(0)
        content_match = _CONTENT_RE.search(tag)
        if not content_match:
            continue
        candidate = content_match.group(1).strip()
        if not candidate:
            continue
        absolute = urljoin(final_url, candidate)
        if absolute.startswith(("http://", "https://")):
            return absolute
    return ""


def _extract_description(text: str) -> str:
    """Extract og:description (preferred) → og:title → <title>.

    Returns empty string when nothing is found.  Result is capped at
    ``_DESC_CAP`` characters with trailing whitespace stripped.
    """
    # og:description or og:title in preference order
    for tag_match in _DESC_META_RE.finditer(text):
        prop = tag_match.group(1).lower()
        tag = tag_match.group(0)
        content_match = _CONTENT_RE.search(tag)
        if not content_match:
            continue
        value = content_match.group(1).strip()
        if not value:
            continue
        if prop == "og:description" and len(value) >= 20:
            return value[:_DESC_CAP].rstrip()
        if prop == "og:title":
            # Keep looking for og:description — only use og:title as fallback
            title_fallback = value[:_DESC_CAP].rstrip()
            # Check if there's a better og:description later; if not, use this
            remaining = text[tag_match.end():]
            for later in _DESC_META_RE.finditer(remaining):
                if later.group(1).lower() == "og:description":
                    later_content = _CONTENT_RE.search(later.group(0))
                    if later_content:
                        later_val = later_content.group(1).strip()
                        if len(later_val) >= 20:
                            return later_val[:_DESC_CAP].rstrip()
            return title_fallback
    # Fall back to <title>
    m = _TITLE_TAG_RE.search(text)
    if m:
        return m.group(1).strip()[:_DESC_CAP].rstrip()
    return ""
