"""Image Candidate Service — shortlist + caption for rich-media tools.

Used by search and news abilities to give the LLM enough information to pick
a meaningful thumbnail for the rich-media card without forcing it to fetch
images itself. Each candidate carries the image URL, the source article title
(so the LLM can match thumbnail → article), and a short caption derived from
the URL filename or article metadata.

Public API:
    build_image_candidates(items, top_n=3, og_meta=None) -> list[dict]

Each returned dict is ``{"url": str, "source_title": str, "caption": str}``.
Failed fetches are dropped silently — the LLM only sees viable candidates.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable
from urllib.parse import unquote, urlparse

import requests

logger = logging.getLogger(__name__)

_FETCH_TIMEOUT = 3.0
_USER_AGENT = "Chalie-RichMedia/1.0 (image-candidate)"

# Wikipedia CDN filenames are prefixed with a pixel-width like "1280px-".
_WIKI_PREFIX_RE = re.compile(r"^\d+px-", re.IGNORECASE)

# A filename is considered opaque when it has no meaningful word tokens.
# Words with ≥ 2 alphabetic characters qualify — but pure lowercase hex
# strings like "abcdef" also match that pattern, so we add a secondary
# check: a single-token result that is entirely hex characters is a hash.
_WORD_RE = re.compile(r"[A-Za-z]{2,}")
_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")


def build_image_candidates(
    items: Iterable,
    top_n: int = 3,
    og_meta: dict | None = None,
) -> list[dict]:
    """Shortlist up to top_n image URLs, validate each via HTTP HEAD, return survivors with captions.

    ``og_meta`` maps article/image URL to its og metadata (from ``resolve_og_images``);
    when present, the description for each image source URL is used as the caption.
    """
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in items:
        if isinstance(item, tuple) and len(item) == 2:
            url, source_title = item
        elif isinstance(item, str):
            url, source_title = item, ""
        else:
            continue
        if not url or not isinstance(url, str) or url in seen:
            continue
        seen.add(url)
        pairs.append((url, source_title or ""))
        if len(pairs) >= top_n:
            break

    if not pairs:
        return []

    fetched: set[str] = set()
    with ThreadPoolExecutor(max_workers=len(pairs)) as pool:
        futures = {pool.submit(_check_image_url, url): url for url, _ in pairs}
        for future in as_completed(futures):
            url = futures[future]
            try:
                ok = future.result()
            except Exception as exc:
                logger.debug("image_candidate: fetch failed for %s: %s", url, exc)
                continue
            if ok:
                fetched.add(url)

    out: list[dict] = []
    for url, source_title in pairs:
        if url not in fetched:
            continue
        caption = _derive_caption(url, og_meta)
        out.append({"url": url, "source_title": source_title, "caption": caption})
    return out


def _check_image_url(url: str) -> bool:
    """Check that the URL returns an image content-type and non-empty body."""
    try:
        with requests.get(
            url,
            timeout=_FETCH_TIMEOUT,
            stream=True,
            headers={"User-Agent": _USER_AGENT, "Accept": "image/*"},
        ) as resp:
            resp.raise_for_status()
            content_type = (resp.headers.get("Content-Type") or "").lower()
            if content_type and not content_type.startswith("image/"):
                return False
            for chunk in resp.iter_content(chunk_size=4096):
                if chunk:
                    return True
            return False
    except Exception as exc:
        logger.debug("image_candidate: GET %s failed: %s", url, exc)
        return False


def _derive_caption(image_url: str, og_meta: dict | None) -> str:
    """Derive a caption with priority: 1) og:description (og_meta if key matches image_url and ≥ 20 chars), 2) filename heuristic."""
    # og_meta is keyed by article URL.  The image URL itself won't match, but
    # callers that pass og_meta also key it by image_url for direct lookup.
    if og_meta:
        meta = og_meta.get(image_url)
        if meta:
            desc = (meta.get("description") or "").strip()
            if len(desc) >= 20:
                return desc

    return _caption_from_filename(image_url)


def _caption_from_filename(url: str) -> str:
    """Derive a caption from the URL path filename.

    Wikipedia (the dominant image source) uses descriptive filenames like:
    ``Mt._Everest_from_Gokyo_Ri_November_5%2C_2012.jpg``

    Steps:
    1. URL-decode the path.
    2. Take the last path segment (filename).
    3. Strip Wikipedia thumbnail prefix ``\\d+px-``.
    4. Strip file extension.
    5. Replace underscores with spaces, collapse whitespace.
    6. Return empty string if the result has no meaningful word tokens
       (opaque CDN hash filenames like ``abc123def456.jpg``).
    """
    try:
        path = urlparse(url).path
    except Exception:
        return ""

    filename = path.rstrip("/").rsplit("/", 1)[-1]
    if not filename:
        return ""

    filename = unquote(filename)

    # Strip Wikipedia thumbnail prefix (e.g. "1280px-", "960px-")
    filename = _WIKI_PREFIX_RE.sub("", filename)

    # Strip extension
    if "." in filename:
        filename = filename.rsplit(".", 1)[0]

    # Underscores → spaces
    caption = filename.replace("_", " ").strip()

    # Collapse multiple spaces
    caption = re.sub(r"\s{2,}", " ", caption)

    if not caption:
        return ""

    # Opaque heuristic 1: no word has ≥ 2 alphabetic characters → treat as hash
    if not _WORD_RE.search(caption):
        return ""

    # Opaque heuristic 2: single token that is entirely hex digits is a CDN hash
    # (e.g. "abcdef1234567890abcdef1234567890" — looks alphabetic but is a hash)
    if " " not in caption and _HEX_RE.match(caption):
        return ""

    return caption
