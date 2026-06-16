

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable
from urllib.parse import urljoin

import requests

logger = logging.getLogger(__name__)

_FETCH_TIMEOUT = 3.0
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
    r"""<meta\s+[^>]*(?:property|name)\s*=\s*['"](og:image(?::secure_url)?|twitter:image)['"][^>]*>""",
    re.IGNORECASE,
)

# Match og:description / og:title meta tags.
_DESC_META_RE = re.compile(
    r"""<meta\s+[^>]*property\s*=\s*['"](og:description|og:title)['"][^>]*>""",
    re.IGNORECASE,
)

# Match <title>…</title>.
_TITLE_TAG_RE = re.compile(r"<title[^>]*?>([^<]+)</title>", re.IGNORECASE | re.DOTALL)

_CONTENT_RE = re.compile(
    r"""content\s*=\s*['"]([^'"]+)['"]""",
    re.IGNORECASE,
)


def resolve_og_images(
    urls: Iterable[str], max_workers: int = 3, timeout: float = _FETCH_TIMEOUT
) -> dict[str, dict]:
    """description derived from og:description (≥ 20 chars) → og:title → <title>."""
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
        try:
            resp_ctx = requests.get(
                url,
                timeout=timeout,
                stream=True,
                allow_redirects=True,
                headers={"User-Agent": _USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
                cookies=_DEFAULT_COOKIES,
                verify=True,
            )
        except requests.exceptions.SSLError:
            logger.debug("og_image: SSL verify failed for %s, retrying without verification", url)
            resp_ctx = requests.get(
                url,
                timeout=timeout,
                stream=True,
                allow_redirects=True,
                headers={"User-Agent": _USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
                cookies=_DEFAULT_COOKIES,
                verify=False,  # noqa: S501 — intentional fallback for sites with broken certs
            )
        with resp_ctx as resp:
            resp.raise_for_status()
            ct = (resp.headers.get("Content-Type") or "").lower()
            if "html" not in ct:
                return {}
            buf = bytearray()
            for chunk in resp.iter_content(chunk_size=16 * 1024, decode_unicode=False):
                if not chunk:
                    continue
                buf.extend(chunk)
                # Early exit once </head> seen — head is all we need.
                # The request timeout bounds total work; no byte cap.
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

    Returns empty string when nothing is found.
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
            return value
        if prop == "og:title":
            # Keep looking for og:description — only use og:title as fallback
            title_fallback = value
            remaining = text[tag_match.end():]
            for later in _DESC_META_RE.finditer(remaining):
                if later.group(1).lower() == "og:description":
                    later_content = _CONTENT_RE.search(later.group(0))
                    if later_content:
                        later_val = later_content.group(1).strip()
                        if len(later_val) >= 20:
                            return later_val
            return title_fallback
    # Fall back to <title>
    m = _TITLE_TAG_RE.search(text)
    if m:
        return m.group(1).strip()
    return ""
