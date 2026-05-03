"""Image Candidate Service — shortlist + OCR for rich-media tools.

Used by search and news abilities to give the LLM enough information to pick
a meaningful thumbnail for the rich-media card without forcing it to fetch
images itself. Each candidate carries the image URL plus the text RapidOCR
extracted from the image — so the LLM can match a candidate against the
result it cites and decide whether to render it (or skip the image slot).

Public API:
    build_image_candidates(urls: list[str], top_n: int = 3) -> list[dict]

Each returned dict is ``{"url": str, "ocr_text": str}``. Failed fetches and
images that yield no text are dropped silently — the LLM only sees viable
candidates.
"""

from __future__ import annotations

import io
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

import requests

logger = logging.getLogger(__name__)

_FETCH_TIMEOUT = 3.0
_MAX_BYTES = 5 * 1024 * 1024
_OCR_TEXT_CAP = 300
_USER_AGENT = "Chalie-RichMedia/1.0 (image-candidate)"


def build_image_candidates(urls: Iterable[str], top_n: int = 3) -> list[dict]:
    """Shortlist up to top_n image URLs, fetch and OCR each, return survivors.

    URLs are deduplicated in input order. Fetching is parallel; OCR is
    sequential because RapidOCR's lazy engine init is lock-protected and
    parallel inference offers no measured win against fast text-light images.
    """
    candidates: list[str] = []
    seen: set[str] = set()
    for url in urls:
        if not url or not isinstance(url, str):
            continue
        if url in seen:
            continue
        seen.add(url)
        candidates.append(url)
        if len(candidates) >= top_n:
            break

    if not candidates:
        return []

    fetched: dict[str, bytes] = {}
    with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
        futures = {pool.submit(_fetch_image_bytes, url): url for url in candidates}
        for future in as_completed(futures):
            url = futures[future]
            try:
                data = future.result()
            except Exception as exc:
                logger.debug("image_candidate: fetch failed for %s: %s", url, exc)
                continue
            if data:
                fetched[url] = data

    out: list[dict] = []
    for url in candidates:
        data = fetched.get(url)
        if not data:
            continue
        text = _ocr_bytes(data)
        out.append({"url": url, "ocr_text": text})
    return out


def _fetch_image_bytes(url: str) -> bytes | None:
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
                return None
            buf = bytearray()
            for chunk in resp.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                buf.extend(chunk)
                if len(buf) > _MAX_BYTES:
                    return None
            return bytes(buf)
    except Exception as exc:
        logger.debug("image_candidate: GET %s failed: %s", url, exc)
        return None


def _ocr_bytes(data: bytes) -> str:
    try:
        from PIL import Image
        from services.ocr_service import _extract_text
    except Exception as exc:
        logger.warning("image_candidate: OCR unavailable (%s)", exc)
        return ""

    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            text = _extract_text(img)
    except Exception as exc:
        logger.debug("image_candidate: OCR failed: %s", exc)
        return ""

    text = (text or "").strip()
    if len(text) > _OCR_TEXT_CAP:
        text = text[:_OCR_TEXT_CAP].rstrip() + "…"
    return text
