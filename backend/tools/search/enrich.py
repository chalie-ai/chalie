"""
Hybrid summary/date fallback — fetches page metadata for results whose
``summary`` field is blank.

Only results with an empty/blank ``summary`` are touched; results that already
have a summary are returned unchanged and never fetched.

Concurrency: at most ``_MAX_WORKERS`` pages are fetched simultaneously via a
``ThreadPoolExecutor``.  Every fetch/parse error is logged at DEBUG level and
leaves the affected result untouched — this is spec-mandated fail-open
best-effort enrichment, so a broad ``except Exception`` + log is intentional
here and nowhere else.

This runs AFTER the global top-N cut so at most a handful of pages are ever
fetched per search invocation.
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import cast

logger = logging.getLogger(__name__)

_MAX_WORKERS = 4

# Patterns to extract <meta> content values from raw HTML without a full parser.
_OG_DESC_RE = re.compile(
    r'<meta\s(?:(?!property=)[^>])*property=["\']og:description["\'](?:(?!content=)[^>])*content=["\']([^"\']*+)["\']',
    re.IGNORECASE | re.DOTALL,
)
_META_DESC_RE = re.compile(
    r'<meta\s(?:(?!name=)[^>])*name=["\']description["\'](?:(?!content=)[^>])*content=["\']([^"\']*+)["\']',
    re.IGNORECASE | re.DOTALL,
)
# Alternate attribute order: content= before property=/name=
_OG_DESC_RE2 = re.compile(
    r'<meta\s(?:(?!content=)[^>])*content=["\']([^"\']*+)["\'](?:(?!property=)[^>])*property=["\']og:description["\']',
    re.IGNORECASE | re.DOTALL,
)
_META_DESC_RE2 = re.compile(
    r'<meta\s(?:(?!content=)[^>])*content=["\']([^"\']*+)["\'](?:(?!name=)[^>])*name=["\']description["\']',
    re.IGNORECASE | re.DOTALL,
)

_OG_PUB_RE = re.compile(
    r'<meta\s(?:(?!property=)[^>])*property=["\'](?:article:published_time|og:published_time)["\'](?:(?!content=)[^>])*content=["\']([^"\']*+)["\']',
    re.IGNORECASE | re.DOTALL,
)
_OG_PUB_RE2 = re.compile(
    r'<meta\s(?:(?!content=)[^>])*content=["\']([^"\']*+)["\'](?:(?!property=)[^>])*property=["\'](?:article:published_time|og:published_time)["\']',
    re.IGNORECASE | re.DOTALL,
)


def _extract_meta(html: str) -> tuple[str, str | None]:
    """Return ``(description, date_or_none)`` extracted from *html*.

    ``description`` is ``og:description`` or ``<meta name="description">`` (in
    that preference order).  ``date`` is ``article:published_time`` or
    ``og:published_time`` truncated to the first 10 characters.
    """
    desc = ""
    for pat in (_OG_DESC_RE, _OG_DESC_RE2, _META_DESC_RE, _META_DESC_RE2):
        m = pat.search(html)
        if m:
            desc = m.group(1).strip()
            break

    date: str | None = None
    for pat in (_OG_PUB_RE, _OG_PUB_RE2):
        m = pat.search(html)
        if m:
            raw = m.group(1).strip()
            if raw:
                date = raw[:10]
            break

    return desc, date


def _enrich_one(result: dict[str, object]) -> dict[str, object]:
    """Fetch the page at ``result["url"]`` and fill in ``summary`` (and
    optionally ``date``) from the page's meta tags.

    Returns the result dict (potentially mutated) on success, or the original
    dict unchanged on any error.
    """
    url = cast("str", result.get("url", ""))
    if not url:
        return result

    try:
        from services.web_fetch import fetch_text
        html = fetch_text(url)
        desc, date = _extract_meta(html)
        if desc:
            result = {**result, "summary": desc}
        if date and not (cast("str", result.get("date")) or "").strip():
            result = {**result, "date": date}
    except Exception as exc:  # noqa: BLE001 — spec-mandated fail-open
        logger.debug("[SEARCH] enrich fetch failed for %s: %s", url, exc)

    return result


def _enrich_concurrently(
    need_enrich: list[tuple[int, dict[str, object]]],
) -> dict[int, dict[str, object]]:
    """Enrich the blank-summary subset concurrently, keyed by original index."""
    enriched_by_index: dict[int, dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(_enrich_one, r): idx
            for idx, r in need_enrich
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                enriched_by_index[idx] = future.result()
            except Exception as exc:  # noqa: BLE001 — belt-and-suspenders
                logger.debug("[SEARCH] enrich future error at index %d: %s", idx, exc)
                enriched_by_index[idx] = dict(need_enrich)[idx]
    return enriched_by_index


def enrich_missing_summaries(results: list[dict[str, object]]) -> list[dict[str, object]]:
    """Fill ``summary`` (and optionally ``date``) for results with blank summaries.

    Results that already have a non-empty ``summary`` are returned untouched
    without any network fetch.  Results with a blank/empty ``summary`` are
    fetched concurrently (up to ``_MAX_WORKERS`` at a time) and enriched
    best-effort from ``og:description`` / ``<meta name="description">`` meta
    tags.

    Args:
        results: List of result dicts.  The caller guarantees this is the
            post-cut list (at most a handful of items) so page-fetching is
            bounded.

    Returns:
        A new list in the same order as *results* with blank-summary items
        enriched where possible.  Never raises.
    """
    if not results:
        return results

    # Separate items that need enrichment from those that do not.
    need_enrich: list[tuple[int, dict[str, object]]] = []
    already_ok: dict[int, dict[str, object]] = {}
    for i, r in enumerate(results):
        if (cast("str", r.get("summary")) or "").strip():
            already_ok[i] = r
        else:
            need_enrich.append((i, r))

    if not need_enrich:
        return results

    # Fetch concurrently for the subset that needs enrichment.
    enriched_by_index: dict[int, dict[str, object]] = {}
    if len(need_enrich) == 1:
        idx, r = need_enrich[0]
        enriched_by_index[idx] = _enrich_one(r)
    else:
        enriched_by_index = _enrich_concurrently(need_enrich)

    # Reassemble in original order.
    out: list[dict[str, object]] = []
    for i, r in enumerate(results):
        if i in already_ok:
            out.append(already_ok[i])
        else:
            out.append(enriched_by_index.get(i, r))
    return out
