"""Image search fetcher — DDG images backend only.

Thin wrapper around ``ddgs.images()``. No parallel providers, no circuit
breakers, no HTML scraping. Returns a deduplicated, capped list of image
results shaped ``{url, title, source, thumb_url}``.

Engine failure is loud: a DDG / network / parse error raises ``RuntimeError``
so the caller can surface it as an error result. An empty return means the
engine succeeded and genuinely found nothing — the two are never conflated.
"""

from __future__ import annotations

import logging

from ddgs.exceptions import RatelimitException as DDGRatelimitException

from exceptions.rate_limit import RateLimitException

logger = logging.getLogger(__name__)


def fetch(query: str, limit: int = 5) -> list[dict[str, str]]:
    """Search DDG images for ``query``.

    Args:
        query: Text query to search for.
        limit: Maximum number of results (clamped to 1..5).

    Returns:
        Up to ``limit`` de-duplicated dicts shaped
        ``{url, title, source, thumb_url}``. An empty list means the engine
        succeeded but found nothing.

    Raises:
        RateLimitException: The DDG engine enforced a rate limit.
        RuntimeError: The DDG engine failed (network, parse).
            Engine failure is loud so the caller can distinguish it from an
            empty result set.
    """
    if not query or not query.strip():
        return []

    limit = max(1, min(5, limit))

    seen: set[str] = set()
    results: list[dict[str, str]] = []

    try:
        from ddgs import DDGS

        raw = list(DDGS().images(query, max_results=limit))

        # Parsing is inside the try on purpose: a malformed item (e.g. a future
        # ddgs shape change) is an engine failure too, and must raise the same
        # loud RuntimeError the docstring promises — never leak an AttributeError.
        for item in raw:
            url = item.get("image") or ""
            if not url or url in seen:
                continue
            seen.add(url)
            results.append(
                {
                    "url": url,
                    "title": item.get("title") or "",
                    "source": item.get("url") or "",
                    "thumb_url": item.get("thumbnail") or "",
                }
            )
    except DDGRatelimitException:
        raise RateLimitException("DDG image search rate-limited")
    except Exception as e:
        logger.warning("[IMAGE_SEARCH] DDG engine error for %r: %s", query, e)
        raise RuntimeError(f"image search engine failed: {e}") from e

    return results[:limit]
