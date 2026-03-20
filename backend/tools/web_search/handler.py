"""
Web Search Tool Handler — DuckDuckGo web search.

Privacy-focused, zero-config search via duckduckgo-search library.
Rate-limit aware: 2s cooldown between calls, 3 retries with exponential backoff.
Parallel image fetch (1.5s cap) runs alongside text search.
"""

import logging
import re
import threading
import time
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Minimum seconds between DDG API calls (rate-limit guard)
_DDG_COOLDOWN = 2.0
_last_call_time = 0.0
_call_lock = threading.Lock()

_TIME_RANGE_MAP = {
    "day": "d",
    "week": "w",
    "month": "m",
    "year": "y",
}


def execute(topic: str, params: dict, config: dict = None, telemetry: dict = None) -> dict:
    """
    Search the web via DuckDuckGo and return top results.

    Args:
        topic: Conversation topic (unused directly)
        params: {
            "query": str (required),
            "limit": int (optional, default 5, clamped 1-8),
            "time_range": str (optional: day/week/month/year)
        }
        config: Tool config (unused — no API key needed)
        telemetry: Client telemetry (unused)

    Returns:
        {
            "results": [{"title", "snippet", "url", "domain"}],
            "count": int,
            "_meta": {observability fields}
        }
    """
    query = (params.get("query") or "").strip()
    if not query:
        return {"results": [], "count": 0, "_meta": {}}

    limit_raw = params.get("limit")
    limit = max(1, min(8, int(limit_raw) if limit_raw is not None else 5))
    time_range_raw = (params.get("time_range") or "").strip().lower()
    timelimit = _TIME_RANGE_MAP.get(time_range_raw)

    # Launch image fetch in parallel thread (bounded 1.5s)
    image_future_result = []
    image_thread = threading.Thread(
        target=_fetch_images_ddg,
        args=(query, image_future_result),
        daemon=True,
    )
    image_thread.start()

    t0 = time.time()
    results, retry_used, error = _search_ddg(query, limit, timelimit)
    fetch_latency_ms = int((time.time() - t0) * 1000)

    # Collect images (bounded wait)
    image_thread.join(timeout=1.5)
    images = image_future_result[0] if image_future_result else []

    if error and not results:
        logger.error(
            '{"event":"ddg_fetch_error","query":"%s","error":"%s","latency_ms":%d}',
            query, str(error)[:120], fetch_latency_ms,
        )
        return {"results": [], "count": 0, "error": str(error)[:200], "_meta": {}}

    unique_domains = len({_domain(r["url"]) for r in results if r.get("url")})

    logger.info(
        '{"event":"ddg_search_ok","query":"%s","count":%d,"unique_domains":%d,'
        '"has_images":%s,"retry_used":%s,"latency_ms":%d}',
        query, len(results), unique_domains,
        str(bool(images)).lower(), str(retry_used).lower(), fetch_latency_ms,
    )

    return {
        "results": results,
        "count": len(results),
        "_meta": {
            "fetch_latency_ms": fetch_latency_ms,
            "source_count": len(results),
            "unique_domains": unique_domains,
            "has_images": bool(images),
            "retry_used": retry_used,
            "time_range": time_range_raw or None,
        },
    }


# ── DuckDuckGo search ─────────────────────────────────────────────────────────

def _search_ddg(query: str, limit: int, timelimit: str | None):
    """Run DDG text search with cooldown enforcement and retry on rate-limit."""
    from duckduckgo_search import DDGS
    from duckduckgo_search.exceptions import RatelimitException, DuckDuckGoSearchException

    global _last_call_time

    retry_used = False
    last_error = None

    for attempt in range(3):
        # Enforce minimum cooldown between calls
        with _call_lock:
            elapsed = time.time() - _last_call_time
            if elapsed < _DDG_COOLDOWN:
                time.sleep(_DDG_COOLDOWN - elapsed)
            _last_call_time = time.time()

        try:
            raw = list(DDGS().text(
                keywords=query,
                max_results=limit,
                timelimit=timelimit,
            ))

            results = []
            seen_urls = set()
            for r in raw:
                url = (r.get("href") or "").strip()
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                results.append({
                    "title": (r.get("title") or "").strip(),
                    "snippet": _clean_snippet(r.get("body") or ""),
                    "url": url,
                    "domain": _domain(url),
                })

            return results, retry_used, None

        except RatelimitException as e:
            last_error = e
            retry_used = True
            backoff = 2 ** attempt * 3  # 3s, 6s, 12s
            logger.warning(
                '{"event":"ddg_ratelimit","attempt":%d,"backoff_s":%d}',
                attempt + 1, backoff,
            )
            time.sleep(backoff)

        except DuckDuckGoSearchException as e:
            last_error = e
            logger.warning(
                '{"event":"ddg_search_exception","attempt":%d,"error":"%s"}',
                attempt + 1, str(e)[:120],
            )
            break

        except Exception as e:
            last_error = e
            logger.warning(
                '{"event":"ddg_unexpected_error","attempt":%d,"error":"%s"}',
                attempt + 1, str(e)[:120],
            )
            break

    return [], retry_used, last_error


def _fetch_images_ddg(query: str, result_holder: list) -> None:
    """Fetch up to 3 DDG image results. Writes into result_holder[0]."""
    try:
        from duckduckgo_search import DDGS
        raw = list(DDGS().images(keywords=query, max_results=3))
        images = [
            {
                "url": r.get("image", ""),
                "thumbnail": r.get("thumbnail", ""),
                "title": (r.get("title") or "").strip(),
                "source": r.get("url", ""),
            }
            for r in raw
            if r.get("image")
        ]
        result_holder.append(images)
    except Exception:
        result_holder.append([])


# ── Utilities ─────────────────────────────────────────────────────────────────

def _domain(url: str) -> str:
    """Extract domain from URL for dedup and display."""
    try:
        return urlparse(url).netloc.lstrip("www.")
    except Exception:
        return ""


def _clean_snippet(text: str) -> str:
    """Strip excess whitespace from DDG snippets."""
    text = re.sub(r"\s{2,}", " ", text.strip())
    return text[:300] + ("\u2026" if len(text) > 300 else "")
