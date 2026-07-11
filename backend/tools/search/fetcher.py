"""
Fetcher — parallel provider API calls with circuit breakers.

Handles:
- Parallel fetch via ThreadPoolExecutor (max 3 workers)
- Per-provider timeouts from providers table
- Per-provider rate limit delays
- Circuit breaker pattern (3 failures → open, 60s recovery probe)
- DDG auto-fallback when all providers return empty
"""

import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import cast
from urllib.parse import quote_plus

import requests

from tools.search.transformers import transform
from exceptions.rate_limit import RateLimitException

logger = logging.getLogger(__name__)

_MAX_WORKERS = 3

_DDG_TIME_RANGE_MAP = {"day": "d", "week": "w", "month": "m", "year": "y"}


# ── Circuit Breaker ──────────────────────────────────────────────────────────

_breakers: dict[str, dict[str, object]] = {}  # provider_name → breaker state
_breaker_lock = threading.RLock()

_BREAKER_THRESHOLD = 3     # consecutive failures to trip
_BREAKER_RECOVERY_S = 60   # seconds before probe attempt


def _get_breaker(name: str) -> dict[str, object]:
    """Get or create breaker state for a provider."""
    with _breaker_lock:
        if name not in _breakers:
            _breakers[name] = {
                'consecutive_failures': 0,
                'state': 'closed',
                'opened_at': None,
            }
        return _breakers[name]


def _is_breaker_open(name: str) -> bool:
    """Check if a provider's circuit breaker is open (tripped)."""
    with _breaker_lock:
        b = _get_breaker(name)
        if b['state'] == 'closed':
            return False
        if b['opened_at'] and (time.time() - cast("float", b['opened_at'])) >= _BREAKER_RECOVERY_S:
            return False  # allow one probe
        return True


def _record_success(name: str) -> None:
    """Record a successful fetch — reset breaker."""
    with _breaker_lock:
        b = _get_breaker(name)
        b['consecutive_failures'] = 0
        b['state'] = 'closed'
        b['opened_at'] = None


def _record_failure(name: str) -> None:
    """Record a failed fetch — increment failures, trip if threshold met."""
    with _breaker_lock:
        b = _get_breaker(name)
        b['consecutive_failures'] = cast("int", b['consecutive_failures']) + 1
        if cast("int", b['consecutive_failures']) >= _BREAKER_THRESHOLD:
            b['state'] = 'open'
            b['opened_at'] = time.time()
            logger.warning(
                '[SEARCH] circuit breaker OPEN for %s after %s failures',
                name, b['consecutive_failures'],
            )


# ── Rate limit tracking ─────────────────────────────────────────────────────

_last_call_times: dict[str, float] = {}  # provider_name → timestamp
_rate_lock = threading.RLock()


def _enforce_rate_limit(provider: dict[str, object]) -> None:
    """Sleep if needed to respect provider rate limits."""
    name = cast("str", provider['name'])

    # Mandatory delay (e.g. ArXiv 3s)
    delay = cast("float", provider.get('request_delay_seconds', 0))

    # Rate limit based delay
    rate = provider.get('rate_limit_per_second')
    if rate and cast("float", rate) > 0:
        min_interval = 1.0 / cast("float", rate)
        delay = max(delay, min_interval)

    if delay <= 0:
        return

    with _rate_lock:
        last = _last_call_times.get(name, 0)
        elapsed = time.time() - last
        if elapsed < delay:
            time.sleep(delay - elapsed)
        _last_call_times[name] = time.time()


# ── Single provider fetch ────────────────────────────────────────────────────

def _fetch_one(provider: dict[str, object], query: str, limit: int) -> list[dict[str, object]]:
    """
    Fetch results from a single provider.

    Returns list of standardized result dicts, or empty list on error.
    """
    name = cast("str", provider['name'])

    if _is_breaker_open(name):
        logger.info('[SEARCH] skipping %s — circuit breaker open', name)
        return []

    _enforce_rate_limit(provider)

    # Build URL
    url = cast("str", provider['endpoint_template']).format(
        query=quote_plus(query),
        limit=min(limit, 10),
    )

    # Build headers
    headers: dict[str, str] = {'User-Agent': 'Chalie/1.0 (cognitive-agent)'}
    if provider.get('headers_json'):
        import json
        try:
            extra = cast("dict[str, str]", json.loads(cast("str", provider['headers_json'])))
            headers.update(extra)
        except (json.JSONDecodeError, TypeError):
            pass

    timeout = float(cast("float", provider.get('timeout_seconds', 8)))
    fmt = cast("str", provider['response_format'])

    t0 = time.time()
    try:
        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=timeout,
                allow_redirects=True,
                verify=True,
            )
        except requests.exceptions.SSLError:
            logger.debug('[SEARCH] SSL verify failed for %s, retrying without verification', name)
            response = requests.get(
                url,
                headers=headers,
                timeout=timeout,
                allow_redirects=True,
                verify=False,  # noqa: S501 — intentional fallback for providers with broken certs
            )
        response.raise_for_status()

        latency_ms = int((time.time() - t0) * 1000)

        # Get raw data based on format
        # Note: json_gzip is auto-decompressed by requests (Accept-Encoding),
        # so treat it like regular JSON
        if fmt in ('atom_xml', 'rss_xml'):
            raw_data = response.text
        else:
            raw_data = response.json()

        results = transform(name, fmt, raw_data, limit)

        _record_success(name)

        logger.info(
            f'[SEARCH] provider={name} status=ok results={len(results)} '
            f'latency_ms={latency_ms}'
        )

        return results

    except requests.Timeout:
        latency_ms = int((time.time() - t0) * 1000)
        _record_failure(name)
        logger.warning(
            f'[SEARCH] provider={name} error=timeout latency_ms={latency_ms}'
        )
        return []

    except requests.HTTPError as e:
        latency_ms = int((time.time() - t0) * 1000)
        _record_failure(name)
        logger.warning(
            f'[SEARCH] provider={name} '
            f'error=HTTP_{e.response.status_code if e.response is not None else "?"} '
            f'latency_ms={latency_ms}'
        )
        return []

    except Exception as e:
        latency_ms = int((time.time() - t0) * 1000)
        _record_failure(name)
        logger.warning(
            f'[SEARCH] provider={name} error={type(e).__name__} '
            f'detail={str(e)[:120]} latency_ms={latency_ms}'
        )
        return []


# ── Parallel fetch ───────────────────────────────────────────────────────────

def fetch_providers(providers: list[dict[str, object]], query: str, limit: int = 5) -> list[dict[str, object]]:
    """
    Fetch results from multiple providers in parallel.

    Args:
        providers: List of provider dicts (from providers table)
        query: Search query string
        limit: Max results per provider

    Returns:
        Flattened list of standardized result dicts from all providers.
    """
    if not providers:
        return []

    # Single provider — no thread pool overhead
    if len(providers) == 1:
        return _fetch_one(providers[0], query, limit)

    all_results: list[dict[str, object]] = []

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
        futures = {
            executor.submit(_fetch_one, p, query, limit): cast("str", p['name'])
            for p in providers
        }

        for future in as_completed(futures):
            provider_name = futures[future]
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as e:
                logger.warning(
                    f'[SEARCH] fetch thread error for {provider_name}: {e}'
                )

    return all_results


def _transform_ddg_results(raw: list[dict[str, object]]) -> list[dict[str, object]]:
    """Convert raw DDG result dicts to the standard result format."""
    results: list[dict[str, object]] = []
    seen: set[str] = set()
    for r in raw:
        url = (cast("str", r.get("href")) or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        snippet = re.sub(r"\s{2,}", " ", (cast("str", r.get("body")) or "").strip())
        results.append({
            'title': (cast("str", r.get('title')) or '').strip(),
            'snippet': snippet,
            'url': url,
            'provider': 'ddg',
            'date': None,
        })
    return results


def fetch_ddg_fallback(query: str, limit: int = 5) -> list[dict[str, object]]:
    """Fall back to DDG web search. Returns results in standard format.

    Raises:
        RateLimitException: DDG enforced a rate limit — let it surface to the
            caller instead of retrying or silently returning empty.
    """
    try:
        from ddgs import DDGS
        from ddgs.exceptions import RatelimitException, DDGSException as DuckDuckGoSearchException

        limit = max(1, min(8, limit))

        try:
            raw = cast("list[dict[str, object]]", list(DDGS().text(query, max_results=limit)))
        except RatelimitException as e:
            raise RateLimitException("DDG web search rate-limited") from e
        except (DuckDuckGoSearchException, Exception) as e:
            logger.warning(f'[SEARCH] DDG fallback error: {e}')
            return []

        return _transform_ddg_results(raw)
    except RateLimitException:
        raise
    except Exception as e:
        logger.warning(f'[SEARCH] DDG fallback failed: {e}')
        return []
