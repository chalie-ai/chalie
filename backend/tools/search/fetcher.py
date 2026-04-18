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
from urllib.parse import quote_plus, urlparse

import requests

from tools.search.transformers import transform

logger = logging.getLogger(__name__)

_MAX_WORKERS = 3

# ── DuckDuckGo rate-limit guard ──────────────────────────────────────────────
_DDG_COOLDOWN = 2.0
_ddg_last_call = 0.0
_ddg_lock = threading.Lock()

_DDG_TIME_RANGE_MAP = {"day": "d", "week": "w", "month": "m", "year": "y"}


# ── Circuit Breaker ──────────────────────────────────────────────────────────

_breakers: dict = {}  # provider_name → breaker state
_breaker_lock = threading.RLock()

_BREAKER_THRESHOLD = 3     # consecutive failures to trip
_BREAKER_RECOVERY_S = 60   # seconds before probe attempt


def _get_breaker(name: str) -> dict:
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
        if b['opened_at'] and (time.time() - b['opened_at']) >= _BREAKER_RECOVERY_S:
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
        b['consecutive_failures'] += 1
        if b['consecutive_failures'] >= _BREAKER_THRESHOLD:
            b['state'] = 'open'
            b['opened_at'] = time.time()
            logger.warning(
                f'[SEARCH] circuit breaker OPEN for {name} '
                f'after {b["consecutive_failures"]} failures'
            )


# ── Rate limit tracking ─────────────────────────────────────────────────────

_last_call_times: dict = {}  # provider_name → timestamp
_rate_lock = threading.RLock()


def _enforce_rate_limit(provider: dict) -> None:
    """Sleep if needed to respect provider rate limits."""
    name = provider['name']

    # Mandatory delay (e.g. ArXiv 3s)
    delay = provider.get('request_delay_seconds', 0)

    # Rate limit based delay
    rate = provider.get('rate_limit_per_second')
    if rate and rate > 0:
        min_interval = 1.0 / rate
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

def _fetch_one(provider: dict, query: str, limit: int) -> list:
    """
    Fetch results from a single provider.

    Returns list of standardized result dicts, or empty list on error.
    """
    name = provider['name']

    if _is_breaker_open(name):
        logger.info(f'[SEARCH] skipping {name} — circuit breaker open')
        return []

    _enforce_rate_limit(provider)

    # Build URL
    url = provider['endpoint_template'].format(
        query=quote_plus(query),
        limit=min(limit, 10),
    )

    # Build headers
    headers = {'User-Agent': 'Chalie/1.0 (cognitive-agent)'}
    if provider.get('headers_json'):
        import json
        try:
            extra = json.loads(provider['headers_json'])
            headers.update(extra)
        except (json.JSONDecodeError, TypeError):
            pass

    timeout = provider.get('timeout_seconds', 8)
    fmt = provider['response_format']

    t0 = time.time()
    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=timeout,
            allow_redirects=True,
            verify=False,
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
            f'[SEARCH] provider={name} error=HTTP_{e.response.status_code} '
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

def fetch_providers(providers: list, query: str, limit: int = 5) -> list:
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

    all_results = []

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
        futures = {
            executor.submit(_fetch_one, p, query, limit): p['name']
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


def fetch_ddg_fallback(query: str, limit: int = 5) -> list:
    """Fall back to DDG web search. Returns results in standard format."""
    try:
        from ddgs import DDGS
        from ddgs.exceptions import RatelimitException, DDGSException as DuckDuckGoSearchException

        global _ddg_last_call
        limit = max(1, min(8, limit))

        for attempt in range(3):
            with _ddg_lock:
                elapsed = time.time() - _ddg_last_call
                if elapsed < _DDG_COOLDOWN:
                    time.sleep(_DDG_COOLDOWN - elapsed)
                _ddg_last_call = time.time()
            try:
                raw = list(DDGS().text(query, max_results=limit))
                results = []
                seen = set()
                for r in raw:
                    url = (r.get("href") or "").strip()
                    if not url or url in seen:
                        continue
                    seen.add(url)
                    snippet = re.sub(r"\s{2,}", " ", (r.get("body") or "").strip())[:300]
                    try:
                        urlparse(url).netloc.lstrip("www.")
                    except Exception:
                        pass
                    results.append({
                        'title': (r.get('title') or '').strip(),
                        'snippet': snippet,
                        'url': url,
                        'provider': 'ddg',
                        'date': None,
                    })
                return results
            except RatelimitException:
                time.sleep(2 ** attempt * 3)
            except (DuckDuckGoSearchException, Exception) as e:
                logger.warning(f'[SEARCH] DDG fallback error (attempt {attempt+1}): {e}')
                break
        return []
    except Exception as e:
        logger.warning(f'[SEARCH] DDG fallback failed: {e}')
        return []
