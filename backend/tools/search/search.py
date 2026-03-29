"""
Search Tool — multi-provider search with semantic routing.

Entry point for the unified search tool. Routes queries to the best
provider(s) via embedding similarity, or allows explicit provider selection.

Phase 1: Explicit provider selection only (routing comes in Phase 2).
"""

import logging
import os
import sqlite3
import time

from tools.search.fetcher import fetch_providers, fetch_ddg_fallback

logger = logging.getLogger(__name__)

_DB_PATH = os.path.join(os.path.dirname(__file__), 'data', 'providers.sqlite')

# In-memory cache of provider metadata (loaded once)
_providers_cache: dict | None = None
_providers_lock = None


def _get_providers_lock():
    """Lazy init threading lock (avoids import-time side effects)."""
    global _providers_lock
    if _providers_lock is None:
        import threading
        _providers_lock = threading.Lock()
    return _providers_lock


def _load_providers() -> dict:
    """Load all enabled providers from SQLite into memory. Cached."""
    global _providers_cache

    lock = _get_providers_lock()
    with lock:
        if _providers_cache is not None:
            return _providers_cache

        providers = {}
        try:
            conn = sqlite3.connect(_DB_PATH)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                'SELECT * FROM providers WHERE enabled = 1'
            ).fetchall()
            conn.close()

            for row in rows:
                providers[row['name']] = dict(row)

        except Exception as e:
            logger.error(f'[SEARCH] Failed to load providers: {e}')

        _providers_cache = providers
        return providers


def _get_provider(name: str) -> dict | None:
    """Get a single provider by name."""
    providers = _load_providers()
    return providers.get(name)


# ── Tool entry point ─────────────────────────────────────────────────────────

def execute(topic: str, params: dict, config: dict = None, telemetry: dict = None) -> dict:
    """
    Search across multiple providers with semantic routing.

    Args:
        topic: Conversation topic (unused)
        params: {
            "query": str (required),
            "provider": str (optional — force a specific provider or 'ddg'),
            "limit": int (optional, default 5, max 10)
        }
        config: Tool config (unused)
        telemetry: Client telemetry (unused)

    Returns:
        {
            "results": [{"title", "snippet", "url", "provider", "date"}],
            "count": int,
            "providers_used": [str],
            "_meta": {observability fields}
        }
    """
    query = (params.get('query') or '').strip()
    if not query:
        return {'results': [], 'count': 0, 'providers_used': [], '_meta': {}}

    limit_raw = params.get('limit')
    limit = max(1, min(10, int(limit_raw) if limit_raw is not None else 5))

    forced_provider = (params.get('provider') or '').strip().lower()

    t0 = time.time()

    if forced_provider:
        results, providers_used, meta = _execute_forced(
            query, forced_provider, limit,
        )
    else:
        results, providers_used, meta = _execute_routed(
            query, limit,
        )

    fetch_latency_ms = int((time.time() - t0) * 1000)
    meta['fetch_latency_ms'] = fetch_latency_ms

    # DDG auto-fallback if all providers returned empty
    if not results and 'ddg' not in providers_used:
        logger.info('[SEARCH] all providers empty, falling back to DDG')
        results = fetch_ddg_fallback(query, limit)
        if results:
            providers_used.append('ddg')
            meta['ddg_fallback'] = True

    logger.info(
        f'[SEARCH] query="{query}" providers_used={providers_used} '
        f'result_count={len(results)} latency_ms={fetch_latency_ms}'
    )

    return {
        'results': results,
        'count': len(results),
        'providers_used': providers_used,
        '_meta': meta,
    }


# ── Forced provider ──────────────────────────────────────────────────────────

def _execute_forced(query: str, provider_name: str, limit: int):
    """Execute search against a specific provider."""
    if provider_name == 'ddg':
        results = fetch_ddg_fallback(query, limit)
        return results, ['ddg'], {'routing_method': 'forced', 'forced_provider': 'ddg'}

    provider = _get_provider(provider_name)
    if not provider:
        return [], [], {
            'routing_method': 'forced',
            'forced_provider': provider_name,
            'error': f'Unknown provider: {provider_name}',
        }

    results = fetch_providers([provider], query, limit)
    return results, [provider_name], {
        'routing_method': 'forced',
        'forced_provider': provider_name,
    }


# ── Routed search (Phase 2 — currently falls back to multi-provider) ────────

def _execute_routed(query: str, limit: int):
    """
    Route query to best provider(s) via semantic matching.

    Phase 1: Try router, fall back to DDG if router not available yet.
    Phase 2: Full embedding-based routing.
    """
    try:
        from tools.search.router import route_query
        ranked = route_query(query)

        if ranked:
            provider_names = [p['name'] for p in ranked]
            provider_scores = {p['name']: p['score'] for p in ranked}

            # Fetch from top providers
            provider_dicts = []
            for p in ranked:
                provider = _get_provider(p['name'])
                if provider:
                    provider_dicts.append(provider)

            results = fetch_providers(provider_dicts, query, limit)

            return results, [p['name'] for p in provider_dicts], {
                'routing_method': 'auto',
                'providers_considered': provider_names,
                'provider_scores': provider_scores,
            }

    except ImportError:
        # Router not built yet (Phase 1)
        pass
    except Exception as e:
        logger.warning(f'[SEARCH] router error, falling back to DDG: {e}')

    # Fallback: DDG
    results = fetch_ddg_fallback(query, limit)
    return results, ['ddg'], {
        'routing_method': 'fallback',
        'reason': 'router_unavailable',
    }
