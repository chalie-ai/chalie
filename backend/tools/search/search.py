"""
Search tool — multi-provider search with semantic routing.

Routes queries to best provider(s) via embedding similarity,
or allows explicit provider selection. Falls back to DDG.
"""

import json
import logging
import os
import sqlite3
import time

from tools.search.fetcher import fetch_providers, fetch_ddg_fallback

logger = logging.getLogger(__name__)

_DB = os.path.join(os.path.dirname(__file__), 'assets', 'search_tool_providers.sqlite')

# Provider metadata cache (loaded once, 12 rows)
_providers: dict | None = None


def _load_providers() -> dict:
    """Load enabled providers from SQLite. Cached after first success."""
    global _providers
    if _providers is not None:
        return _providers

    try:
        conn = sqlite3.connect(_DB)
        conn.row_factory = sqlite3.Row
        rows = conn.execute('SELECT * FROM providers WHERE enabled = 1').fetchall()
        conn.close()
        _providers = {row['name']: dict(row) for row in rows}
    except Exception as e:
        logger.error('[SEARCH] Failed to load providers: %s', e)
        return {}

    return _providers


def execute(topic: str, params: dict, config: dict = None, telemetry: dict = None) -> dict:
    """Search across providers. Params: query (str), provider (str, optional), limit (int, optional).

    Returns {'text': json_or_error} where text is either:
    - JSON: {"results":[{"title":"...","desc":"...","url":"..."},...]}
    - Plain error string on failure (timeout, rate limit, no results)
    """
    query = (params.get('query') or '').strip()
    if not query:
        return {'text': 'No results found'}

    limit = max(1, min(10, int(params['limit']) if params.get('limit') is not None else 5))
    forced = (params.get('provider') or '').strip().lower()

    t0 = time.time()

    if forced:
        results, used, meta = _search_forced(query, forced, limit)
    else:
        results, used, meta = _search_routed(query, limit)

    meta['fetch_latency_ms'] = int((time.time() - t0) * 1000)

    # DDG auto-fallback if all providers returned empty
    if not results and 'ddg' not in used:
        results = fetch_ddg_fallback(query, limit)
        if results:
            used.append('ddg')
            meta['ddg_fallback'] = True

    logger.info('[SEARCH] query="%s" providers=%s count=%d', query, used, len(results))

    if not results:
        return {'text': 'No results found'}

    return {'text': json.dumps({'results': [
        {'title': r.get('title', ''), 'desc': r.get('snippet', ''), 'url': r.get('url', '')}
        for r in results
    ]})}


def _search_forced(query: str, provider_name: str, limit: int):
    if provider_name == 'ddg':
        return fetch_ddg_fallback(query, limit), ['ddg'], {'routing_method': 'forced'}

    providers = _load_providers()
    provider = providers.get(provider_name)
    if not provider:
        return [], [], {'routing_method': 'forced', 'error': f'Unknown: {provider_name}'}

    return fetch_providers([provider], query, limit), [provider_name], {'routing_method': 'forced'}


def _search_routed(query: str, limit: int):
    try:
        from tools.search.router import route_query
        ranked = route_query(query)
        if ranked:
            providers = _load_providers()
            provider_dicts = [providers[p['name']] for p in ranked if p['name'] in providers]
            results = fetch_providers(provider_dicts, query, limit)
            used = [p['name'] for p in provider_dicts]
            return results, used, {
                'routing_method': 'auto',
                'provider_scores': {p['name']: p['score'] for p in ranked},
            }
    except Exception as e:
        logger.warning('[SEARCH] router error, falling back to DDG: %s', e)

    return fetch_ddg_fallback(query, limit), ['ddg'], {'routing_method': 'fallback'}
