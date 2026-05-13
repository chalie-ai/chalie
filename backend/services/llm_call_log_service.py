"""LLM Call Log Service — persistent per-call token usage tracking."""

import logging
from services.database_service import get_shared_db_service
from services.time_utils import utc_now

logger = logging.getLogger(__name__)

_WINDOW_OFFSETS = {
    'hour':     '-1 hour',
    'day':      '-1 day',
    'week':     '-7 days',
    'month':    '-30 days',
    'lifetime': None,
}

VALID_WINDOWS = frozenset(_WINDOW_OFFSETS)


def log_call(
    job_name: str,
    provider: str,
    model: str,
    tokens_input: int,
    tokens_output: int,
    latency_ms: int,
    tokens_cache_read: int = 0,
    tokens_cache_create: int = 0,
    tokens_thinking: int = 0,
    usage_class: str = 'chat',
):
    """Record a single LLM call with full token counts and latency."""
    try:
        db = get_shared_db_service()
        db.execute(
            """INSERT INTO llm_call_log
               (job_name, provider, model,
                tokens_input, tokens_output,
                tokens_cache_read, tokens_cache_create, tokens_thinking,
                latency_ms, usage_class)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                job_name, provider, model,
                tokens_input or 0, tokens_output or 0,
                tokens_cache_read or 0, tokens_cache_create or 0,
                tokens_thinking or 0,
                latency_ms or 0,
                usage_class or 'chat',
            ),
        )
    except Exception as e:
        logger.debug(f"[LLM_CALL_LOG] Failed to log call: {e}")


def get_token_usage(window: str, usage_class: str | None = None) -> dict:
    """Return time-bucketed token usage statistics.

    Args:
        window: One of 'hour', 'day', 'week', 'month', 'lifetime'.
        usage_class: Optional filter — 'chat', 'subagent', or 'subconscious'.

    Returns:
        dict with 'window', 'generated_at', 'summary', and 'entries' keys.
        entries: list of dicts with bucket, tokens_input, tokens_output,
                 tokens_cache_read, tokens_cache_create, tokens_thinking,
                 usage_class, model, provider.
        summary: total_tokens, cache_hit_pct, tokens_today, most_active_model.
    """
    db = get_shared_db_service()
    offset = _WINDOW_OFFSETS[window]

    # Bucket width: hour/day → by-hour buckets; week/month/lifetime → by-day.
    bucket_fmt = "%Y-%m-%dT%H:00:00" if window in ('hour', 'day') else "%Y-%m-%d"

    where_clauses = []
    params: list = []

    if offset is not None:
        where_clauses.append("created_at >= datetime('now', ?)")
        params.append(offset)

    if usage_class:
        where_clauses.append("usage_class = ?")
        params.append(usage_class)

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    rows = db.fetch_all(
        f"""SELECT
               strftime('{bucket_fmt}', created_at) AS bucket,
               usage_class,
               model,
               provider,
               SUM(tokens_input)        AS tokens_input,
               SUM(tokens_output)       AS tokens_output,
               SUM(tokens_cache_read)   AS tokens_cache_read,
               SUM(tokens_cache_create) AS tokens_cache_create,
               SUM(tokens_thinking)     AS tokens_thinking
           FROM llm_call_log
           {where_sql}
           GROUP BY bucket, usage_class, model, provider
           ORDER BY bucket DESC""",
        params,
    )

    entries = [
        {
            'bucket': r['bucket'],
            'usage_class': r['usage_class'],
            'model': r['model'],
            'provider': r['provider'],
            'tokens_input': r['tokens_input'] or 0,
            'tokens_output': r['tokens_output'] or 0,
            'tokens_cache_read': r['tokens_cache_read'] or 0,
            'tokens_cache_create': r['tokens_cache_create'] or 0,
            'tokens_thinking': r['tokens_thinking'] or 0,
        }
        for r in (rows or [])
    ]

    summary = _compute_summary(entries)

    return {
        'generated_at': utc_now().isoformat(),
        'window': window,
        'summary': summary,
        'entries': entries,
    }


def _compute_summary(entries: list) -> dict:
    """Derive aggregate summary stats from bucketed entries."""
    total_input = sum(e['tokens_input'] for e in entries)
    total_output = sum(e['tokens_output'] for e in entries)
    total_cache_read = sum(e['tokens_cache_read'] for e in entries)
    total_cache_create = sum(e['tokens_cache_create'] for e in entries)
    total_thinking = sum(e['tokens_thinking'] for e in entries)
    total_tokens = total_input + total_output + total_cache_read + total_cache_create + total_thinking

    # Cache hit %: cache_read / (cache_read + tokens_input) when both > 0.
    # In Anthropic context: tokens_input = uncached input, cache_read = cached portion.
    prompt_total = total_input + total_cache_read
    cache_hit_pct = round(total_cache_read / prompt_total * 100, 1) if prompt_total > 0 else 0.0

    # Tokens today — re-query the DB for the current UTC day.
    try:
        db = get_shared_db_service()
        today_rows = db.fetch_all(
            """SELECT COALESCE(SUM(tokens_input + tokens_output +
                                  tokens_cache_read + tokens_cache_create +
                                  tokens_thinking), 0) AS total
               FROM llm_call_log
               WHERE created_at >= date('now')""",
        )
        tokens_today = today_rows[0]['total'] if today_rows else 0
    except Exception:
        tokens_today = 0

    # Most active model by total call count across all time.
    try:
        db = get_shared_db_service()
        model_rows = db.fetch_all(
            """SELECT model, COUNT(*) AS cnt
               FROM llm_call_log
               GROUP BY model
               ORDER BY cnt DESC
               LIMIT 1"""
        )
        most_active_model = model_rows[0]['model'] if model_rows else None
    except Exception:
        most_active_model = None

    return {
        'total_tokens': total_tokens,
        'cache_hit_pct': cache_hit_pct,
        'tokens_today': tokens_today,
        'most_active_model': most_active_model,
    }
