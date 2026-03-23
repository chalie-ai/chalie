"""LLM Call Log Service — persistent per-call token usage tracking."""

import logging
from services.database_service import get_shared_db_service

logger = logging.getLogger(__name__)


def log_call(job_name: str, provider: str, model: str,
             tokens_input: int, tokens_output: int, latency_ms: int):
    """Record a single LLM call with token counts and latency."""
    try:
        db = get_shared_db_service()
        db.execute(
            """INSERT INTO llm_call_log
               (job_name, provider, model, tokens_input, tokens_output, latency_ms)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (job_name, provider, model,
             tokens_input or 0, tokens_output or 0, latency_ms or 0),
        )
    except Exception as e:
        logger.debug(f"[LLM_CALL_LOG] Failed to log call: {e}")


def get_stats(job_name: str, hours: int = 24) -> dict:
    """Return usage statistics for a job within the given time window.

    Args:
        job_name: The cognitive job identifier.
        hours: Look-back window in hours (1, 24, 168, 720).

    Returns:
        dict with 'summary' (totals) and 'calls' (individual records).
    """
    db = get_shared_db_service()
    cutoff = f"-{hours} hours"

    rows = db.fetch_all(
        """SELECT provider, model, tokens_input, tokens_output, latency_ms, created_at
           FROM llm_call_log
           WHERE job_name = ? AND created_at > datetime('now', ?)
           ORDER BY created_at DESC
           LIMIT 500""",
        (job_name, cutoff),
    )

    calls = [
        {
            'provider': r['provider'],
            'model': r['model'],
            'tokens_input': r['tokens_input'],
            'tokens_output': r['tokens_output'],
            'latency_ms': r['latency_ms'],
            'created_at': r['created_at'],
        }
        for r in rows
    ]

    total_input = sum(c['tokens_input'] for c in calls)
    total_output = sum(c['tokens_output'] for c in calls)
    total_latency = sum(c['latency_ms'] for c in calls)
    count = len(calls)

    return {
        'job_name': job_name,
        'hours': hours,
        'summary': {
            'total_calls': count,
            'total_tokens_input': total_input,
            'total_tokens_output': total_output,
            'avg_latency_ms': round(total_latency / count) if count else 0,
        },
        'calls': calls,
    }
