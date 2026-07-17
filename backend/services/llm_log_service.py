"""LlmLogService — per-call token/latency telemetry + usage aggregates (§4.2)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

from models.llm_call_log import LlmCallLog
from services.database import Database
from services.time_utils import utc_now

if TYPE_CHECKING:
    from typing import Callable

    from controllers.message_processor import MessageProcessor
    from models.provider_response import ProviderResponse

logger = logging.getLogger(__name__)

_WINDOW_OFFSETS = {
    'hour':     '-1 hour',
    'day':      '-1 day',
    'week':     '-7 days',
    'month':    '-30 days',
    'lifetime': None,
}

VALID_WINDOWS = frozenset(_WINDOW_OFFSETS)


class LlmLogService:
    """Persists one ``LlmCallLog`` row per provider call and reports token usage."""

    def __init__(self, mp: MessageProcessor) -> None:
        self.mp = mp

    def record(self, response: ProviderResponse) -> None:
        """Persist one ``LlmCallLog`` row for a completed provider call.

        Reached only from ``ProviderService.send`` — the single provider
        chokepoint — so what gets billed is decided by the call path, not by a
        flag on the request. Probe calls (candidate-provider checks in
        ``vision_service`` / ``provider_probe``) build their own client and
        never route through ``send``, which is exactly why they are absent from
        the ledger: spend the user did not incur is not spend.
        """
        try:
            LlmCallLog(
                type=self.mp.config.USAGE_TYPE,
                provider=response.provider or '',
                model=response.model,
                tokens_input=response.tokens_input or 0,
                tokens_output=response.tokens_output or 0,
                tokens_cache_read=response.tokens_cache_read or 0,
                tokens_cache_create=response.tokens_cache_create or 0,
                tokens_thinking=response.tokens_thinking or 0,
                latency_ms=response.latency_ms or 0,
            ).save()
        except Exception as e:
            logger.debug(f"[LLM_LOG] Failed to log call: {e}")

    @staticmethod
    def token_usage(window: str, usage_type: str | None = None) -> dict[str, object]:
        """Time-bucketed token usage statistics for ``window``, optionally filtered by type."""
        bucket_fmt = "%Y-%m-%dT%H:00:00" if window in ('hour', 'day') else "%Y-%m-%d"
        rows = LlmLogService._usage_buckets(bucket_fmt, _WINDOW_OFFSETS[window], usage_type)

        entries = [
            {
                'bucket': r['bucket'],
                'type': r['type'],
                'model': r['model'],
                'provider': r['provider'],
                'tokens_input': r['tokens_input'] or 0,
                'tokens_output': r['tokens_output'] or 0,
                'tokens_cache_read': r['tokens_cache_read'] or 0,
                'tokens_cache_create': r['tokens_cache_create'] or 0,
                'tokens_thinking': r['tokens_thinking'] or 0,
            }
            for r in rows
        ]

        return {
            'generated_at': utc_now().isoformat(),
            'window': window,
            'summary': LlmLogService._summarize(entries),
            'entries': entries,
        }

    @staticmethod
    def _usage_buckets(
        bucket_fmt: str,
        offset: str | None,
        usage_type: str | None,
    ) -> list[dict[str, object]]:
        """Token sums grouped by (bucket, type, model, provider).

        ``bucket_fmt`` is the ``strftime`` bucket width (by-hour or by-day),
        ``offset`` an optional ``datetime('now', ?)`` window bound (None =
        lifetime), ``usage_type`` an optional 'chat'/'system' filter.
        """
        where_clauses: list[str] = []
        params: list[object] = []
        if offset is not None:
            where_clauses.append("created_at >= datetime('now', ?)")
            params.append(offset)
        if usage_type:
            where_clauses.append("type = ?")
            params.append(usage_type)
        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        cursor = Database.conn().execute(
            f"""SELECT
                   strftime('{bucket_fmt}', created_at) AS bucket,
                   type,
                   model,
                   provider,
                   SUM(tokens_input)        AS tokens_input,
                   SUM(tokens_output)       AS tokens_output,
                   SUM(tokens_cache_read)   AS tokens_cache_read,
                   SUM(tokens_cache_create) AS tokens_cache_create,
                   SUM(tokens_thinking)     AS tokens_thinking
               FROM llm_call_log
               {where_sql}
               GROUP BY bucket, type, model, provider
               ORDER BY bucket DESC""",
            params,
        )
        return [dict(row) for row in cursor.fetchall()]

    @staticmethod
    def _tokens_today() -> int:
        """Total tokens (chat and system alike) logged since the start of the UTC day."""
        row = Database.conn().execute(
            """SELECT COALESCE(SUM(tokens_input + tokens_output +
                                  tokens_cache_read + tokens_cache_create +
                                  tokens_thinking), 0) AS total
               FROM llm_call_log
               WHERE created_at >= date('now')""",
        ).fetchone()
        return int(row["total"]) if row is not None else 0

    @staticmethod
    def _summarize(entries: list[dict[str, object]]) -> dict[str, object]:
        """Derive aggregate summary stats (totals, cache hit %, most-active model) from bucketed entries."""
        total_input = sum(cast(int, e['tokens_input']) for e in entries)
        total_output = sum(cast(int, e['tokens_output']) for e in entries)
        total_cache_read = sum(cast(int, e['tokens_cache_read']) for e in entries)
        total_cache_create = sum(cast(int, e['tokens_cache_create']) for e in entries)
        total_thinking = sum(cast(int, e['tokens_thinking']) for e in entries)
        total_tokens = total_input + total_output + total_cache_read + total_cache_create + total_thinking

        has_cache_data = (total_cache_read + total_cache_create) > 0
        prompt_total = total_input + total_cache_read
        cache_hit_pct: float | None
        if not has_cache_data:
            cache_hit_pct = None
        elif prompt_total > 0:
            cache_hit_pct = round(total_cache_read / prompt_total * 100, 1)
        else:
            cache_hit_pct = 0.0

        model_totals: dict[str, int] = {}
        for e in entries:
            m = cast(str, e.get('model') or 'unknown')
            model_totals[m] = model_totals.get(m, 0) + cast(int, e['tokens_input']) + cast(int, e['tokens_output'])
        most_active_model = max(model_totals, key=cast("Callable[[str], int]", model_totals.get)) if model_totals else None

        return {
            'total_tokens': total_tokens,
            'cache_hit_pct': cache_hit_pct,
            'tokens_today': LlmLogService._tokens_today(),
            'most_active_model': most_active_model,
        }
