"""LlmLogService — per-call token/latency telemetry + usage aggregates (§4.2)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

from models.llm_call_log import LlmCallLog
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
        """Persist one ``LlmCallLog`` row for a completed provider call."""
        try:
            LlmCallLog(
                job_name=self.mp.config.job,
                provider=response.provider or '',
                model=response.model,
                tokens_input=response.tokens_input or 0,
                tokens_output=response.tokens_output or 0,
                tokens_cache_read=response.tokens_cache_read or 0,
                tokens_cache_create=response.tokens_cache_create or 0,
                tokens_thinking=response.tokens_thinking or 0,
                latency_ms=response.latency_ms or 0,
                usage_class=self.mp.config.usage_class,
            ).save()
        except Exception as e:
            logger.debug(f"[LLM_LOG] Failed to log call: {e}")

    def last_request_tokens(self, job_name: str = 'user:user') -> int | None:
        """``tokens_input`` of the most recent call logged for ``job_name``."""
        return LlmCallLog.last_request_tokens(job_name)

    def token_usage(self, window: str, usage_class: str | None = None) -> dict[str, object]:
        """Time-bucketed token usage statistics for ``window``, optionally filtered by class."""
        bucket_fmt = "%Y-%m-%dT%H:00:00" if window in ('hour', 'day') else "%Y-%m-%d"
        rows = LlmCallLog.usage_buckets(bucket_fmt, _WINDOW_OFFSETS[window], usage_class)

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
            for r in rows
        ]

        return {
            'generated_at': utc_now().isoformat(),
            'window': window,
            'summary': self._summarize(entries),
            'entries': entries,
        }

    def _summarize(self, entries: list[dict[str, object]]) -> dict[str, object]:
        """Derive aggregate summary stats (totals, cache hit %, most-active model) from bucketed entries."""
        total_input = sum(cast(int, e['tokens_input']) for e in entries)
        total_output = sum(cast(int, e['tokens_output']) for e in entries)
        total_cache_read = sum(cast(int, e['tokens_cache_read']) for e in entries)
        total_cache_create = sum(cast(int, e['tokens_cache_create']) for e in entries)
        total_thinking = sum(cast(int, e['tokens_thinking']) for e in entries)
        total_tokens = total_input + total_output + total_cache_read + total_cache_create + total_thinking

        has_cache_data = (total_cache_read + total_cache_create) > 0
        prompt_total = total_input + total_cache_read
        cache_hit_pct = (
            round(total_cache_read / prompt_total * 100, 1)
            if has_cache_data and prompt_total > 0
            else None if not has_cache_data else 0.0
        )

        model_totals: dict[str, int] = {}
        for e in entries:
            m = cast(str, e.get('model') or 'unknown')
            model_totals[m] = model_totals.get(m, 0) + cast(int, e['tokens_input']) + cast(int, e['tokens_output'])
        most_active_model = max(model_totals, key=cast("Callable[[str], int]", model_totals.get)) if model_totals else None

        return {
            'total_tokens': total_tokens,
            'cache_hit_pct': cache_hit_pct,
            'tokens_today': LlmCallLog.tokens_today(),
            'most_active_model': most_active_model,
        }
