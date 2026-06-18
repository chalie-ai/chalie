"""Providers — standalone orchestrator facade for all provider API communication."""

import json
import logging
import threading
import time
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from services.llm_clients.base import ProviderClient
    from services.provider_api import ProviderApiRequest, ProviderApiResponse, ProviderType

logger = logging.getLogger(__name__)

# First-failure-warning flags (mirrors llm_request_logger._warned_on_failure).
# Disk-full / DB-lock errors are surfaced at WARNING once per process; all
# subsequent failures drop to DEBUG to avoid log spam.
_log_call_warned = False
_log_call_warn_lock = threading.Lock()

# Hard ceiling on any provider's reported context window.
MAX_CONTEXT_WINDOW = 200_000


def resolve_thinking_mode(
    config_thinking_mode: str | None,
    override: str | None,
    level: str | None,
) -> str | None:
    """Single precedence rule for a send's thinking level.

    1. config_thinking_mode — a config that hard-pins a level (thinking ability
       and both compactors pin "high"). Wins over everything.
    2. override — the persisted user override ('medium' | 'high' | None).
    3. level — the gate-computed mp.thinking_level (auto behaviour).
    """
    return config_thinking_mode or override or level


class Providers:
    """Standalone orchestrator facade — no mp, no scaffolding.

    Each caller builds a ProviderApiRequest from its own state and calls
    send(dto). Provider resolution is driven entirely by dto.type.
    """

    # ── Resolution ──────────────────────────────────────────────────────────

    def _resolve(self, provider_type: object = None) -> "ProviderClient":
        """Return the ProviderClient for the given ProviderType.

        CHAT → globally selected provider (ProviderCacheService).
        VISION → DB vision provider (ProviderDbService).
        DELEGATE → DB delegate provider (ProviderDbService); falls back to the
        selected (main) provider when none is pinned.
        Any other type (e.g. the reserved VISUAL_OUTPUT) raises ProviderError
        rather than silently resolving to the CHAT provider.
        """
        from services.provider_api import ProviderType, ProviderError  # noqa: PLC0415
        from services.llm_clients.factory import build_client  # noqa: PLC0415

        pt = provider_type or ProviderType.CHAT

        if pt == ProviderType.VISION:
            from services.provider_db_service import ProviderDbService  # noqa: PLC0415
            from services.database_service import get_shared_db_service  # noqa: PLC0415
            vp = ProviderDbService(get_shared_db_service()).get_vision_provider()
            if not vp:
                raise RuntimeError("VISION type requested but no vision provider configured")
            return build_client(dict(vp))

        if pt == ProviderType.DELEGATE:
            from services.provider_db_service import ProviderDbService  # noqa: PLC0415
            from services.database_service import get_shared_db_service  # noqa: PLC0415
            dp = ProviderDbService(get_shared_db_service()).get_delegate_provider()
            if not dp:
                raise RuntimeError("DELEGATE type requested but no delegate or selected provider configured")
            return build_client(dict(dp))

        if pt != ProviderType.CHAT:
            raise ProviderError(f"Unsupported provider type for send: {pt}")

        from services.provider_cache_service import ProviderCacheService  # noqa: PLC0415
        config = ProviderCacheService.get_selected_provider()
        if not config:
            providers = ProviderCacheService.get_providers()
            config = dict(next(iter(providers.values()))) if providers else {}
        return build_client(dict(config))

    # ── Public API ──────────────────────────────────────────────────────────

    def send(self, dto: "ProviderApiRequest") -> "ProviderApiResponse":
        """Pre-flight check, call, telemetry — the single chokepoint.

        Steps:
          1. Resolve client by dto.type.
          2. Get context window (capped at MAX_CONTEXT_WINDOW).
          3. Pre-flight: estimate tokens; raise RequestOverCapError if over cap.
          4. client.send(dto) — may raise ResponseOverLimitError / ProviderResponseError.
          5. Fold telemetry.
          6. Return ProviderApiResponse.
        """
        from services.provider_api import RequestOverCapError  # noqa: PLC0415

        client = self._resolve(dto.type)
        window = min(client.get_context_limit(), MAX_CONTEXT_WINDOW)
        cap = window - max(int(0.10 * window), 8000)

        measured = client.estimate_request_tokens(dto)
        if measured >= cap:
            raise RequestOverCapError(
                f"Request ({measured} tokens) exceeds cap ({cap}); window={window}",
                window=window,
                measured=measured,
                cap=cap,
                provider=getattr(client, 'provider', ''),
                model=getattr(client, 'model', ''),
            )

        t0 = time.monotonic()
        response = client.send(dto)
        wall_ms = int((time.monotonic() - t0) * 1000)
        self._log_after_call(dto, response, wall_ms)
        return response

    def measure(self, dto: "ProviderApiRequest") -> int:
        """Return the estimated token cost of dto without sending.

        Used by ChatHistoryCompactor to size a candidate request before
        deciding whether to compact further.
        """
        client = self._resolve(dto.type)
        return client.estimate_request_tokens(dto)

    def get_context_limit(self, provider_type: "ProviderType | None" = None) -> int:
        """Declared context window for the active provider/model, hard-capped at MAX_CONTEXT_WINDOW.

        Reads the backfilled providers.max_tokens first (fast path; set by
        provider_token_limits.backfill_one). Falls back to the live client
        method if the column is unset (before first backfill).
        """
        from services.provider_api import ProviderType  # noqa: PLC0415
        pt = provider_type or ProviderType.CHAT

        if pt == ProviderType.CHAT:
            from services.provider_cache_service import ProviderCacheService  # noqa: PLC0415
            config = ProviderCacheService.get_selected_provider() or {}
            declared = config.get("max_tokens")
            if declared and int(cast(int, declared)) > 0:
                return min(int(cast(int, declared)), MAX_CONTEXT_WINDOW)

        return min(self._resolve(pt).get_context_limit(), MAX_CONTEXT_WINDOW)

    def selected_provider(self) -> "ProviderClient":
        """Return the resolved CHAT provider client.

        Consumed by configs/channels/_common.py:15 for CONTENT_FIELD_LABEL
        system-prompt placeholder substitution. Preserved per design resolution #3.
        """
        return self._resolve()

    # ── Telemetry ────────────────────────────────────────────────────────────

    def _log_after_call(self, dto: "ProviderApiRequest", response: "ProviderApiResponse", wall_ms: int | None = None) -> None:
        """Write the LLM request log file and persist per-call token accounting.

        Single logging chokepoint every LLM request flows through.
        Absorbs LoggingLLMService (deleted) — including log_call (resolution #4).
        """
        # Metrics (record_llm_call, accumulate) are the CALLER's responsibility —
        # the mp calls _record_metrics() after send() returns. For standalone
        # calls (test-connection, vision-probe) no accumulator is available.

        # Per-call token accounting (was LoggingLLMService._log_llm_call).
        # job_name is carried on the DTO via the _job_name metadata key (set
        # by MessageProcessor.send() and the probe callers).
        job_name = getattr(dto, '_job_name', None) or ''
        usage_class = getattr(dto, '_usage_class', None) or 'chat'
        if job_name:
            try:
                from services.llm_call_log_service import log_call  # noqa: PLC0415
                log_call(
                    job_name=job_name,
                    provider=getattr(response, 'provider', 'unknown'),
                    model=getattr(response, 'model', 'unknown'),
                    tokens_input=getattr(response, 'tokens_input', 0) or 0,
                    tokens_output=getattr(response, 'tokens_output', 0) or 0,
                    tokens_cache_read=getattr(response, 'tokens_cache_read', 0) or 0,
                    tokens_cache_create=getattr(response, 'tokens_cache_create', 0) or 0,
                    tokens_thinking=getattr(response, 'tokens_thinking', 0) or 0,
                    latency_ms=getattr(response, 'latency_ms', 0) or 0,
                    usage_class=usage_class,
                )
            except Exception as exc:
                global _log_call_warned  # noqa: PLW0603
                with _log_call_warn_lock:
                    first = not _log_call_warned
                    _log_call_warned = True
                if first:
                    logger.warning("[LLM LOG] log_call failed (non-fatal): %s", exc)
                else:
                    logger.debug("[LLM LOG] log_call failed (non-fatal): %s", exc)

        # File-based request log (verbatim prompt/response log).
        try:
            from services.llm_request_logger import log_llm_request  # noqa: PLC0415
            log_llm_request(
                caller=getattr(dto, '_caller', 'Providers'),
                job=job_name,
                provider=getattr(response, 'provider', 'unknown'),
                model=getattr(response, 'model', 'unknown'),
                system_message=dto.system or '',
                user_message=self._render_messages_for_log(dto.messages),
                tools=cast("list[object] | None", dto.tools),
            )
        except Exception as exc:
            logger.debug("[LLM LOG] log_llm_request failed: %s", exc, exc_info=True)

    @staticmethod
    def _record_send_counters(proc: object) -> None:
        """Record per-send, per-channel turn counters (spec §4e).

        Called by MessageProcessor._record_metrics after every send.
        Every send increments requests_total; the channel-specific turn counter
        is selected from the bound processor's config.channel.
        """
        try:
            channel = getattr(getattr(proc, 'config', None), 'channel', '') or ''
            from services.metrics_service import MetricsService  # noqa: PLC0415
            m = MetricsService()
            m.record_counter('requests_total')
            if channel == 'dmn':
                m.record_counter('dmn_turns_total')
            elif channel.startswith('delegate:'):
                m.record_counter('subagent_turns_total')
        except Exception as exc:
            logger.debug("[LLM LOG] send-counter record failed: %s", exc)

    @staticmethod
    def _render_messages_for_log(messages: list[dict[str, object]]) -> str:
        """Render the messages array verbatim for a log file."""
        msgs: list[dict[str, object]] = messages or []

        def _content_to_str(content: object) -> str:
            if isinstance(content, str):
                return content
            return json.dumps(content, ensure_ascii=False)

        if len(msgs) == 1 and msgs[0].get('role') == 'user':
            return _content_to_str(msgs[0].get('content', ''))
        return '\n\n'.join(
            f"[{m.get('role', '?')}] {_content_to_str(m.get('content', ''))}"
            for m in msgs
        )
