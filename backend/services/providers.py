"""Providers — standalone orchestrator facade for all provider API communication."""

import json
import logging
import threading
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from services.llm_clients.base import ProviderClient
    from services.provider_api import ProviderApiRequest, ProviderApiResponse

logger = logging.getLogger(__name__)

# First-failure-warning flags (mirrors llm_request_logger._warned_on_failure).
# Disk-full / DB-lock errors are surfaced at WARNING once per process; all
# subsequent failures drop to DEBUG to avoid log spam.
_log_call_warned = False
_log_call_warn_lock = threading.Lock()

# Hard ceiling on any provider's reported context window.
MAX_CONTEXT_WINDOW = 200_000

# Deadline (seconds) for a single provider API call, enforced by every thin
# client at its own HTTP boundary (the only place a call can actually be
# interrupted — a Python thread cannot be killed). On expiry the client raises
# ProviderTimeoutError, which the retry helper surfaces immediately. This is the
# ONE provider-call timeout; clients import this constant rather than hard-coding.
PROVIDER_CALL_TIMEOUT_S = 300


def resolve_thinking_mode(config_thinking_mode: str | None, override: str | None, level: str) -> str | None:
    """Single precedence rule for a send's thinking level."""
    return config_thinking_mode or override or level


class Providers:
    """Standalone orchestrator facade — no mp, no scaffolding."""

    # ── Resolution ──────────────────────────────────────────────────────────

    def _resolve(self, provider_type: object = None) -> "ProviderClient":
        """Return the ProviderClient for the given ProviderType."""
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
        """Pre-flight check, call, telemetry — the single chokepoint."""
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

        response = client.send(dto)
        self._log_after_call(dto, response)
        return response

    def measure(self, dto: "ProviderApiRequest") -> int:
        """Return the estimated token cost of dto without sending."""
        client = self._resolve(dto.type)
        return client.estimate_request_tokens(dto)

    def get_context_limit(self, provider_type: object = None) -> int:
        """Declared context window for the active provider/model, hard-capped at MAX_CONTEXT_WINDOW."""
        from services.provider_api import ProviderType  # noqa: PLC0415
        pt = provider_type or ProviderType.CHAT

        if pt == ProviderType.CHAT:
            from services.provider_cache_service import ProviderCacheService  # noqa: PLC0415
            config = ProviderCacheService.get_selected_provider() or {}
            declared = config.get("max_tokens")
            if declared and int(cast("int | str", declared)) > 0:
                return min(int(cast("int | str", declared)), MAX_CONTEXT_WINDOW)

        return min(self._resolve(pt).get_context_limit(), MAX_CONTEXT_WINDOW)

    def selected_provider(self) -> "ProviderClient":
        """Return the resolved CHAT provider client."""
        return self._resolve()

    # ── Telemetry ────────────────────────────────────────────────────────────

    def _log_after_call(self, dto: "ProviderApiRequest", response: "ProviderApiResponse") -> None:
        """Write the LLM request log file and persist per-call token accounting."""
        # Per-call token accounting (was LoggingLLMService._log_llm_call).
        # job_name is carried on the DTO via the _job_name metadata key (set
        # by MessageProcessor.send() and the probe callers).
        job_name = getattr(dto, '_job_name', None) or ''
        if job_name:
            self._log_token_accounting(dto, response, job_name)

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

    def _log_token_accounting(self, dto: "ProviderApiRequest", response: "ProviderApiResponse", job_name: str) -> None:
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
                usage_class=getattr(dto, '_usage_class', None) or 'chat',
            )
        except Exception as exc:
            global _log_call_warned  # noqa: PLW0603
            with _log_call_warn_lock:
                first = not _log_call_warned
                _log_call_warned = True
            logger.warning("[LLM LOG] log_call failed (non-fatal): %s", exc) if first else logger.debug("[LLM LOG] log_call failed (non-fatal): %s", exc)

    @staticmethod
    def _record_send_counters(proc: object) -> None:
        """Record per-send, per-channel turn counters (spec §4e)."""
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
    def _render_messages_for_log(messages: list[dict[str, object]] | None) -> str:
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
