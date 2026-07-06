"""ProviderService — the provider-communication coordinating service (§4.2).

Resolves the per-turn provider selection, builds and holds the thin transport
client, and sends one :class:`~models.provider_request.ProviderRequest`
through the pre-flight cap chokepoint, then logs the call. Retry is NOT this
layer's job (§6.4): a size fault raises ``RequestOverCapError`` /
``ResponseOverLimitError`` (the MessageProcessor's cue to compact-then-retry);
any other provider failure bubbles straight up for the MP's own resend policy
to catch. The one mid-resend notice this layer DOES own is ``provider_retry``
— a transient toast, not a turn state — emitted through ``mp.push_websocket`` while
the MP's retry loop is still in flight.

Owns the thin ``llm_clients/*`` (constructed and held here; transport-only,
no ``mp``) and the per-turn provider *selection* reads (main/vision/delegate
resolution). Admin provider-config write CRUD stays off-spine — leaf admin
(``services/provider_db_service.py``, ``services/provider_cache_service.py``)
is read-only reached from here, unmodified.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, cast

from configs.enums.provider_type import ProviderType
from models.provider_errors import ProviderError, RequestOverCapError
from models.turn_signal import TurnSignal
from services.llm_clients.factory import build_client
from services.provider_api import MAX_CONTEXT_WINDOW

if TYPE_CHECKING:
    from controllers.message_processor import MessageProcessor
    from models.provider_request import ProviderRequest
    from models.provider_response import ProviderResponse
    from contracts.provider_client import ProviderClient
    from services.provider_api import ProviderApiRequest

# Leaked provider sentinel tokens (``<|...|>``) stripped from tool args.
_LLM_SENTINEL_PATTERNS = (
    re.compile(r'<\|[^|<>]*\|>'),
    re.compile(r'<\|[^|<>]*\|'),
)


class ProviderService:
    """Sends provider requests, resolves selection/limits, sanitises tool args."""

    def __init__(self, mp: MessageProcessor) -> None:
        self.mp = mp

    def send(self, request: ProviderRequest) -> ProviderResponse:
        """Pre-flight cap check, call, log — the single provider chokepoint."""
        config = self._select(request.type)
        client = build_client(config)
        window = min(client.get_context_limit(), MAX_CONTEXT_WINDOW)
        cap = window - max(int(0.10 * window), 8000)
        measured = client.estimate_request_tokens(cast("ProviderApiRequest", request))
        if measured >= cap:
            raise RequestOverCapError(
                f"Request ({measured} tokens) exceeds cap ({cap}); window={window}",
                window=window, measured=measured, cap=cap,
                provider=cast(str, config.get("platform") or ""),
                model=cast(str, config.get("model") or ""),
            )
        response = cast("ProviderResponse", client.send(cast("ProviderApiRequest", request)))
        self.mp.llm_log_service.record(response)
        return response

    def measure(self, request: ProviderRequest) -> int:
        """Estimated token cost of ``request`` without sending (compaction sizing)."""
        return self._resolve(request.type).estimate_request_tokens(cast("ProviderApiRequest", request))

    def context_limit(self, provider_type: ProviderType = ProviderType.CHAT) -> int:
        """Declared context window for ``provider_type``, hard-capped at MAX_CONTEXT_WINDOW."""
        if provider_type is ProviderType.CHAT:
            declared = self._select(ProviderType.CHAT).get("max_tokens")
            if declared and int(cast("int | str", declared)) > 0:
                return min(int(cast("int | str", declared)), MAX_CONTEXT_WINDOW)
        return min(self._resolve(provider_type).get_context_limit(), MAX_CONTEXT_WINDOW)

    def selected_provider(self) -> ProviderClient:
        """The resolved CHAT provider client (e.g. for prompt-template metadata)."""
        return self._resolve(ProviderType.CHAT)

    def resolve_thinking_mode(self) -> str | None:
        """Precedence: a config-pinned mode > the user's override > the turn's gated level."""
        return cast(
            "str | None",
            self.mp.config.thinking_mode or self.mp.thinking_override or self.mp.thinking_level,
        )

    def sanitize_args(self, value: object) -> object:
        """Strip leaked provider sentinel tokens (``<|...|>``) from tool args, recursively."""
        if isinstance(value, str):
            for pattern in _LLM_SENTINEL_PATTERNS:
                value = pattern.sub("", value)
            return value.strip()
        if isinstance(value, list):
            return [self.sanitize_args(v) for v in value]
        if isinstance(value, dict):
            return {k: self.sanitize_args(v) for k, v in value.items()}
        return value

    def emit_retry(self, attempt: int, max_attempts: int, message: str) -> None:
        """Toast that an upstream resend is underway (§6.4 — no retry, one notice).
        The broadcast gate lives in ``mp.push_websocket``."""
        signal = TurnSignal.provider_retry(
            self.mp.turn_id, self.mp.config.type_value(), attempt, max_attempts, message,
        )
        self.mp.push_websocket(signal)

    def _select(self, provider_type: ProviderType) -> dict[str, object]:
        """Resolve the provider config dict for ``provider_type`` — the one per-turn selection read."""
        from services.provider_db_service import ProviderDbService  # noqa: PLC0415

        if provider_type is ProviderType.VISION:
            vision = ProviderDbService().get_vision_provider()
            if not vision:
                raise RuntimeError("VISION type requested but no vision provider configured")
            return dict(vision)
        if provider_type is ProviderType.DELEGATE:
            delegate = ProviderDbService().get_delegate_provider()
            if not delegate:
                raise RuntimeError("DELEGATE type requested but no delegate or selected provider configured")
            return dict(delegate)
        if provider_type is not ProviderType.CHAT:
            raise ProviderError(f"Unsupported provider type for send: {provider_type}")

        from services.provider_cache_service import ProviderCacheService  # noqa: PLC0415
        selected = ProviderCacheService.get_selected_provider()
        if selected:
            return dict(selected)
        providers = ProviderCacheService.get_providers()
        return dict(next(iter(providers.values()))) if providers else {}

    def _resolve(self, provider_type: ProviderType) -> ProviderClient:
        """Build the thin transport client for ``provider_type``'s selected config."""
        return build_client(self._select(provider_type))
