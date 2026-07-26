"""ProviderService — the provider-communication coordinating service (§4.2).

Resolves the per-turn provider selection, builds and holds the thin transport
client, and sends one :class:`~models.provider_request.ProviderRequest`
through the pre-flight cap chokepoint, then logs the call. Retry is NOT this
layer's job (§6.4): a size fault raises ``ContextLimit`` — measured pre-flight
here, or reported by the provider and re-raised with this turn attached — which
is the MessageProcessor's cue to compact-then-retry;
any other provider failure bubbles straight up for the MP's own resend policy
to catch. The two transient notices this layer DOES own — neither a turn state,
both emitted through ``mp.push_websocket``, which is their only broadcast gate —
are ``provider_retry`` (a toast raised while the MP's retry loop is still in
flight) and ``context_usage`` (how full each CHAT request was against its
window). Both belong here because this layer already owns that judgement: it
computes the window and the measured size to enforce the cap, so the meter is
the same reading reported instead of enforced.

Owns the thin ``llm_clients/*`` (constructed and held here; transport-only,
no ``mp``) and the per-turn provider *selection* reads (main/vision/delegate
resolution). Admin provider-config write CRUD stays off-spine — leaf admin
(``services/provider_db_service.py``, ``services/provider_cache_service.py``)
is read-only reached from here, unmodified.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, cast

from configs.enums.provider_type import ProviderType
from exceptions import ContextLimit, ProviderError
from models.turn_signal import TurnSignal
from services.llm_clients.factory import build_client
from services.provider_api import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_CONTEXT_WINDOWS,
    MAX_CONTEXT_WINDOW,
)

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

logger = logging.getLogger(__name__)


def _persist_window(config: dict[str, object], window: int) -> None:
    """Write a freshly probed window onto the provider row.

    Without this, a row that predates window-seeding would make every single
    send re-probe the host: ``send()`` builds a fresh client per call, so the
    client's own per-instance cache dies with it. Persisting converges the row
    on first use, which is why no migration backfill is needed.

    The cache bust is not optional — provider rows are served from
    ``ProviderCacheService``, and invalidation lives at the API layer
    (``api/endpoints/providers.py``), not in ``ProviderDbService``. Skipping it
    would leave the stale ``context_window: None`` in the cached dict and the
    probe would repeat on every send anyway.

    Best-effort by design: a failure here costs one repeated probe, never the turn.
    """
    provider_id = config.get("id")
    if not isinstance(provider_id, int):
        return
    try:
        from services.provider_db_service import ProviderDbService  # noqa: PLC0415
        from services.provider_probe import invalidate_provider_cache  # noqa: PLC0415
        ProviderDbService().update_provider(provider_id, {"context_window": window})
        invalidate_provider_cache()
        config["context_window"] = window
        logger.info(
            "Stored context window %d for provider id=%s (%s)",
            window, provider_id, config.get("model") or "",
        )
    except Exception as exc:
        logger.warning(
            "Could not persist context window for provider id=%s: %s", provider_id, exc,
        )


def _window_of(client: "ProviderClient", config: dict[str, object]) -> int:
    """The one context-window computation, hard-capped at MAX_CONTEXT_WINDOW.

    Prefers the window stored on the provider row. Only when that is unset does
    it fall back to the client's own live answer. When the client also cannot
    tell (no host, transient probe failure, model not advertised), it drops to
    the platform's documented operating default and logs a WARNING naming
    provider+model. Never returns None.

    The last step deliberately does NOT use MAX_CONTEXT_WINDOW: that is the most
    permissive value there is, so a failed probe against a small model would
    size compaction against 200k and let every turn run until the provider hard-
    rejects it. An unknown window must degrade conservatively, not optimistically.
    """
    provider = cast(str, config.get("platform") or "")
    stored = config.get("context_window")
    limit: int | None
    if isinstance(stored, int) and stored > 0:
        limit = stored
    else:
        limit = client.get_context_limit()
        if limit is not None:
            _persist_window(config, min(limit, MAX_CONTEXT_WINDOW))
    if limit is None:
        limit = DEFAULT_CONTEXT_WINDOWS.get(provider, DEFAULT_CONTEXT_WINDOW)
        logger.warning(
            "No context window available for provider=%s model=%s; using the "
            "platform default of %d tokens",
            provider, cast(str, config.get("model") or ""), limit,
        )
    return min(limit, MAX_CONTEXT_WINDOW)


class ProviderService:
    """Sends provider requests, resolves selection/limits, sanitises tool args."""

    def __init__(self, mp: MessageProcessor) -> None:
        self.mp = mp

    def send(self, request: ProviderRequest) -> ProviderResponse:
        """Pre-flight cap check, call, log, report usage — the single provider
        chokepoint. A CHAT call that came back with a token count emits
        ``context_usage``: this is the one place both halves of the meter's
        fraction exist together (``window`` above, ``tokens_input`` below), and
        one call is exactly one move of it. A provider that reports no count is
        silent rather than zero — the surface hides a meter it has no reading
        for, and a fabricated 0 would read as a real, empty context."""
        config = self._select(request.type)
        client = build_client(config)
        window = _window_of(client, config)
        cap = int(0.90 * window)
        measured = client.estimate_request_tokens(cast("ProviderApiRequest", request))
        if measured >= cap:
            raise ContextLimit(
                f"Request ({measured} tokens) reached 90% of the {window}-token context window",
                self.mp, window=window, measured=measured,
                provider=cast(str, config.get("platform") or ""),
                model=cast(str, config.get("model") or ""),
            )
        try:
            response = cast("ProviderResponse", client.send(cast("ProviderApiRequest", request)))
        except ContextLimit as limit:
            # The client knows the provider said "too long"; only this layer
            # knows whose turn it was. Attach it so the handler can compact.
            limit.mp = self.mp
            limit.window = limit.window or window
            raise
        self.mp.llm_log_service.record(response)
        if request.type is ProviderType.CHAT and response.tokens_input is not None:
            self.mp.push_websocket(
                TurnSignal.context_usage(self.mp, response.tokens_input, window),
            )
        return response

    def measure(self, request: ProviderRequest) -> int:
        """Estimated token cost of ``request`` without sending (compaction sizing)."""
        return self._resolve(request.type).estimate_request_tokens(cast("ProviderApiRequest", request))

    def context_limit(self, provider_type: ProviderType = ProviderType.CHAT) -> int:
        """Context window for ``provider_type`` — see :func:`_window_of`."""
        config = self._select(provider_type)
        return _window_of(build_client(config), config)

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
