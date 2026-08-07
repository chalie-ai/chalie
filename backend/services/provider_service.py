"""ProviderService — the provider-communication coordinating service (§4.2).

Resolves the per-turn provider selection, builds and holds the thin transport
client, and sends one :class:`~models.provider_request.ProviderRequest`
through the cap chokepoint, then logs the call. Retry is NOT this layer's job
(§6.4): a size fault raises ``ContextLimit`` — either because the PREVIOUS
request on this channel came back reported at the cap, or because the provider
rejected this one and the client raised — which is the MessageProcessor's cue to
compact-then-retry; any other provider failure bubbles straight up for the MP's
own resend policy to catch.

Nothing here estimates or counts anything. The only size signal is what the
provider itself reported — its whole prompt, cached slices included (see
``ProviderResponse.context_tokens``) — which this layer wrote to
``llm_call_log`` on the previous send. The gate is therefore *reactive*: it
learns a channel is full from the answer, never from inspecting the question. It
also disarms after firing once, because the compaction that follows makes that
ledger reading stale and a re-read would fire forever.

The two transient notices this layer DOES own — neither a turn state, both
emitted through ``mp.push_websocket``, which is their only broadcast gate — are
``provider_retry`` (a toast raised while the MP's retry loop is still in flight)
and ``context_usage`` (how full each CHAT request was against its window). Both
belong here because this layer already holds both halves of that fraction: the
window it enforces against, and the count the provider reported.

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
from services.llm_clients.factory import Factory
from services.llm_log_service import LlmLogService

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


def _window_of(config: dict[str, object]) -> int:
    """The context window for ``config`` — always ``providers.context_window``.

    A thin pass-through to :meth:`ProviderDbService.pin_context_window`, which
    self-heals the column (probe once, clamp, persist) and raises when the
    window cannot be determined. There is deliberately no fallback here: this
    layer is a reader of that column, never a second opinion on it.
    """
    from services.provider_db_service import ProviderDbService  # noqa: PLC0415

    return ProviderDbService().pin_context_window(config)


class ProviderService:
    """Sends provider requests, resolves selection/limits, sanitises tool args."""

    def __init__(self, mp: MessageProcessor) -> None:
        self.mp = mp
        self._gate_armed: bool = True

    def send(self, request: ProviderRequest) -> ProviderResponse:
        """Cap check, call, log, report usage — the single provider chokepoint.

        The cap check reads what the provider reported for the LAST request on
        this channel; at or past 90% of the window the next one is withheld
        rather than sent, which is what gives compaction room to work before the
        provider would have refused. Once fired the gate disarms for this turn.

        A CHAT call that came back with a token count emits ``context_usage``:
        this is the one place both halves of the meter's fraction exist together
        (``window`` above, ``context_tokens`` below), and one call is exactly one
        move of it. A provider that reports no count is silent rather than zero —
        the surface hides a meter it has no reading for, and a fabricated 0 would
        read as a real, empty context."""
        config = self._select(request.type)
        window = _window_of(config)
        # Stamp the DTO before the client ever sees it: a client that asked its
        # own provider would be a second window for the same send.
        request.context_window = window
        client = Factory.build_client(config)
        if self._gate_applies(request):
            measured = LlmLogService.last_context_tokens(self.mp.channel)
            if measured >= int(0.90 * window):
                self._gate_armed = False
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
        self.mp.llm_log_service.record(response, channel=self.mp.channel if request.type is ProviderType.CHAT else None)
        occupied = response.context_tokens
        if request.type is ProviderType.CHAT and occupied is not None:
            self.mp.push_websocket(
                TurnSignal.context_usage(self.mp, occupied, window),
            )
        return response

    def _gate_applies(self, request: ProviderRequest) -> bool:
        """Whether the previous request's reported usage may gate this one.

        Three conditions, each of which alone makes the reading meaningless:

        * **CHAT only.** Only chat calls carry the conversation, and only chat
          calls write a channel onto their ledger row.
        * **Still armed.** The gate disarms after firing, because the compaction
          that follows makes the reading it fired on stale; re-reading it would
          fire forever.
        * **The channel accumulates history.** ``suppress_history`` channels are
          one-shots — a vision pass, a browse summary, a compaction — sized by
          the payload they were handed, never by a history that grows. What the
          last one cost predicts nothing about this one. They are also exactly
          the channels ``ContextLimit.recover()`` refuses (it re-raises: there is
          no history to compact), so gating them could only ever convert a
          request that might well have fit into a certain failure.
        """
        return (
            request.type is ProviderType.CHAT
            and self._gate_armed
            and not self.mp.config.suppress_history
        )

    def context_limit(self, provider_type: ProviderType = ProviderType.CHAT) -> int:
        """Context window for ``provider_type`` — see :func:`_window_of`."""
        return _window_of(self._select(provider_type))

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
        """Strip leaked provider sentinel tokens (``<|...|>``) from tool args and trim
        surrounding whitespace, recursively.

        The trim is right for an argument the tool INTERPRETS — a path, a query, a
        city name — where padding is noise from the model's formatting. It is wrong
        for one the tool STORES, so an ability names those params in
        ``Ability.VERBATIM`` and :meth:`DispatchService._scrub_values` routes them
        around this method entirely."""
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
        return Factory.build_client(self._select(provider_type))
