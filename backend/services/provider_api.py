from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from configs.enums.provider_type import ProviderType
from configs.enums.thinking_level import ThinkingLevel

# Hard ceiling on any provider's stored context window, applied on every write
# to ``providers.context_window`` regardless of how much the provider claims to
# support. Nothing downstream re-checks it: the column is already clamped.
MAX_CONTEXT_WINDOW = 200_000

# Provider prose that means "your request was too big". Some providers have no
# machine-readable size-fault code — Gemini reports 400/INVALID_ARGUMENT, which
# also covers wrong params and regions; codex reports a non-zero exit; a
# non-OpenAI host behind the OpenAI-compatible API rarely sends OpenAI's
# 'context_length_exceeded'. Those can only be told apart by their message, so
# the vocabulary lives here once instead of drifting per client. A miss must
# fall through to the generic error and log — never be swallowed.
TOKEN_LIMIT_STRINGS = frozenset({
    'token limit',
    'context length',
    'too long',
    'maximum context',
    'input too large',
    'request payload size',
    'exceeds the limit',
})


def is_token_limit_message(text: str | None) -> bool:
    """True when a provider's error prose reports a size rejection.

    Owns the casefold so no caller can match case-sensitively by accident.
    """
    return any(s in (text or '').lower() for s in TOKEN_LIMIT_STRINGS)


# Deadline (seconds) for a single provider API call, enforced by every thin
# client at its own HTTP boundary (the only place a call can actually be
# interrupted — a Python thread cannot be killed). On expiry the client raises
# ProviderTimeoutError, which the retry helper surfaces immediately. This is the
# ONE provider-call timeout; clients import this constant rather than hard-coding.
PROVIDER_CALL_TIMEOUT_S = 300


@dataclass
class ProviderApiRequest:
    """Caller-built, provider-neutral send request.

    The caller assembles every field from its own state; ProviderService.send()
    does NOT reach into mp or any external context — it only receives this DTO.

    Fields mirror the informal send_messages() signature that existed across
    all clients; defaults match today's production behaviour.
    """

    # Required: message content
    system: str
    messages: list[dict[str, object]]

    # Provider routing
    type: ProviderType = field(default=ProviderType.CHAT)

    # Optional request shaping
    tools: Optional[list[dict[str, object]]] = field(default=None)
    thinking_mode: ThinkingLevel = field(default=ThinkingLevel.MEDIUM)
    format: str = field(default="text")
    cache_prefix: bool = field(default=True)

    # Output-token ceiling — None means "use formula" (see resolve_max_tokens).
    max_tokens: Optional[int] = field(default=None)

    # The context window this request will be sent against, stamped by
    # ProviderService.send() from providers.context_window. Clients read it
    # rather than asking their provider, so the DB column is the only window
    # any send is ever sized against.
    context_window: Optional[int] = field(default=None)

    def resolve_max_tokens(self) -> int:
        """Return the output-token ceiling for this request.

        Uses the explicit max_tokens when set; otherwise reserves the same
        headroom used by the over-cap check (max(10% window, 8 000)) so the
        request-sizing formula and the output ceiling stay symmetric.

        Raises when neither is available: a silent guess here is exactly the
        fiction ``providers.context_window`` exists to eliminate.
        """
        if self.max_tokens is not None:
            return self.max_tokens
        if self.context_window is None:
            raise ValueError(
                "resolve_max_tokens() needs either an explicit max_tokens or a "
                "context_window stamped by ProviderService.send()"
            )
        return max(int(self.context_window * 0.1), 8000)


@dataclass
class ProviderApiResponse:
    """Wire-layer reply returned by every thin ``llm_clients/*`` client.

    Transport-only: ``ProviderService.send()`` converts it — field-for-field —
    into :class:`models.provider_response.ProviderResponse`, which is what the
    rest of the app (the context gate's ``context_tokens`` read included)
    consumes. Keep the two field lists in step.

    Carries ``thinking_block`` (reasoning content, surfaced for telemetry) and
    ``response_code`` (HTTP / provider status) alongside the token counters.
    """

    text: str
    model: str
    provider: Optional[str] = None
    tokens_input: Optional[int] = None
    tokens_output: Optional[int] = None
    tokens_thinking: Optional[int] = None
    tokens_cache_read: Optional[int] = None
    tokens_cache_create: Optional[int] = None
    tool_calls: Optional[list[dict[str, object]]] = None
    stop_reason: Optional[str] = None
    latency_ms: Optional[int] = None
    prefill_ms: Optional[float] = None
    decode_ms: Optional[float] = None
    thinking_block: Optional[str] = None
    response_code: Optional[int] = None
