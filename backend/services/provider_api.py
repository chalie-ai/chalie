from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from configs.enums.provider_type import ProviderType
from configs.enums.thinking_level import ThinkingLevel

# Hard ceiling on any provider's reported context window.
MAX_CONTEXT_WINDOW = 200_000

# Operating default per platform, used ONLY when nothing truthful is known: the
# provider row carries no window AND the client's live probe returned None.
# Deliberately separate from ``ProviderClient.get_context_limit()``, which
# reports measurements only — a guess must never be persisted onto a provider
# row, but a send still needs some number to size compaction against. Values are
# each platform's documented baseline. The unknown-platform floor is deliberately
# small: compacting too eagerly is recoverable, overshooting the real window is
# a dead turn.
DEFAULT_CONTEXT_WINDOW = 8_192
DEFAULT_CONTEXT_WINDOWS: dict[str, int] = {
    'anthropic': 200_000,   # Claude 3+ documented window
    'gemini': 1_000_000,    # Gemini 1.5+ documented window
    'openai': 128_000,      # GPT-4 class documented window
    'openai_compatible': 128_000,
    'codex_cli': 272_000,   # GPT-5 class default reported by the codex CLI
    'ollama': DEFAULT_CONTEXT_WINDOW,
}

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

    def resolve_max_tokens(self, window: int) -> int:
        """Return the output-token ceiling for a given context window.

        Uses the explicit max_tokens when set; otherwise reserves the same
        headroom used by the over-cap check (max(10% window, 8 000)) so the
        request-sizing formula and the output ceiling stay symmetric.
        """
        if self.max_tokens is not None:
            return self.max_tokens
        return max(int(window * 0.1), 8000)


@dataclass
class ProviderApiResponse:
    """Standardised response returned by all provider clients.

    Extends today's LLMResponse with two new fields:
      - thinking_block: the reasoning content (Anthropic: was parsed then
        dropped; now surfaced for telemetry and possible future display).
      - response_code: the HTTP / provider status code for telemetry.

    Replaces LLMResponse everywhere — callers must import from here.
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
    thinking_block: Optional[str] = None
    response_code: Optional[int] = None
