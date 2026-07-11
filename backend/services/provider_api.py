from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from configs.enums.provider_type import ProviderType
from configs.enums.thinking_level import ThinkingLevel

# Hard ceiling on any provider's reported context window.
MAX_CONTEXT_WINDOW = 200_000

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

    # Telemetry metadata — set by the caller at construction time so the
    # ProviderService chokepoint can log without reaching into mp or external state.
    # Not part of the provider-neutral contract: excluded from __eq__/repr so
    # they never affect compaction or test assertions.
    # job_name: empty string means "skip log_call" (probe DTOs leave this empty).
    _job_name: Optional[str] = field(default=None, compare=False, repr=False)
    _usage_class: str = field(default='chat', compare=False, repr=False)
    _caller: str = field(default='', compare=False, repr=False)

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
