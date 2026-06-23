from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ProviderType(Enum):
    CHAT = "chat"
    VISION = "vision"
    DELEGATE = "delegate"
    VISUAL_OUTPUT = "visual_output"  # reserved, not wired


class ThinkingLevel(Enum):
    """Formalises the existing low/medium/high strings; adds MAX (additive).

    Each thin client maps the level to its native flag internally.
    LOW is the floor — maps to "no flag" on Anthropic/OpenAI/Gemini.
    The Ollama quirk (think gated on model capability, level ignored) is
    preserved in OllamaClient and not represented here.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAX = "max"


@dataclass
class ProviderApiRequest:
    """Caller-built, provider-neutral send request.

    The caller assembles every field from its own state; Providers.send()
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
    # Providers chokepoint can log without reaching into mp or external state.
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


# ── Exception hierarchy ───────────────────────────────────────────────────────


class ProviderError(Exception):
    """Base for all provider communication errors."""


class RequestOverCapError(ProviderError):
    """Pre-flight: measured request exceeds the context window cap.

    Replaces the OVER_CAP sentinel. The ACT loop catches this and fires
    both compactors before retrying.
    """

    def __init__(self, message: str, window: int = 0, measured: int = 0,
                 cap: int = 0, provider: str = "", model: str = "") -> None:
        super().__init__(message)
        self.window = window
        self.measured = measured
        self.cap = cap
        self.provider = provider
        self.model = model


class ResponseOverLimitError(ProviderError):
    """Post-flight: the provider rejected the request server-side for size.

    Replaces PayloadTooLargeError and is raised by ALL provider clients on
    their native size-rejection signals:
      - Anthropic: HTTP 413
      - OpenAI: HTTP 400 with error.code == 'context_length_exceeded'
      - Gemini: token-count exceeded error
      - Ollama: HTTP 413

    The ACT loop catches this with the same compact-then-retry path as
    RequestOverCapError.
    """

    def __init__(self, message: str, response_code: int = 0,
                 provider: str = "", model: str = "") -> None:
        super().__init__(message)
        self.response_code = response_code
        self.provider = provider
        self.model = model


class ProviderResponseError(ProviderError):
    """The provider call failed or returned an unusable response.

    Carries the HTTP/status code (when available) and the provider's error
    message. Does not trigger compact-and-retry — it is a genuine API error.
    """

    def __init__(self, message: str, response_code: int = 0, provider: str = "") -> None:
        super().__init__(message)
        self.response_code = response_code
        self.provider = provider


class RateLimitError(ProviderResponseError):
    """HTTP 429 — rate limit. Keeps retry_after for the retry helper."""

    def __init__(self, message: str, retry_after: Optional[float] = None,
                 provider: str = "") -> None:
        super().__init__(message, response_code=429, provider=provider)
        self.retry_after = retry_after
