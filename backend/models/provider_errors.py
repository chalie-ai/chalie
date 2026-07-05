from __future__ import annotations


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
    """HTTP 429 — rate limit. A typed provider error; not retried at this layer."""

    def __init__(self, message: str, provider: str = "") -> None:
        super().__init__(message, response_code=429, provider=provider)


class ProviderTimeoutError(ProviderResponseError):
    """The provider call exceeded PROVIDER_CALL_TIMEOUT_S at the HTTP boundary.

    Every thin client maps its SDK's native timeout to this type. The provider
    layer never retries; this (like any provider error) bubbles up to the
    MessageProcessor, which owns the resend policy.
    """


class ProviderRetriesExhaustedError(ProviderResponseError):
    """Raised by the MessageProcessor after every provider resend attempt failed.

    Carries a clean, user-facing message: it is surfaced verbatim on user-facing
    channels (chat error bubble / external-agent reply) once the turn terminates.
    """
