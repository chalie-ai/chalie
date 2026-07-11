"""Every Chalie custom exception, in one place.

All exceptions extend :class:`~backend.exceptions.chalie_exception.ChalieException`,
which logs each raise at ERROR level. Import from the package surface::

    from exceptions import RateLimitError, NotFoundError

The existing layer hierarchies are preserved — ``EndpointError`` /
``NotFoundError`` / ``ForbiddenError`` for the API envelope, and ``ProviderError``
and its subclasses for provider communication — so ``except <Base>`` catches keep
their meaning and nothing downstream changes semantically.
"""

from __future__ import annotations

from typing import ClassVar

from exceptions.chalie_exception import ChalieException


# ── API endpoint layer ────────────────────────────────────────────────────────


class EndpointError(ChalieException):
    """Typed endpoint failure; the base maps it onto the error envelope with its status."""

    status: ClassVar[int] = 400


class NotFoundError(EndpointError):
    """Requested resource does not exist."""

    status: ClassVar[int] = 404


class ForbiddenError(EndpointError):
    """Authenticated, but not permitted to perform this action."""

    status: ClassVar[int] = 403


# ── Provider layer ────────────────────────────────────────────────────────────


class ProviderError(ChalieException):
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


# ── Search layer ──────────────────────────────────────────────────────────────


class RateLimitException(ChalieException):
    """Raised when a downstream search engine enforces a rate limit."""


# ── MCP layer ─────────────────────────────────────────────────────────────────


class McpToolUnknown(ChalieException):
    """An ``_mcp_*`` name resolves to no enabled/registered server.

    Distinct from a reachable-but-failing call: the tool name itself cannot be
    routed (no matching server, or the matching server is disabled), so retrying
    is pointless until the server is (re-)added/enabled.
    """


class McpServerUnreachable(ChalieException):
    """The remote MCP server could not be reached (transport/connect/timeout).

    Carries the human-facing ``server_name`` so the proxy can NAME the failing
    endpoint in its error envelope instead of leaking a transport stack trace.
    """

    def __init__(self, server_name: str, detail: str) -> None:
        super().__init__(f"MCP server {server_name!r} is unreachable: {detail}")
        self.server_name = server_name
        self.detail: str = detail


# ── Web fetch layer ───────────────────────────────────────────────────────────


class FetchBlocked(ChalieException):
    """Raised when the SSRF guard refuses a destination (private/internal host)."""


class DownloadTooLarge(ChalieException):
    """Raised by :func:`stream_to_file` when a download exceeds its byte cap.

    The partial file is removed before this is raised — never a silent truncation.
    """

    def __init__(self, max_bytes: int) -> None:
        super().__init__(f"download exceeds the {max_bytes}-byte cap")
        self.max_bytes = max_bytes


# ── News layer ────────────────────────────────────────────────────────────────


class NewsFetchError(ChalieException):
    """A news provider was unreachable or returned a transport-level error.

    Raised by :meth:`NewsService.fetch_google_news` when the HTTP fetch to the
    Google News RSS endpoint fails (connection refused, timeout, non-2xx). The
    message carries the provider/URL context. The ability maps this to
    ``code=provider-unreachable`` instead of letting a dead provider masquerade
    as an empty result set.

    Per-feed RSS failures inside :meth:`NewsService._parse_feed` are NOT raised:
    a single dead feed in a multi-feed aggregate is normal and tolerated.
    """


# ── Vault layer ───────────────────────────────────────────────────────────────


class VaultLockedError(ChalieException):
    """Raised when an encrypt/decrypt operation is attempted on a sealed vault.

    The vault must be unlocked via :meth:`VaultService.unlock` before any
    cryptographic operations can be performed.
    """


# ── Snapshot layer ────────────────────────────────────────────────────────────


class SnapshotError(ChalieException):
    """Raised when an import is rejected loudly (bad password handled by the
    zip layer, corrupt zip, checksum mismatch, or a schema-downgrade block)."""


# ── Ability / tool layer ──────────────────────────────────────────────────────


class ToolParamError(ChalieException):
    """Raised by ``Ability.param`` when an input is missing/invalid/out-of-choice.

    Carries the same self-correction fields as an error ``ToolResult`` so the
    dispatcher can render it canonically (``code``/``hint``/``valid``) without the
    ability ever formatting an envelope.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "invalid-param",
        hint: str | None = None,
        valid: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.hint = hint
        self.valid = tuple(valid)
