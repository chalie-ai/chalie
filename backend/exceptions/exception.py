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

import logging
from typing import TYPE_CHECKING, ClassVar

from exceptions.chalie_exception import ChalieException

if TYPE_CHECKING:
    from controllers.message_processor import MessageProcessor

logger = logging.getLogger(__name__)


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


class ConflictError(EndpointError):
    """Raised when a request conflicts with the current state of the resource."""

    status: ClassVar[int] = 409


# ── Provider layer ────────────────────────────────────────────────────────────


class ProviderError(ChalieException):
    """Base for all provider communication errors."""


class ContextLimit(ProviderError):
    """The request no longer fits the model's context window.

    One exception for the two ways that fact surfaces, because they mean the
    same thing and take the same recovery:

    - Pre-flight, when the measured payload reaches 90% of the window
      (``ProviderService.send``).
    - Mid-inference, when the provider itself rejects the payload for length —
      every client's native size signal (Anthropic/Ollama HTTP 413, OpenAI
      ``context_length_exceeded``). Providers with no size-specific signal —
      Gemini's catch-all 400, codex's non-zero exit, any openai_compatible host
      not using OpenAI's code — are matched on their message prose instead, via
      ``services.provider_api.is_token_limit_message``.

    Carries the :class:`MessageProcessor` whose turn hit the wall. Clients raise
    it without one — a client has no turn — and ``ProviderService.send``
    attaches its own before re-raising, so any handler can reach the turn that
    has to shrink.
    """

    def __init__(self, message: str, mp: MessageProcessor | None = None, *,
                 window: int = 0, measured: int = 0,
                 provider: str = "", model: str = "") -> None:
        super().__init__(message)
        self.mp = mp
        self.window = window
        self.measured = measured
        self.provider = provider
        self.model = model

    def recover(self) -> None:
        """Compact the turn's history so the next send has room.

        Fires the chat-history compactor on the turn that raised and arms the
        transcript reviewer, so the model can go and read whatever was folded
        away. Re-sending is deliberately NOT done here: the retry belongs to
        the step loop that owns the recursion, not to an exception.
        """
        if self.mp is None:
            raise RuntimeError(
                "ContextLimit.recover() called without a MessageProcessor — "
                "ProviderService.send must attach one before re-raising",
            )
        logger.warning(
            "[ContextLimit] %s tokens against a %s window — compacting turn %s",
            self.measured or "?", self.window or "?", self.mp.turn_id,
        )
        self.mp.dispatch_service.dispatch(
            "chat_history_compactor", {"act_summary": "Compacting conversation"},
        )
        self.mp.post_compaction_continuation = True
        if "review_transcript" not in self.mp.active_tools:
            self.mp.active_tools.append("review_transcript")


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


# ── Turn-execution layer ──────────────────────────────────────────────────────


class RunAwayLoop(ChalieException):
    """A turn's step chain is repeating itself instead of converging.

    A turn has no iteration cap (``processor_config`` — a runaway is a
    hard-restart condition, not a silently-capped one), so a model stuck
    re-emitting the same tool call or the same prose would loop until the context
    caps out. The MessageProcessor trips this loudly when, within one
    ``turn_execution``, either the same ``(tool, params)`` is invoked or the same
    non-empty response text is emitted past its threshold; ``_drive`` catches it
    and stamps the turn CRASHED.
    """


class EmptyCompletionLoop(ChalieException):
    """The model keeps returning completely empty completions.

    A completion with no tool calls and no text, on a turn that has run no
    tools at all, means the model did nothing whatsoever — settling it
    would render silence as an answered turn. The MessageProcessor steers
    a bounded retry first; when the model stays empty past the limit it
    trips this loudly and ``_drive`` stamps the turn CRASHED.
    """


# ── Search layer ──────────────────────────────────────────────────────────────


class UnroutedPromptChannel(ChalieException):
    """A turn's channel has no prompt builder in :meth:`PromptService.user_prompt`.

    A missing dispatch arm is a wiring error at the config, not a turn that
    happens to have nothing to say. Returning an empty body instead handed the
    model a message with no content: a lenient provider answered plausible
    nonsense, a strict one rejected the request and the crash named the vendor
    rather than the omission (see Case V in ``docs/CASE-LAW.md``). Raising stamps
    the turn CRASHED with the unrouted channel in the reason, which is where the
    fault actually is. ``tests/test_code_agent_prompt.py`` enumerates every
    ProcessorConfig so no shipped channel can reach this.
    """

    def __init__(self, channel: str) -> None:
        super().__init__(
            f"Channel {channel!r} has no prompt builder in PromptService.user_prompt — "
            f"a config was added without its dispatch arm"
        )
        self.channel = channel


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


# ── Text reader layer ─────────────────────────────────────────────────────────


class SystemPathBlocked(ChalieException):
    """Reading a system path (/etc, /proc, /dev, /sys, /var/run) was refused.

    The ``read`` ability maps this to ``code=system-path-blocked``.
    """


class NotAFile(ChalieException):
    """The path exists but is not a regular file.

    The ``read`` ability maps this to ``code=not-a-file``.
    """


class SourceIsImage(ChalieException):
    """The source is an image; ``read`` is a text reader.

    The ``read`` ability maps this to ``code=not-text``.
    """


class NoTextContent(ChalieException):
    """A file was read but yielded no text.

    The ``read`` ability maps this to ``code=no-text-content``.
    """


class NoReadableContent(ChalieException):
    """A URL was fetched but yielded no readable text.

    The ``read`` ability maps this to ``code=no-readable-content``.
    """
