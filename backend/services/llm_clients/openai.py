"""
Official OpenAI (api.openai.com).

The one OpenAI-protocol provider that publishes no size field anywhere: not on
``/models``, not on a completion. So this is the only subclass whose window
probe cannot read a number off a listing, and the only one carrying a table of
documented figures.

Reasoning traces: the chat-completions API returns no reasoning text for
o-series/reasoning models (only ``reasoning_tokens`` counts), so no thinking
trace can be captured here — a platform limit, not a gap in the client.

Consumed by: services.llm_clients.registry (platform dispatch).
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Callable, ClassVar, cast

if TYPE_CHECKING:
    import openai as _openai_mod

from services.llm_clients.openai_compatible import OpenAICompatibleClient

logger = logging.getLogger(__name__)

# Published context windows for official OpenAI models. api.openai.com serves no
# size field on any endpoint, so the documented figure is the primary source
# here, backed by the overflow probe for models not listed yet.
# Matched longest prefix first, because 'gpt-4o'/'gpt-4.1' must not be caught by
# 'gpt-4'.
_OPENAI_PUBLISHED_WINDOWS: tuple[tuple[str, int], ...] = (
    ('gpt-3.5-turbo', 16_385),
    ('gpt-4-turbo', 128_000),
    ('gpt-4.1', 1_047_576),
    ('gpt-4o', 128_000),
    ('gpt-4', 8_192),
    ('gpt-5', 400_000),
    ('o1', 200_000),
    ('o3', 200_000),
    ('o4', 200_000),
)

_CONTEXT_LIMIT_RE = re.compile(
    r"maximum context length is\s+([\d,_]+)\s*tokens", re.I,
)

# Deliberately absurd, so no real model can satisfy it and the rejection is
# guaranteed. The request is never generated from — it is refused on validation.
_OVERFLOW_TOKENS = 1_000_000_000


def _window_from_error_message(message: str) -> int | None:
    """The window a provider named while rejecting an over-large request.

    Parses the OpenAI-style error string: ``"This model's maximum context
    length is 16385 tokens. However, you requested 44366 tokens ..."``. The
    captured group is accepted with commas and underscores stripped.
    """
    match = _CONTEXT_LIMIT_RE.search(message or "")
    if match is None:
        return None
    text = match.group(1).replace(",", "").replace("_", "")
    try:
        value = int(text)
        return value if value > 0 else None
    except ValueError:
        return None


def _extract_window_from_error(exc: object) -> int | None:
    """The window named inside a rejection, from either place it can appear.

    ``str(exc)`` is always available; the SDK additionally parses the JSON body
    onto ``exc.body``, where the message lives under ``error.message``. Either
    is a real measurement — the server is stating its own limit.
    """
    window = _window_from_error_message(str(exc))
    if window is not None:
        return window
    body = getattr(exc, 'body', None)
    error = body.get('error') if isinstance(body, dict) else None
    message = error.get('message') if isinstance(error, dict) else None
    return _window_from_error_message(message) if isinstance(message, str) else None


def _published_openai_window(model: str) -> int | None:
    """Documented window for an official OpenAI model slug, or None if unknown.

    Unknown means unknown: a new slug falls through to the probe rather than
    borrowing a neighbouring family's number, which could be wrong by two
    orders of magnitude.
    """
    slug = (model or '').lower()
    for prefix, window in sorted(
        _OPENAI_PUBLISHED_WINDOWS, key=lambda pair: len(pair[0]), reverse=True,
    ):
        if slug.startswith(prefix):
            return window
    return None


class OpenAIClient(OpenAICompatibleClient):
    """api.openai.com — the SDK's own default base URL."""

    PLATFORM: ClassVar[str] = 'openai'
    LABEL: ClassVar[str] = 'OpenAI'
    DEFAULT_BASE_URL: ClassVar[str] = ''   # the SDK already points at OpenAI
    REQUIRES_KEY: ClassVar[bool] = True
    REQUIRES_HOST: ClassVar[bool] = False

    # OpenAI validates the request body strictly and rejects unknown top-level
    # keys, so the vendor-extension thinking toggle must never be sent here.
    SENDS_THINKING_EXTRA_BODY: ClassVar[bool] = False

    # /models returns id/object/created/owned_by and nothing else — no size
    # field exists to read, which is why the probe below never consults one.
    WINDOW_FIELDS: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def fetch_models(
        cls, host: str, api_key: str,
    ) -> tuple[list[dict[str, str | None]] | None, str | None]:
        """Keyed against api.openai.com, which the SDK already knows.

        Overridden because the inherited implementation needs a base URL to
        call, and this platform deliberately has none to give.
        """
        from services.provider_probe import fetch_openai_models  # noqa: PLC0415
        return fetch_openai_models(api_key)

    def _probe_context_window(self) -> int | None:
        """Documented figure first, then the number OpenAI quotes in a refusal.

        The table leads because it is exact for every model it knows. The
        overflow probe covers the rest — a model released after this table was
        written still gets a real measurement rather than a default.
        """
        window = _published_openai_window(self.model)
        if window is None:
            window = self._probe_via_overflow()
        if window is None:
            window = self._probe_via_ping()
        return window

    def _probe_via_overflow(self) -> int | None:
        """The window named in the provider's refusal of an impossible budget.

        api.openai.com publishes no size field on any endpoint, so the only
        number that comes from the provider itself is the one it quotes while
        refusing: ``"This model's maximum context length is 16385 tokens."``
        Asking for a billion completion tokens is refused on validation, before
        any generation happens. Older models reject ``max_completion_tokens``
        outright, so the retry re-asks with ``max_tokens``.

        Best-effort by design: a model may instead refuse by quoting its
        *output* cap, which is a different number and correctly goes unmatched,
        yielding None. That is why this sits behind the published table rather
        than in front of it.

        Restricted to the official endpoint, and not inherited by any subclass:
        a third-party server may clamp an oversized budget instead of refusing
        it, turning the probe into a real generation request.
        """
        if self._config.get('host'):
            return None

        import openai as openai_mod  # noqa: PLC0415
        try:
            client = self._get_client()
            create = cast(
                "Callable[..., _openai_mod.types.chat.ChatCompletion]",
                client.chat.completions.create,
            )
            try:
                create(
                    model=self.model,
                    messages=[{"role": "user", "content": "."}],
                    max_completion_tokens=_OVERFLOW_TOKENS,
                )
            except openai_mod.APIStatusError as first_err:
                extracted = _extract_window_from_error(first_err)
                if extracted is not None:
                    return extracted
                # Older models reject max_completion_tokens but accept max_tokens — retry.
                try:
                    create(
                        model=self.model,
                        messages=[{"role": "user", "content": "."}],
                        max_tokens=_OVERFLOW_TOKENS,
                    )
                except openai_mod.APIStatusError as second_err:
                    return _extract_window_from_error(second_err)
        except Exception as exc:
            logger.warning(
                "[%s] Overflow probe failed for model=%s: %s",
                type(self).__name__, self.model, exc,
            )
            return None

        return None
