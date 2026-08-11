"""Transient LLM reply — the provider-neutral response the app runs on.
``ProviderService.send()`` builds it (field-for-field) from the wire dataclass
the thin clients return (``services.provider_api.ProviderApiResponse``).

A data-model (Rule 8) but transport-only: it never touches disk (no ``save``/
``delete``, no query entry) and holds no ``mp``. It projects "as a dict and as
a json string" through :class:`~models.serializable.Serializable` — the one
shared wire-encoding step — so telemetry / logging can serialize a reply the
same way every other frame is serialized (Essential 8: ZERO duplication).

Ported field-for-field from the former ``ProviderApiResponse`` dataclass.
``thinking_block`` (reasoning content, surfaced for telemetry) and
``response_code`` (HTTP/provider status) are carried alongside the token
counters; each thin client fills what its native payload exposes and leaves
the rest ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

from models.serializable import Serializable


@dataclass
class ProviderResponse(Serializable):
    """Standardised reply returned by every provider client."""

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

    @property
    def context_tokens(self) -> Optional[int]:
        """How much of the context window this request actually occupied.

        The three prompt-side counters are DISJOINT slices of one prompt, not
        overlapping views of it: a cached token is reported under
        ``tokens_cache_read``/``tokens_cache_create`` and is NOT also counted in
        ``tokens_input``. ``tokens_input`` alone therefore measures only the
        uncached remainder — on a long conversation with caching active that is
        a small fraction of what is really in the window, and reading it as
        occupancy would under-report the context by everything cached.

        The same disjoint reading is what
        :meth:`LlmLogService._summarize` computes its cache-hit rate from
        (``tokens_input + tokens_cache_read``) and what its daily total sums, so
        this is the codebase's existing meaning of these columns, stated once
        here for the two places that need occupancy rather than a breakdown:
        the context gate and the usage meter.

        ``None`` when the provider reported no input count at all — absent is
        not zero, and a fabricated 0 would read as a real, empty context.
        """
        if self.tokens_input is None:
            return None
        return self.tokens_input + (self.tokens_cache_read or 0) + (self.tokens_cache_create or 0)

    def to_dict(self) -> dict[str, object]:
        """Project every declared field to a plain dict (schema-driven off the
        dataclass fields; :meth:`Serializable.to_json` renders it)."""
        return {field.name: getattr(self, field.name) for field in fields(self)}
