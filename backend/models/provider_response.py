"""Transient LLM reply — the provider-neutral response every thin client
returns from ``send()``.

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

    def to_dict(self) -> dict[str, object]:
        """Project every declared field to a plain dict (schema-driven off the
        dataclass fields; :meth:`Serializable.to_json` renders it)."""
        return {field.name: getattr(self, field.name) for field in fields(self)}
