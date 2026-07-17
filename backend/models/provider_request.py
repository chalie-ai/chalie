"""Transient LLM send payload — the provider-neutral request model.

A pure data model (Rule 8): holds no ``mp``, imports no service, never
persists (no table, no ``save``). The caller assembles every field from its
own state; ``ProviderService.send`` receives this model and hands it to the
resolved thin client, which reads its fields directly. ``to_dict``/``to_json``
project the provider-neutral contract (the telemetry metadata is excluded —
it is not part of that contract).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from configs.enums.provider_type import ProviderType
from configs.enums.thinking_level import ThinkingLevel
from models.serializable import Serializable


@dataclass
class ProviderRequest(Serializable):
    """Caller-built, provider-neutral send request.

    Fields mirror the informal ``send_messages()`` signature that existed
    across all clients; defaults match production behaviour.
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

        Uses the explicit ``max_tokens`` when set; otherwise reserves the same
        headroom used by the over-cap check (``max(10% window, 8 000)``) so the
        request-sizing formula and the output ceiling stay symmetric.
        """
        if self.max_tokens is not None:
            return self.max_tokens
        return max(int(window * 0.1), 8000)

    def to_dict(self) -> dict[str, object]:
        """Project the provider-neutral request fields (enums as their value)."""
        return {
            "system": self.system,
            "messages": self.messages,
            "type": self.type.value,
            "tools": self.tools,
            "thinking_mode": self.thinking_mode.value,
            "format": self.format,
            "cache_prefix": self.cache_prefix,
            "max_tokens": self.max_tokens,
        }
