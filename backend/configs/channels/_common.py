from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from services.message_processor import MessageProcessor

_CONTENT_FIELD_PLACEHOLDER = "{{provider_content_field_name}}"


def substitute_provider_content_field(body: str, mp: "MessageProcessor") -> str:
    """Replace {{provider_content_field_name}} with the active provider's
    CONTENT_FIELD_LABEL, read through the mp-owned providers gateway. Best-effort:
    placeholder absent or resolution fails → body unchanged (design §6.3)."""
    if _CONTENT_FIELD_PLACEHOLDER not in body:
        return body
    try:
        label = mp.providers.selected_provider().CONTENT_FIELD_LABEL
    except Exception:
        label = None
    if not label:
        return body
    return body.replace(_CONTENT_FIELD_PLACEHOLDER, label)




# The single per-config tool list left: the framework discovery tools pinned on
# every discovery-capable channel. Discovery scope itself is not a list — a tool
# is reachable via find_tools iff its ``Ability.DISCOVERABLE`` is True (a global
# trait) and the channel pins ``find_tools`` here. There is no per-channel
# discoverable/blocked roster anymore.
DEFAULT_ALWAYS_AVAILABLE: list[str] = [
    "find_skills",
    "find_tools",
    "memory",
]
