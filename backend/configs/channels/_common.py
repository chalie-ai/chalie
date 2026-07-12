from __future__ import annotations

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
