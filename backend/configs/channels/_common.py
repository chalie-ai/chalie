from __future__ import annotations

from abilities.find_skills import FindSkillsAbility
from abilities.find_tools import FindToolsAbility
from abilities.memory import MemoryAbility

# The single per-config tool list left: the framework discovery tools pinned on
# every discovery-capable channel. Discovery scope itself is not a list — a tool
# is reachable via find_tools iff its ``Ability.DISCOVERABLE`` is True (a global
# trait) and the channel pins ``find_tools`` here. There is no per-channel
# discoverable/blocked roster anymore.
DEFAULT_ALWAYS_AVAILABLE: list[str] = [
    FindSkillsAbility.NAME,
    FindToolsAbility.NAME,
    MemoryAbility.NAME,
]
