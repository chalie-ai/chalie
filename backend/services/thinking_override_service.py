"""Persisted global thinking-level override.

Set to ``medium`` or ``high`` to bypass the deliberation gate on every request,
across all MessageProcessor instances (user chat, dmn, delegates, background).
Absence means ``auto`` — the gate decides normally.

Stored in the ``settings`` table via the ``Setting`` model so it persists
across restarts and is identical on every device.
"""

from models.setting import Setting

SETTING_KEY = 'thinking_level_override'
VALID_OVERRIDES = frozenset({'medium', 'high'})


def get_thinking_override() -> 'str | None':
    value = Setting.get(SETTING_KEY)
    return value if value in VALID_OVERRIDES else None
