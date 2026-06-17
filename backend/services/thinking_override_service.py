"""Persisted global thinking-level override.

Set to ``medium`` or ``high`` to bypass the deliberation gate on every request,
across all MessageProcessor instances (user chat, dmn, delegates, background).
Absence means ``auto`` — the gate decides normally.

Stored in the ``settings`` table via SettingsService so it persists across restarts
and is identical on every device.
"""

from services.database_service import get_shared_db_service
from services.settings_service import SettingsService

SETTING_KEY = 'thinking_level_override'
VALID_OVERRIDES = frozenset({'medium', 'high'})


def get_thinking_override() -> 'str | None':
    value = SettingsService(get_shared_db_service()).get(SETTING_KEY)
    return value if value in VALID_OVERRIDES else None
