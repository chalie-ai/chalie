"""Thinking-level override — a single persisted global setting.

When set to ``medium`` or ``high`` the user has chosen to bypass the deliberation
gate and force that thinking level on every request, across every
MessageProcessor instance (user chat, dmn, delegates, background). Absence of the
setting means ``auto`` — the deliberation gate decides, as it always has.

The value is stored in the ``settings`` table (key ``thinking_level_override``)
via the same SettingsService used for ``selected_provider_id``, so it persists
across restarts and is identical on every device.
"""

from services.database_service import get_shared_db_service
from services.settings_service import SettingsService

SETTING_KEY = 'thinking_level_override'
VALID_OVERRIDES = frozenset({'medium', 'high'})


def get_thinking_override() -> 'str | None':
    """Return the persisted override (``'medium'`` | ``'high'``) or None (auto).

    Any value outside the valid set — including a missing row — resolves to
    None so the deliberation gate stays in control.
    """
    value = SettingsService(get_shared_db_service()).get(SETTING_KEY)
    return value if value in VALID_OVERRIDES else None
