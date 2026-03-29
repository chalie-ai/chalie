"""Quiet window guard — suppress proactive prompts during off-hours and focus.

Checks two conditions before any proactive brief fires:

1. **Quiet hours**: 22:00-08:00 in the user's local timezone (UTC fallback).
2. **Focus mode**: any active ``focus_session:*`` key in MemoryStore.

Called at the top of each ``maybe_send_*`` function so the dedup flag is
never set during quiet time — the brief retries on the next monitor cycle.
"""

import logging
from datetime import timezone

logger = logging.getLogger(__name__)

QUIET_START_HOUR = 22  # 10 PM local
QUIET_END_HOUR = 8     # 8 AM local


def is_quiet_now(now=None) -> bool:
    """Return True if proactive prompts should be suppressed.

    Checks quiet hours (22:00-08:00 local) and active focus sessions.
    """
    try:
        from services.time_utils import utc_now

        now = now or utc_now()
        local_hour = now.astimezone(_get_user_tz()).hour
        if local_hour >= QUIET_START_HOUR or local_hour < QUIET_END_HOUR:
            return True
        return _focus_active()
    except Exception as exc:
        logger.debug("[quiet_window] check failed: %s", exc)
    return False


def _get_user_tz():
    """Return user's ZoneInfo or UTC."""
    try:
        from zoneinfo import ZoneInfo

        from services.client_context_service import ClientContextService

        name = ClientContextService().get().get("timezone")
        if name:
            return ZoneInfo(name)
    except Exception as exc:
        logger.debug("[quiet_window] _get_user_tz: %s", exc)
    return timezone.utc


def _focus_active() -> bool:
    """Return True if any focus session is active in MemoryStore."""
    try:
        from services.memory_client import MemoryClientService

        return bool(MemoryClientService.create_connection().keys("focus_session:*"))
    except Exception as exc:
        logger.debug("[quiet_window] _focus_active: %s", exc)
        return False
