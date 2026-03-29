"""Canonical user-timezone helpers for capability hooks.

Single source of truth — every proactive hook that needs local time
imports from here instead of re-implementing ``_user_tz()``.
"""

import logging
from datetime import timezone

logger = logging.getLogger(__name__)


def get_user_tz():
    """Return the user's ZoneInfo timezone, falling back to UTC.

    Reads from ``ClientContextService`` which auto-detects timezone
    from the browser on first session.
    """
    try:
        from zoneinfo import ZoneInfo

        from services.client_context_service import ClientContextService

        name = ClientContextService().get().get("timezone")
        if name:
            return ZoneInfo(name)
    except Exception as exc:
        logger.debug("[time_context] get_user_tz: %s", exc)
    return timezone.utc


def user_local_now(now=None):
    """Return *now* converted to the user's local timezone.

    If *now* is ``None``, uses ``utc_now()``.
    """
    from services.time_utils import utc_now

    now = now or utc_now()
    return now.astimezone(get_user_tz())
