"""Conflict-alert proactive hook — real-time overlap detection with reschedule offer.

Fires when the CalDAV monitor detects overlapping future events.  Pushes a
user-facing prompt that names both events, states the overlap, and offers
to find a free slot for rescheduling.

Deduped per conflict pair per UTC day so the same overlap doesn't re-alert.
Respects quiet window.

Called from :meth:`~capabilities.base.AbstractCapability._dispatch_hooks`.
"""

import json
import logging
from datetime import timedelta, timezone

logger = logging.getLogger(__name__)

_FLAG_PREFIX = "conflict_alert:"
_FLAG_TTL = 24 * 3600


def maybe_send_conflict_alert(now=None) -> bool:
    """Push a conflict alert for each new overlap detected today.

    Returns True if at least one alert was enqueued.
    """
    try:
        from capabilities.quiet_window import is_quiet_now
        from services.memory_client import MemoryClientService
        from services.time_utils import utc_now

        now = now or utc_now()
        if is_quiet_now(now):
            return False

        store = MemoryClientService.create_connection()
        conflicts = _fetch_active_conflicts(now)
        if not conflicts:
            return False

        sent_any = False
        for conflict in conflicts:
            canon_key = conflict["canon_key"]
            flag_key = f"{_FLAG_PREFIX}{now.strftime('%Y-%m-%d')}:{canon_key}"
            from capabilities.hook_dedup import is_fired
            if is_fired(flag_key):
                continue

            body = _build_alert_body(conflict, now)
            store.rpush("prompt-queue", json.dumps({
                "prompt": (
                    f"[CONFLICT ALERT]\n{body}\n\n"
                    "Alert the user about this scheduling conflict. "
                    "Name both events and when they overlap. "
                    "Offer to find a free slot to reschedule one of them "
                    "(you can use caldav_find_free_slots). "
                    "Keep it brief — 2-3 sentences. "
                    "Do NOT mention 'CalDAV' or technical details."
                ),
                "metadata": {
                    "type": "proactive_drift",
                    "source": "conflict_alert",
                    "topic": "proactive",
                    "canon_key": canon_key,
                },
            }))
            from capabilities.hook_dedup import mark_fired
            mark_fired(flag_key, _FLAG_TTL)
            logger.info("[conflict_alert] Enqueued for %s.", canon_key)
            sent_any = True
        return sent_any
    except Exception as exc:
        logger.debug("[conflict_alert] failed: %s", exc)
        return False


def _fetch_active_conflicts(now):
    """Query scheduled_items for unresolved CalDAV conflict notifications."""
    from services.database_service import get_shared_db_service

    tz = _user_tz()
    today = now.astimezone(tz).replace(
        hour=0, minute=0, second=0, microsecond=0)
    t_start = today.astimezone(timezone.utc).isoformat()
    t_end = (today + timedelta(days=7)).astimezone(timezone.utc).isoformat()

    with get_shared_db_service().connection() as conn:
        rows = conn.cursor().execute(
            "SELECT message, external_uid FROM scheduled_items "
            "WHERE source='caldav' AND item_type='notification' "
            "AND external_uid LIKE 'caldav:conflict:%' "
            "AND status='pending' "
            "AND created_at >= ? AND created_at < ? "
            "ORDER BY created_at DESC",
            (t_start, t_end),
        ).fetchall()

    conflicts = []
    for msg, ext_uid in rows:
        canon_key = ext_uid.replace("caldav:conflict:", "")
        conflicts.append({"message": msg, "canon_key": canon_key})
    return conflicts


def _build_alert_body(conflict, now):
    """Format a conflict notification into a brief body for the LLM prompt."""
    return (
        f"Conflict detected: {conflict['message']}\n"
        "The user may want to reschedule one of these events."
    )


def _user_tz():
    """Return user's ZoneInfo or UTC."""
    from capabilities.time_context import get_user_tz
    return get_user_tz()


from capabilities.base import register_hook
register_hook(maybe_send_conflict_alert)
