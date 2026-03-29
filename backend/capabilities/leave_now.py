"""Leave-now proactive hook — departure nudge for events with locations.

Fires ~30 minutes before calendar events that have a physical location,
nudging the user to head out.  First hook that crosses the digital-to-
physical boundary — the "Chalie runs my life" moment.

Deduped per event UID so the same event only triggers one departure alert.
Respects quiet window.

Called from :meth:`~capabilities.base.AbstractCapability._dispatch_hooks`.
"""

import json
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)

_FLAG_PREFIX = "leave_now:"
_FLAG_TTL = 6 * 3600
_DEPART_WINDOW_MIN = 35
_DEPART_WINDOW_CLOSE_MIN = 10


def maybe_send_leave_now(now=None) -> bool:
    """Check for upcoming events with locations and nudge departure.

    Fires for events starting in 10-35 minutes that have a non-empty
    location.  Skips all-day events.  Returns True if any nudge was
    enqueued.
    """
    try:
        from capabilities.hook_dedup import is_fired, mark_fired
        from capabilities.quiet_window import is_quiet_now
        from services.database_service import get_shared_db_service
        from services.memory_client import MemoryClientService
        from services.time_utils import parse_utc, utc_now

        now = now or utc_now()
        if is_quiet_now(now):
            return False

        store = MemoryClientService.create_connection()
        window_close = (now + timedelta(minutes=_DEPART_WINDOW_CLOSE_MIN)).isoformat()
        window_far = (now + timedelta(minutes=_DEPART_WINDOW_MIN)).isoformat()

        with get_shared_db_service().connection() as conn:
            rows = conn.cursor().execute(
                "SELECT message, due_at, metadata FROM scheduled_items "
                "WHERE source='caldav' AND item_type='event' AND status='pending' "
                "AND due_at >= ? AND due_at <= ? ORDER BY due_at ASC",
                (window_close, window_far),
            ).fetchall()

        sent_any = False
        for msg, due_raw, meta_raw in rows:
            meta = json.loads(meta_raw) if meta_raw else {}
            location = (meta.get("location") or "").strip()
            if not location or meta.get("all_day"):
                continue

            uid = meta.get("uid", "")
            flag_key = f"{_FLAG_PREFIX}{uid}"
            if is_fired(flag_key):
                continue

            dt = parse_utc(due_raw)
            mins = max(1, int((dt - now).total_seconds() / 60))
            summary = meta.get("summary") or msg or "your event"

            store.rpush("prompt-queue", json.dumps({
                "prompt": (
                    f"[LEAVE NOW]\n"
                    f"Event: {summary}\n"
                    f"Location: {location}\n"
                    f"Starts in: ~{mins} minutes\n\n"
                    "Nudge the user that it's time to head out. "
                    "Mention the event name, location, and how soon it starts. "
                    "Keep it brief and friendly — 1-2 sentences. "
                    "Do NOT mention 'CalDAV' or technical details."
                ),
                "metadata": {
                    "type": "proactive_drift",
                    "source": "leave_now",
                    "topic": "proactive",
                    "event_uid": uid,
                },
            }))
            mark_fired(flag_key, _FLAG_TTL)
            logger.info("[leave_now] Enqueued departure nudge for %s.", uid)
            sent_any = True
        return sent_any
    except Exception as exc:
        logger.debug("[leave_now] failed: %s", exc)
        return False


from capabilities.base import register_hook  # noqa: E402
register_hook(maybe_send_leave_now)
