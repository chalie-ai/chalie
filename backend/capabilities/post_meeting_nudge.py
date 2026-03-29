"""Post-meeting nudge -- surface unanswered emails after meetings end.

Fires within ~5 minutes of a calendar event ending.  Checks for
unanswered email threads from attendees and pushes a nudge to
``prompt-queue``.

Called from :meth:`~capabilities.base.AbstractCapability.run_monitor` after
each successful monitor cycle (same hook point as ``meeting_prep``).
"""

import json
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)

_FLAG_PREFIX = "post_meeting:"
_FLAG_TTL = 4 * 3600
_WINDOW_MIN = 5
_LOOKBACK_DAYS = 7


def maybe_send_post_meeting_nudge(now=None) -> bool:
    """Check recently ended meetings; nudge about unanswered emails."""
    try:
        from capabilities.quiet_window import is_quiet_now
        from services.database_service import get_shared_db_service
        from services.memory_client import MemoryClientService
        from services.time_utils import parse_utc, utc_now

        now = now or utc_now()
        if is_quiet_now(now):
            return False
        store = MemoryClientService.create_connection()
        lookback = (now - timedelta(hours=4)).isoformat()

        with get_shared_db_service().connection() as conn:
            rows = conn.cursor().execute(
                "SELECT message, due_at, metadata FROM scheduled_items "
                "WHERE source='caldav' AND item_type='event' "
                "AND status='pending' AND due_at >= ? AND due_at <= ? "
                "ORDER BY due_at ASC",
                (lookback, now.isoformat()),
            ).fetchall()

        sent_any = False
        for msg, _due, meta_raw in rows:
            meta = json.loads(meta_raw) if meta_raw else {}
            uid = meta.get("uid", "")
            dtend_raw = meta.get("dtend")
            attendees = meta.get("attendees", [])
            flag = f"{_FLAG_PREFIX}{uid}"

            from capabilities.hook_dedup import is_fired
            if (meta.get("all_day") or not attendees
                    or not dtend_raw or is_fired(flag)):
                continue

            dtend = parse_utc(dtend_raw)
            elapsed = (now - dtend).total_seconds() / 60
            if not (0 <= elapsed <= _WINDOW_MIN):
                continue

            nudge = _build_nudge(msg, dtend, attendees, now)
            if not nudge:
                continue

            store.rpush("prompt-queue", json.dumps({
                "prompt": (
                    f"[POST-MEETING NUDGE]\n{nudge}\n\n"
                    "The user's meeting just ended. Mention the "
                    "unanswered emails from attendees so they can "
                    "follow up while context is fresh. 2-3 sentences. "
                    "Do NOT mention 'CalDAV' or 'IMAP'."
                ),
                "metadata": {
                    "type": "proactive_drift",
                    "source": "post_meeting_nudge",
                    "topic": "proactive", "event_uid": uid,
                },
            }))
            from capabilities.hook_dedup import mark_fired
            mark_fired(flag, _FLAG_TTL)
            sent_any = True
        return sent_any
    except Exception as exc:
        logger.debug("[post_meeting_nudge] failed: %s", exc)
        return False


def _build_nudge(summary, dtend, attendees, now):
    """Return nudge body if attendees have unanswered emails, else None."""
    from capabilities import load_capabilities

    cap = load_capabilities().get("imap")
    if not cap or not cap.is_connected():
        return None
    tools = {t["name"]: t["handler"] for t in cap.get_tools()}
    fn = tools["imap_search_email"]

    since = (now - timedelta(days=_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    needs_reply = []
    for addr in attendees[:5]:
        for e in fn(
            None, {"sender": addr, "date_from": since,
                    "unanswered": True, "limit": 3},
        ).get("emails", []):
            needs_reply.append(
                f"- {addr}: \"{e.get('subject', '?')}\"")

    if not needs_reply:
        return None
    return (
        f"Meeting ended: {summary} ({dtend.strftime('%H:%M')})\n"
        f"Attendees: {', '.join(attendees[:5])}\n"
        "\nUnanswered emails:\n" + "\n".join(needs_reply)
    )


from capabilities.base import register_hook  # noqa: E402
register_hook(maybe_send_post_meeting_nudge)
