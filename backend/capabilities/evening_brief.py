"""Evening brief — end-of-day digest: tomorrow's schedule + unanswered emails.

Completes the daily bookend started by ``morning_brief``.  Fires once per
day after 17:00 local time.

Called from :meth:`~capabilities.base.AbstractCapability.run_monitor`.
"""

import json
import logging
from datetime import timedelta, timezone

logger = logging.getLogger(__name__)

_FLAG_PREFIX = "evening_brief:"
_FLAG_TTL = 24 * 3600
_EVENING_HOUR = 17
_UNANSWERED_LIMIT = 5


def maybe_send_evening_brief(now=None) -> bool:
    """Push a once-per-day evening brief via prompt-queue.

    Only fires after 17:00 local.  Deduped per UTC date.
    """
    try:
        from capabilities.quiet_window import is_quiet_now
        from services.memory_client import MemoryClientService
        from services.time_utils import utc_now

        now = now or utc_now()
        if is_quiet_now(now):
            return False
        if now.astimezone(_user_tz()).hour < _EVENING_HOUR:
            return False

        flag_key = f"{_FLAG_PREFIX}{now.strftime('%Y-%m-%d')}"
        from capabilities.hook_dedup import is_fired
        if is_fired(flag_key):
            return False

        cal = _build_tomorrow_section(now)
        email = _build_unanswered_section(now)
        sections = [s for s in (cal, email) if s]
        if not sections:
            return False

        store = MemoryClientService.create_connection()
        hint = " and unanswered emails" if email else ""
        store.rpush("prompt-queue", json.dumps({
            "prompt": (
                "[EVENING BRIEF]\n" + "\n\n".join(sections) + "\n\n"
                f"Give the user a concise evening wrap-up covering "
                f"tomorrow's schedule{hint}. "
                "Warm and brief — 3-5 sentences max."
            ),
            "metadata": {
                "type": "proactive_drift",
                "source": "evening_brief",
                "topic": "proactive",
            },
        }))
        from capabilities.hook_dedup import mark_fired
        mark_fired(flag_key, _FLAG_TTL)
        return True
    except Exception as exc:
        logger.debug("[evening_brief] failed: %s", exc)
        return False


def _user_tz():
    """Return user's ZoneInfo or UTC."""
    from capabilities.time_context import get_user_tz
    return get_user_tz()


def _build_tomorrow_section(now) -> str:
    """Query scheduled_items for tomorrow's events."""
    from services.database_service import get_shared_db_service
    from services.time_utils import parse_utc

    tz = _user_tz()
    tom = (now.astimezone(tz) + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0)
    t_start = tom.astimezone(timezone.utc).isoformat()
    t_end = (tom + timedelta(hours=24)).astimezone(timezone.utc).isoformat()

    with get_shared_db_service().connection() as conn:
        rows = conn.cursor().execute(
            "SELECT message, due_at FROM scheduled_items "
            "WHERE source='caldav' AND item_type='event' "
            "AND status='pending' AND due_at >= ? AND due_at < ? "
            "ORDER BY due_at ASC LIMIT 10",
            (t_start, t_end),
        ).fetchall()

    if not rows:
        return "Tomorrow: No events scheduled."

    lines = [
        f"  {parse_utc(due).astimezone(tz).strftime('%H:%M')} {msg}"
        for msg, due in rows
    ]
    first = parse_utc(rows[0][1]).astimezone(tz)
    return (
        f"Tomorrow ({len(rows)} events):\n" + "\n".join(lines)
        + f"\n\nFirst commitment: {rows[0][0]} at {first.strftime('%H:%M')}"
    )


def _build_unanswered_section(now) -> str:
    """Find today's unanswered emails via IMAP."""
    from capabilities import load_capabilities

    cap = load_capabilities().get("imap")
    if not (cap and cap.is_connected()):
        return ""
    tools = {t["name"]: t["handler"] for t in cap.get_tools()}
    emails = tools["imap_search_email"](
        None,
        {"date_from": now.strftime("%Y-%m-%d"),
         "unanswered": True, "limit": _UNANSWERED_LIMIT},
    ).get("emails", [])
    if not emails:
        return ""
    lines = [f"  - {e.get('from', '?')}: \"{e.get('subject', '?')}\""
             for e in emails]
    return f"Unanswered emails today ({len(emails)}):\n" + "\n".join(lines)


from capabilities.base import register_hook  # noqa: E402
register_hook(maybe_send_evening_brief)
