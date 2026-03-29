"""
Fused morning brief — unified daily digest combining calendar + email.

Self-contained: fetches today's events from ``scheduled_items`` and email
hints from MemoryStore.  Fires once per day during the morning window
(06:00-10:00 local).

Called from :meth:`~capabilities.base.AbstractCapability._dispatch_hooks`.
"""

import json
import logging
from datetime import timedelta, timezone

logger = logging.getLogger(__name__)

_BRIEF_KEY_PREFIX = "morning_brief:"
_BRIEF_TTL_SECONDS = 24 * 3600
_MAX_EVENTS = 10
_MORNING_START = 6   # local hour (inclusive)
_MORNING_END = 10    # local hour (exclusive)


def maybe_send_morning_brief(now=None) -> bool:
    """Push a once-per-day fused morning brief via prompt-queue.

    Combines today's calendar schedule with the cached IMAP inbox hint
    (if available) into a single briefing prompt.  Deduped per UTC day.
    Only fires during the morning window (06:00-10:00 local).
    """
    try:
        from capabilities.quiet_window import is_quiet_now
        from services.memory_client import MemoryClientService
        from services.time_utils import utc_now

        now = now or utc_now()
        if is_quiet_now(now):
            return False

        local_hour = now.astimezone(_user_tz()).hour
        if local_hour < _MORNING_START or local_hour >= _MORNING_END:
            return False

        date_key = now.strftime("%Y-%m-%d")
        flag_key = f"{_BRIEF_KEY_PREFIX}{date_key}"

        from capabilities.hook_dedup import is_fired
        if is_fired(flag_key):
            return False

        store = MemoryClientService.create_connection()

        events = _fetch_today_events(now)
        cal = _build_calendar_section(events)
        b2b = _build_back_to_back_section(events)
        try:
            people = _build_people_section(events)
        except Exception:
            people = ""
        email = _read_cached_inbox_hint(store)

        sections = [cal]
        if b2b:
            sections.append(b2b)
        if people:
            sections.append(people)
        if email:
            sections.append(email)
        body = "\n\n".join(sections)
        instruction = (
            "Give the user a concise morning briefing. "
            "Cover their schedule (conflicts, back-to-back, free blocks)"
        )
        if people:
            instruction += ", mention who they're meeting"
            instruction += " and flag unanswered emails from attendees"
        if email:
            instruction += " and email highlights"
        instruction += ". Keep it warm and brief — 4-5 sentences max."

        store.rpush('prompt-queue', json.dumps({
            'prompt': f"[MORNING BRIEF]\n{body}\n\n{instruction}",
            'metadata': {
                'type': 'proactive_drift',
                'source': 'morning_brief',
                'topic': 'proactive',
            },
        }))
        from capabilities.hook_dedup import mark_fired
        mark_fired(flag_key, _BRIEF_TTL_SECONDS)
        logger.info("[morning_brief] Enqueued for %s.", date_key)
        return True
    except Exception as exc:
        logger.debug("[morning_brief] failed: %s", exc)
        return False


def _user_tz():
    """Return user's ZoneInfo or UTC."""
    from capabilities.time_context import get_user_tz
    return get_user_tz()


def _fetch_today_events(now):
    """Query scheduled_items for today's CalDAV events.

    Returns a list of dicts with ``dtstart``, ``dtend``, ``summary``,
    ``location`` — parsed from the metadata JSON column.
    """
    from services.database_service import get_shared_db_service
    from services.time_utils import parse_utc

    tz = _user_tz()
    today = now.astimezone(tz).replace(
        hour=0, minute=0, second=0, microsecond=0)
    t_start = today.astimezone(timezone.utc).isoformat()
    t_end = (today + timedelta(hours=24)).astimezone(
        timezone.utc).isoformat()

    with get_shared_db_service().connection() as conn:
        rows = conn.cursor().execute(
            "SELECT message, due_at, metadata FROM scheduled_items "
            "WHERE source='caldav' AND item_type='event' "
            "AND status='pending' AND due_at >= ? AND due_at < ? "
            "ORDER BY due_at ASC",
            (t_start, t_end),
        ).fetchall()

    events = []
    for msg, due, meta_raw in rows:
        meta = json.loads(meta_raw) if meta_raw else {}
        dtstart = (parse_utc(meta["dtstart"])
                   if meta.get("dtstart") else parse_utc(due))
        dtend = (parse_utc(meta["dtend"])
                 if meta.get("dtend") else None)
        events.append({
            'summary': msg.split(" [")[0] if " [" in msg else msg,
            'dtstart': dtstart,
            'dtend': dtend,
            'location': meta.get('location'),
            'attendees': meta.get('attendees', []),
        })
    return events


def _build_calendar_section(events):
    """Format today's events into a brief section."""
    if not events:
        return "Schedule: No events today."

    tz = _user_tz()
    shown = events[:_MAX_EVENTS]
    lines = []
    for e in shown:
        t = e['dtstart'].astimezone(tz).strftime('%H:%M')
        line = f"  {t} {e.get('summary', 'Event')}"
        if e.get('location'):
            line += f" @ {e['location']}"
        lines.append(line)

    remaining = len(events) - len(shown)
    if remaining > 0:
        lines.append(f"(+{remaining} more)")
    return f"Schedule ({len(events)} events):\n" + "\n".join(lines)


def _build_back_to_back_section(events):
    """Detect tight transitions (<5min gap) between consecutive events."""
    if len(events) < 2:
        return ""

    pairs = []
    for i in range(len(events) - 1):
        a, b = events[i], events[i + 1]
        if a.get('dtend') and b.get('dtstart'):
            gap = (b['dtstart'] - a['dtend']).total_seconds() / 60
            if 0 <= gap < 5:
                pairs.append((
                    a.get('summary', 'Event'),
                    b.get('summary', 'Event'),
                    int(gap),
                ))

    if not pairs:
        return ""
    lines = [
        f"  {name_a} \u2192 {name_b} ({gap}min gap)"
        for name_a, name_b, gap in pairs
    ]
    return f"Tight transitions ({len(pairs)}):\n" + "\n".join(lines)


def _build_people_section(events) -> str:
    """Resolve attendees across today's events into a people summary.

    Returns a line listing resolved display names, or empty string
    if no attendees are found.
    """
    from capabilities.contact_resolver import resolve

    emails = {a.strip().lower() for e in events
              for a in (e.get('attendees') or []) if a and "@" in str(a)}
    if not emails:
        return ""

    names = []
    for addr in sorted(emails):
        hits = resolve(addr, limit=1)
        if hits and hits[0].get("name"):
            names.append(hits[0]["name"])

    if not names:
        return ""
    return f"Today's attendees: {', '.join(names)}."


def _read_cached_inbox_hint(store) -> str:
    """Read the most recent IMAP hint from WorldState's MemoryStore cache.

    Returns the hint string if found, empty string otherwise.
    """
    try:
        raw_list = store.lrange("world_state:external_signals", 0, -1) or []
        for raw in reversed(raw_list):
            sig = json.loads(raw)
            if sig.get("source") == "imap":
                return sig.get("content", "")
    except Exception:
        pass
    return ""


from capabilities.base import register_hook  # noqa: E402
register_hook(maybe_send_morning_brief)
