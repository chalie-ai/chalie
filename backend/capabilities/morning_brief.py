"""
Fused morning brief — unified daily digest combining calendar + email.

Replaces separate CalDAV daily digest and IMAP scheduled 8am digest with
a single prompt-queue push that summarises both capabilities.
"""

import json
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)

_BRIEF_KEY_PREFIX = "morning_brief:"
_BRIEF_TTL_SECONDS = 24 * 3600
_MAX_EVENTS = 10


def maybe_send_morning_brief(calendar_events: list, now) -> bool:
    """Push a once-per-day fused morning brief via prompt-queue.

    Combines today's calendar schedule with the cached IMAP inbox hint
    (if available) into a single briefing prompt.  Deduped per UTC day.
    """
    try:
        from services.memory_client import MemoryClientService

        date_key = now.strftime("%Y-%m-%d")
        flag_key = f"{_BRIEF_KEY_PREFIX}{date_key}"

        store = MemoryClientService.create_connection()
        if store.get(flag_key):
            return False

        cal = _build_calendar_section(calendar_events, now)
        b2b = _build_back_to_back_section(calendar_events, now)
        email = _read_cached_inbox_hint(store)

        sections = [cal]
        if b2b:
            sections.append(b2b)
        if email:
            sections.append(email)
        body = "\n\n".join(sections)
        instruction = (
            "Give the user a concise morning briefing. "
            "Cover their schedule (conflicts, back-to-back, free blocks)"
        )
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
        store.setex(flag_key, _BRIEF_TTL_SECONDS, "1")
        logger.info("[morning_brief] Enqueued for %s.", date_key)
        return True
    except Exception as exc:
        logger.debug("[morning_brief] failed: %s", exc)
        return False


def _build_calendar_section(events: list, now) -> str:
    """Format today's calendar events into a brief section."""
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    today_end = today_start + timedelta(hours=24)
    todays = sorted(
        [e for e in events
         if e.get('dtstart') and today_start <= e['dtstart'] < today_end],
        key=lambda e: e['dtstart'],
    )
    if not todays:
        return "Schedule: No events today."

    from capabilities.caldav_capability.capability import _format_event_line

    shown = todays[:_MAX_EVENTS]
    lines = [_format_event_line(e) for e in shown]
    remaining = len(todays) - len(shown)
    if remaining > 0:
        lines.append(f"(+{remaining} more)")
    return f"Schedule ({len(todays)} events):\n" + "\n".join(lines)


def _build_back_to_back_section(events: list, now) -> str:
    """Detect tight transitions (<5min gap) and format as a brief section."""
    from capabilities.caldav_capability.capability import _find_back_to_back_pairs

    pairs = _find_back_to_back_pairs(events, now)
    if not pairs:
        return ""
    lines = [
        f"  {a.get('summary', 'Event')} \u2192 "
        f"{b.get('summary', 'Event')} ({gap}min gap)"
        for a, b, gap, _ in pairs
    ]
    return f"Tight transitions ({len(pairs)}):\n" + "\n".join(lines)


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
