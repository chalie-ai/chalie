"""Pre-meeting brief -- proactive cross-capability briefing before meetings.

Fires ~15 minutes before each calendar event that has attendees.  Resolves
attendee names via the people index and surfaces recent email threads,
then pushes an LLM-synthesized brief to ``prompt-queue``.

Called from :meth:`~capabilities.base.AbstractCapability.run_monitor` after
each successful monitor cycle (same hook point as ``first_look``).
"""

import json
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)

_FLAG_PREFIX = "meeting_prep:"
_FLAG_TTL = 4 * 3600
_LOOKAHEAD_MIN = 20
_LOOKBACK_DAYS = 7


def maybe_send_meeting_prep(now=None) -> bool:
    """Check for imminent meetings and enqueue pre-meeting briefs.

    Returns True if any brief was enqueued.
    """
    try:
        from capabilities.quiet_window import is_quiet_now
        from services.database_service import get_shared_db_service
        from services.memory_client import MemoryClientService
        from services.time_utils import parse_utc, utc_now

        now = now or utc_now()
        if is_quiet_now(now):
            return False
        store = MemoryClientService.create_connection()
        cutoff = (now + timedelta(minutes=_LOOKAHEAD_MIN)).isoformat()

        with get_shared_db_service().connection() as conn:
            rows = conn.cursor().execute(
                "SELECT message, due_at, metadata FROM scheduled_items "
                "WHERE source='caldav' AND item_type='event' AND status='pending' "
                "AND due_at >= ? AND due_at <= ? ORDER BY due_at ASC",
                (now.isoformat(), cutoff),
            ).fetchall()

        sent_any = False
        for msg, due, meta_raw in rows:
            meta = json.loads(meta_raw) if meta_raw else {}
            attendees = meta.get("attendees", [])
            uid = meta.get("uid", "")
            flag = f"{_FLAG_PREFIX}{uid}"
            if meta.get("all_day") or not attendees or store.get(flag):
                continue

            brief = _build_brief(
                msg, parse_utc(due), meta.get("location"), attendees, now,
            )
            dt = parse_utc(due)
            mins = max(1, int((dt - now).total_seconds() / 60))
            store.rpush("prompt-queue", json.dumps({
                "prompt": (
                    f"[PRE-MEETING BRIEF]\n{brief}\n\n"
                    f"Brief the user about their meeting in ~{mins} minutes. "
                    "Highlight attendees and relevant recent email context. "
                    "If there are emails needing reply, emphasize those — "
                    "the user should know before walking into the meeting. "
                    "3-5 sentences max. Do NOT mention 'CalDAV' or 'IMAP'."
                ),
                "metadata": {
                    "type": "proactive_drift", "source": "meeting_prep",
                    "topic": "proactive", "event_uid": uid,
                },
            }))
            store.setex(f"{_FLAG_PREFIX}{uid}", _FLAG_TTL, "1")
            sent_any = True
        return sent_any
    except Exception as exc:
        logger.debug("[meeting_prep] failed: %s", exc)
        return False


def _build_brief(summary, dtstart, location, attendees, now):
    """Build brief body: headline, resolved names, and email context."""
    from capabilities import load_capabilities
    from capabilities.contact_resolver import resolve

    headline = f"Meeting: {summary} at {dtstart.strftime('%H:%M')}"
    if location:
        headline += f" @ {location}"

    resolved = []
    for addr in attendees[:5]:
        hits = resolve(addr, limit=1)
        resolved.append({"email": addr,
                         "name": (hits[0].get("name") or addr) if hits else addr})

    parts = [headline, "Attendees: " + ", ".join(r["name"] for r in resolved)]

    # Best-effort IMAP email context — errors caught by caller
    cap = load_capabilities().get("imap")
    if cap and cap.is_connected():
        fn = next((t["handler"] for t in cap.get_tools()
                   if t["name"] == "imap_search_email"), None)
        if fn:
            since = (now - timedelta(days=_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
            lines = []
            needs_reply = []
            for p in resolved:
                params = {"sender": p["email"],
                          "date_from": since, "limit": 3}
                hits = fn(None, params).get("emails", [])
                for e in hits:
                    subj = e.get("subject", "?")
                    lines.append(f"- From {p['name']}: \"{subj}\"")

                # Unanswered emails from this attendee
                unans = fn(None, {"sender": p["email"], "date_from": since,
                                  "unanswered": True, "limit": 3}
                           ).get("emails", [])
                for e in unans:
                    subj = e.get("subject", "?")
                    needs_reply.append(
                        f"- {p['name']}: \"{subj}\" ({e.get('date', '')[:10]})")

            if lines:
                parts.append("\nRecent emails:\n" + "\n".join(lines))
            if needs_reply:
                parts.append(
                    "\nNeeds reply (you haven't responded):\n"
                    + "\n".join(needs_reply))
    return "\n".join(parts)


from capabilities.base import register_hook
register_hook(maybe_send_meeting_prep)
