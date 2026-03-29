"""First look — one-time cross-capability welcome briefing after onboarding.

Fires once when 2+ capabilities are connected and have data.  Reads
calendar events from ``scheduled_items`` and inbox state from world-state
cache, then pushes a synthesis prompt to ``prompt-queue``.
"""

import json
import logging
from datetime import timedelta

from services.time_utils import utc_now

logger = logging.getLogger(__name__)

_FLAG_KEY = "first_look:sent"
_FLAG_TTL = 365 * 24 * 3600
_MAX_EVENTS = 8


def maybe_send_first_look() -> bool:
    """Send a one-time cross-capability welcome briefing.

    Returns True if the briefing was enqueued, False if skipped.
    """
    try:
        from capabilities.hook_dedup import is_fired

        if is_fired(_FLAG_KEY):
            return False

        from capabilities import load_capabilities

        caps = load_capabilities()
        connected = sum(c.is_connected() for c in caps.values())
        if connected < 2:
            return False

        from services.memory_client import MemoryClientService
        store = MemoryClientService.create_connection()

        cal = _build_calendar_snapshot()
        email = _read_inbox_snapshot(store)
        sections = list(filter(None, (cal, email)))
        if not sections:
            return False

        body = "\n\n".join(sections)
        store.rpush("prompt-queue", json.dumps({
            "prompt": (
                "[FIRST LOOK — Welcome Briefing]\n" + body + "\n\n"
                "This is the user's FIRST interaction after connecting their "
                "calendar and email. Deliver a concise, impressive opening "
                "briefing showing you understand their life. Cross-reference "
                "people across calendar and email where possible. Surface "
                "conflicts, important emails, and what needs attention today. "
                "Be warm, specific, and brief — 4-6 sentences max. "
                "Do NOT mention technical source names like 'CalDAV' or 'IMAP'."
            ),
            "metadata": {"type": "proactive_drift", "source": "first_look", "topic": "proactive"},
        }))
        from capabilities.hook_dedup import mark_fired
        mark_fired(_FLAG_KEY, _FLAG_TTL)
        logger.info("[first_look] Enqueued cross-capability welcome briefing.")
        return True

    except Exception as exc:
        logger.debug("[first_look] Failed: %s", exc)
        return False


def _build_calendar_snapshot() -> str:
    """Read next-48h events from scheduled_items."""
    try:
        from services.database_service import get_shared_db_service

        db = get_shared_db_service()
        now = utc_now()
        end = now + timedelta(hours=48)

        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT message, due_at, metadata FROM scheduled_items "
                "WHERE source='caldav' AND item_type='event' AND status='pending' "
                "AND due_at >= ? AND due_at <= ? ORDER BY due_at ASC LIMIT ?",
                (now.isoformat(), end.isoformat(), _MAX_EVENTS),
            )
            rows = cursor.fetchall()

        if not rows:
            return ""

        lines = [f"- {due[:16]}: {msg}" for msg, due, _ in rows]
        return f"Calendar ({len(rows)} upcoming):\n" + "\n".join(lines)
    except Exception as exc:
        logger.debug("[first_look] calendar snapshot failed: %s", exc)
        return ""


def _read_inbox_snapshot(store) -> str:
    """Read the cached IMAP inbox hint from world state."""
    try:
        for raw in reversed(store.lrange("world_state:external_signals", 0, -1)):
            sig = json.loads(raw)
            if sig.get("source") == "imap":
                return sig.get("content", "")
    except Exception:
        pass
    return ""


from capabilities.base import register_hook
register_hook(maybe_send_first_look)
