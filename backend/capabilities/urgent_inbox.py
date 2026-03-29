"""Urgent inbox hook — real-time alert for high-priority inbound emails.

Fires when a recent actionable email (< 15 min old) arrives from a known
contact.  Pushes an immediate heads-up to ``prompt-queue`` so Chalie can
surface it without waiting for the next morning/evening brief or draft nudge.

Deduped by message UID (24 h TTL) so each email only alerts once.
Respects quiet window.

Called from :meth:`~capabilities.base.AbstractCapability._dispatch_hooks`.
"""

import json
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)

_FLAG_PREFIX = "urgent_inbox:"
_FLAG_TTL = 24 * 3600
_RECENT_MINUTES = 15
_MAX_ALERTS = 3


def maybe_send_urgent_inbox(now=None) -> bool:
    """Push an alert for each new high-priority email from a known contact.

    Only surfaces actionable emails that arrived in the last
    ``_RECENT_MINUTES`` from senders already in the contact index.
    Returns True if at least one alert was enqueued.
    """
    from capabilities.quiet_window import is_quiet_now
    from services.memory_client import MemoryClientService
    from services.time_utils import utc_now

    if now is None:
        now = utc_now()
    if is_quiet_now(now):
        return False

    store = MemoryClientService.create_connection()
    urgent = _find_urgent_emails(now, store)
    if not urgent:
        return False

    from capabilities.hook_dedup import mark_fired

    for email in urgent:
        store.rpush("prompt-queue", json.dumps({
            "prompt": (
                f"[URGENT EMAIL]\n"
                f"From: {email['from_name']} ({email['from_addr']})\n"
                f"Subject: {email['subject']}\n\n"
                "A new email just arrived from a known contact. "
                "Mention who it's from and the subject briefly. "
                "Offer to read it or draft a reply. "
                "Keep it casual — 1-2 sentences. "
                "Do NOT mention 'IMAP' or technical details."
            ),
            "metadata": {
                "type": "proactive_drift",
                "source": "urgent_inbox",
                "topic": "proactive",
                "uid": email["uid"],
            },
        }))
        mark_fired(f"{_FLAG_PREFIX}{email['uid']}", _FLAG_TTL)
    return True


def _find_urgent_emails(now, store) -> list[dict]:
    """Return recent actionable emails from known contacts, not yet alerted."""
    from capabilities import load_capabilities
    from capabilities.contact_resolver import resolve
    from services.time_utils import parse_utc

    cap = load_capabilities().get("imap")
    if not cap or not cap.is_connected():
        return []
    tools = {t["name"]: t["handler"] for t in cap.get_tools()}
    search_fn = tools.get("imap_search_email")
    if not search_fn:
        return []

    since = (now - timedelta(days=1)).strftime("%Y-%m-%d")
    emails = search_fn(
        None,
        {"date_from": since, "unseen": True,
         "triage": "actionable", "limit": 20},
    ).get("emails", [])

    cutoff = now - timedelta(minutes=_RECENT_MINUTES)
    urgent = []
    for e in emails:
        uid = e.get("uid")
        if not uid:
            continue
        date_str = e.get("date", "")
        if not date_str or parse_utc(date_str) < cutoff:
            continue
        from capabilities.hook_dedup import is_fired
        if is_fired(f"{_FLAG_PREFIX}{uid}"):
            continue
        addr = e.get("from_addr", "").strip().lower()
        if not resolve(addr, limit=1):
            continue
        urgent.append({
            "uid": uid,
            "from_name": e.get("from_name", addr),
            "from_addr": addr,
            "subject": e.get("subject", "(no subject)"),
        })
    return urgent[:_MAX_ALERTS]


from capabilities.base import register_hook  # noqa: E402
register_hook(maybe_send_urgent_inbox)
