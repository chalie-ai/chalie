"""Draft reply nudge — detect emails idle >4 hours without a reply.

Fires once per day (deduped by date).  Scans for unanswered actionable
emails that arrived more than ``_IDLE_HOURS`` ago and pushes a nudge
to ``prompt-queue`` so Chalie can remind the user.

Called from :meth:`~capabilities.base.AbstractCapability.run_monitor`.
"""

import json
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)

_FLAG_PREFIX = "draft_nudge:"
_FLAG_TTL = 24 * 3600
_IDLE_HOURS = 4
_MAX_EMAILS = 5


def maybe_send_draft_nudge(now=None) -> bool:
    """Push a once-per-day draft-reply nudge via prompt-queue.

    Only surfaces actionable emails older than ``_IDLE_HOURS``.
    Deduped per UTC date.  Respects quiet window.
    """
    try:
        from capabilities.quiet_window import is_quiet_now
        from services.memory_client import MemoryClientService
        from services.time_utils import utc_now

        now = now or utc_now()
        if is_quiet_now(now):
            return False

        flag_key = f"{_FLAG_PREFIX}{now.strftime('%Y-%m-%d')}"
        from capabilities.hook_dedup import is_fired
        if is_fired(flag_key):
            return False

        stale = _find_stale_emails(now)
        if not stale:
            return False

        store = MemoryClientService.create_connection()
        lines = [
            f"- {e['from']}: \"{e['subject']}\" (idle {e['hours']}h)"
            for e in stale
        ]
        body = (
            f"Emails awaiting your reply ({len(stale)}):\n"
            + "\n".join(lines)
        )
        store.rpush("prompt-queue", json.dumps({
            "prompt": (
                f"[DRAFT NUDGE]\n{body}\n\n"
                "The user has emails that may need a reply. "
                "Mention each briefly with sender and subject. "
                "Offer to draft a reply for any of them. "
                "Keep it casual — 2-4 sentences. "
                "Do NOT mention 'IMAP' or technical details."
            ),
            "metadata": {
                "type": "proactive_drift",
                "source": "draft_nudge",
                "topic": "proactive",
            },
        }))
        from capabilities.hook_dedup import mark_fired
        mark_fired(flag_key, _FLAG_TTL)
        return True
    except Exception as exc:
        logger.debug("[draft_nudge] failed: %s", exc)
        return False


def _find_stale_emails(now) -> list[dict]:
    """Return actionable unanswered emails older than _IDLE_HOURS."""
    from capabilities import load_capabilities
    from services.time_utils import parse_utc

    cap = load_capabilities().get("imap")
    if not cap or not cap.is_connected():
        return []
    tools = {t["name"]: t["handler"] for t in cap.get_tools()}
    search_fn = tools.get("imap_search_email")
    if not search_fn:
        return []

    cutoff = now - timedelta(hours=_IDLE_HOURS)
    since = (now - timedelta(days=2)).strftime("%Y-%m-%d")
    from capabilities.reply_sentinel import filter_replied
    emails = filter_replied(search_fn(
        None,
        {"date_from": since, "unanswered": True,
         "triage": "actionable", "limit": _MAX_EMAILS * 2},
    ).get("emails", []))

    stale = []
    for e in emails:
        date_str = e.get("date")
        if not date_str or parse_utc(date_str) >= cutoff:
            continue
        hours = int((now - parse_utc(date_str)).total_seconds() / 3600)
        stale.append({
            "from": e.get("from_name") or e.get("from_addr", "?"),
            "subject": e.get("subject", "(no subject)"),
            "hours": hours,
        })
        if len(stale) >= _MAX_EMAILS:
            break
    return stale


from capabilities.base import register_hook  # noqa: E402
register_hook(maybe_send_draft_nudge)
