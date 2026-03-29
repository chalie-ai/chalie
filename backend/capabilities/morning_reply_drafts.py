"""Morning reply drafts — pre-generate candidate replies for actionable emails.

Fires once per day during the morning window (06:00-10:00 local), after
the morning brief.  Searches for actionable unanswered emails from the
last 12 hours, formats them, and pushes a prompt asking the LLM to draft
one-sentence candidate replies the user can approve with "send 1 and 3".

This is the first "act" hook — previous hooks inform, this one *does work*
for the user.

Called from :meth:`~capabilities.base.AbstractCapability._dispatch_hooks`.
"""

import json
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)

_FLAG_PREFIX = "morning_reply_drafts:"
_FLAG_TTL = 24 * 3600
_MAX_EMAILS = 5
_LOOKBACK_HOURS = 12
_MORNING_START = 6   # local hour (inclusive)
_MORNING_END = 10    # local hour (exclusive)


def maybe_send_morning_reply_drafts(now=None) -> bool:
    """Push once-per-day draft replies for actionable morning emails.

    Finds unanswered actionable emails from the last 12 hours, formats
    them with sender + subject, and asks the LLM to draft one-sentence
    candidate replies.  Deduped per UTC date.  Respects quiet window.
    """
    try:
        from capabilities.time_context import get_user_tz as _user_tz
        from capabilities.quiet_window import is_quiet_now
        from services.memory_client import MemoryClientService
        from services.time_utils import utc_now

        now = now or utc_now()
        if is_quiet_now(now):
            return False

        local_hour = now.astimezone(_user_tz()).hour
        if not (_MORNING_START <= local_hour < _MORNING_END):
            return False

        flag_key = f"{_FLAG_PREFIX}{now.strftime('%Y-%m-%d')}"
        from capabilities.hook_dedup import is_fired
        if is_fired(flag_key):
            return False

        emails = _find_actionable_emails(now)
        if not emails:
            return False

        lines = [
            f"{i}. From: {e['from']} — \"{e['subject']}\""
            for i, e in enumerate(emails, 1)
        ]
        body = "\n".join(lines)

        # Persist index→UID mapping so "send 1" can resolve to a real email
        draft_map = [
            {"index": i, "uid": e["uid"], "from_addr": e["from_addr"],
             "subject": e["subject"], "message_id": e.get("message_id", "")}
            for i, e in enumerate(emails, 1)
        ]
        store = MemoryClientService.create_connection()
        map_key = f"morning_drafts:{now.strftime('%Y-%m-%d')}"
        store.set(map_key, json.dumps(draft_map), ex=_FLAG_TTL)

        uid_ref = " | ".join(
            f"#{d['index']}=UID:{d['uid']} to:{d['from_addr']} "
            f"subj:Re: {d['subject']} msgid:{d.get('message_id', '')}"
            for d in draft_map
        )
        store.rpush("prompt-queue", json.dumps({
            "prompt": (
                f"[MORNING REPLY DRAFTS]\n"
                f"These {len(emails)} emails arrived recently and may "
                f"need a reply:\n{body}\n\n"
                "For each email, draft a one-sentence candidate reply "
                "the user can approve and send. Numbered list. "
                "Professional but concise — one sentence each. "
                "Tell the user they can say \"send 1 and 3\" to dispatch "
                "or \"edit 2\" to refine one. "
                "Do NOT mention 'IMAP' or technical details.\n\n"
                f"[INTERNAL — draft mapping: {uid_ref}. "
                "When the user says 'send N', call imap_send_email with "
                "to=<from_addr>, subject=<Re: subject>, body=<your draft>, "
                "in_reply_to=<msgid> from the mapping above. This sends "
                "the reply directly — no Drafts folder detour.]"
            ),
            "metadata": {
                "type": "proactive_drift",
                "source": "morning_reply_drafts",
                "topic": "proactive",
            },
        }))
        from capabilities.hook_dedup import mark_fired
        mark_fired(flag_key, _FLAG_TTL)
        logger.info("[morning_reply_drafts] Enqueued %d drafts.", len(emails))
        return True
    except Exception as exc:
        logger.debug("[morning_reply_drafts] failed: %s", exc)
        return False


def _find_actionable_emails(now) -> list[dict]:
    """Return actionable unanswered emails from the last _LOOKBACK_HOURS."""
    from capabilities import load_capabilities
    from services.time_utils import parse_utc

    try:
        cap = load_capabilities()["imap"]
        if not cap.is_connected():
            return []
        search_fn = {t["name"]: t["handler"]
                     for t in cap.get_tools()}["imap_search_email"]
    except (KeyError, AttributeError):
        return []

    cutoff = now - timedelta(hours=_LOOKBACK_HOURS)
    since = (now - timedelta(days=2)).strftime("%Y-%m-%d")
    from capabilities.reply_sentinel import filter_replied
    raw = filter_replied(search_fn(
        None,
        {"date_from": since, "unanswered": True,
         "triage": "actionable", "limit": _MAX_EMAILS},
    ).get("emails", []))

    return [
        {"uid": e.get("uid"),
         "from": e.get("from_name", e.get("from_addr", "?")),
         "from_addr": e.get("from_addr", ""),
         "subject": e.get("subject", "(no subject)"),
         "message_id": e.get("message_id", "")}
        for e in raw
        if e.get("date") and parse_utc(e["date"]) >= cutoff
    ][:_MAX_EMAILS]


from capabilities.base import register_hook
register_hook(maybe_send_morning_reply_drafts)
