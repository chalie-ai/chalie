"""Reply sentinel — prevent hooks from re-surfacing emails Chalie replied to.

When Chalie sends a reply via SMTP, the IMAP server does NOT set the
ANSWERED flag on the original email (SMTP is a separate path).  Without
this guard, ``unanswered=True`` IMAP queries permanently return emails
Chalie already handled — causing trust-destroying re-nagging.

Uses ``hook_dedup`` for persistent storage (SQLite + MemoryStore cache).
"""

import logging

logger = logging.getLogger(__name__)

_KEY_PREFIX = "replied_to:"
_TTL_SECONDS = 48 * 3600  # 48 hours


def mark_replied(message_id: str) -> None:
    """Record that Chalie replied to an email with the given Message-ID."""
    from capabilities.hook_dedup import mark_fired
    mid = message_id or ""
    mark_fired(f"{_KEY_PREFIX}{mid}", _TTL_SECONDS)
    logger.debug("[reply_sentinel] marked replied: %s", mid)


def is_replied(message_id: str) -> bool:
    """Check if Chalie has already replied to the given Message-ID."""
    if not message_id:
        return False
    from capabilities.hook_dedup import is_fired
    return is_fired(f"{_KEY_PREFIX}{message_id}")


def filter_replied(emails: list[dict]) -> list[dict]:
    """Remove emails Chalie has already replied to.

    Each email dict should have a ``message_id`` key.
    Emails without ``message_id`` pass through unchanged.
    """
    return [e for e in emails if not is_replied(e.get("message_id", ""))]
