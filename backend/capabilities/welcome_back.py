"""Welcome-back brief — catch-up digest when user returns after absence.

Detects when the user has been away for 1+ hours (via
``proactive:default:last_interaction_ts`` in MemoryStore) and pushes a
catch-up to ``prompt-queue`` using already-cached world-state hints.

Called from :meth:`~capabilities.base.AbstractCapability.run_monitor`.
"""

import json
import logging
import time

logger = logging.getLogger(__name__)

_INTERACTION_KEY = "proactive:default:last_interaction_ts"
_FLAG_KEY = "welcome_back:sent"
_FLAG_TTL = 8 * 3600
_ABSENCE_THRESHOLD = 3600


def maybe_send_welcome_back(now=None) -> bool:
    """Push a welcome-back brief if user has been absent 1+ hours."""
    try:
        from capabilities.quiet_window import is_quiet_now
        from services.memory_client import MemoryClientService
        from services.time_utils import utc_now

        now = now or utc_now()
        if is_quiet_now(now):
            return False

        from capabilities.hook_dedup import is_fired
        if is_fired(_FLAG_KEY):
            return False

        store = MemoryClientService.create_connection()

        raw = store.get(_INTERACTION_KEY)
        if not raw:
            return False
        gap = time.time() - float(raw)
        if gap < _ABSENCE_THRESHOLD:
            return False

        hints = _read_world_hints(store)
        if not hints:
            return False

        away = f"{max(1, int(gap / 3600))}h"
        store.rpush("prompt-queue", json.dumps({
            "prompt": (
                f"[WELCOME BACK — absent ~{away}]\n{hints}\n\n"
                "The user is returning after being away. "
                "Give a concise catch-up: what's coming up, what needs "
                "attention. Warm and brief — 3-5 sentences max. "
                "Do NOT mention 'CalDAV' or 'IMAP'."
            ),
            "metadata": {
                "type": "proactive_drift",
                "source": "welcome_back",
                "topic": "proactive",
            },
        }))
        from capabilities.hook_dedup import mark_fired
        mark_fired(_FLAG_KEY, _FLAG_TTL)
        return True
    except Exception as exc:
        logger.debug("[welcome_back] failed: %s", exc)
        return False


def _read_world_hints(store) -> str:
    """Read cached capability hints from WorldState MemoryStore."""
    signals = store.lrange("world_state:external_signals", 0, -1)
    return "\n\n".join(
        filter(None, (json.loads(r).get("content", "") for r in signals))
    )


from capabilities.base import register_hook
register_hook(maybe_send_welcome_back)
