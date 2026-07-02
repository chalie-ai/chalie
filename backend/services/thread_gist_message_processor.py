"""Thread-gist daemon fire — launches a delegate MP that gists one thread.

Mirrors ``skill_suggestion_message_processor``: a fire-and-forget daemon thread
that builds an MP via ``object.__new__``, assigns ``ThreadGistConfig``, sets the
trigger context (``_trigger_channel`` / ``_trigger_turn_id``) so the config's
``get_user_prompt`` can read the thread's user messages from the DB, runs the
turn, and upserts the resulting gist. No carried state across calls.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from typing import Protocol

    class _TriggerCtx(Protocol):  # noqa: D100
        _trigger_channel: str
        _trigger_turn_id: int

logger = logging.getLogger(__name__)
_LOG_PREFIX = "[THREAD GIST]"


def maybe_ingest_gist(trigger_channel: str, trigger_turn_id: int) -> None:
    """Fire the gist delegate MP for one thread (daemon, non-blocking)."""
    if trigger_channel is None or trigger_turn_id is None:
        return
    logger.info("%s firing for %s turn=%s", _LOG_PREFIX, trigger_channel, trigger_turn_id)
    t = threading.Thread(
        target=_run_gist_processor,
        args=(trigger_channel, trigger_turn_id),
        daemon=True,
        name="thread-gist",
    )
    t.start()


def generate_gist(trigger_channel: str, trigger_turn_id: int) -> str | None:
    """Build + run the gist delegate MP for one thread; return its label or None.

    The single construction site for the gist delegate — every caller persists
    the returned label into ``thread_gist`` (keyed by channel + turn_id)."""
    from configs.channels import ThreadGistConfig
    from services.message_processor import MessageProcessor

    mp = object.__new__(MessageProcessor)
    MessageProcessor.__init__(mp, ThreadGistConfig())
    mp.uid = None
    mp.cancel_event = threading.Event()
    mp.thinking_level = "low"
    cast("_TriggerCtx", mp)._trigger_channel = trigger_channel
    cast("_TriggerCtx", mp)._trigger_turn_id = trigger_turn_id
    return mp._run() or None


def _run_gist_processor(trigger_channel: str, trigger_turn_id: int) -> None:
    try:
        from services.thread_gist_service import get_thread_gist_service
        gist = generate_gist(trigger_channel, trigger_turn_id)
        if gist:
            get_thread_gist_service().upsert(trigger_channel, trigger_turn_id, gist)
    except Exception as exc:
        logger.warning("%s processor failed: %s", _LOG_PREFIX, exc)
