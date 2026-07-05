"""Thread-gist daemon fire — launches a delegate MP that gists one thread.

Mirrors ``skill_suggestion_message_processor``: a fire-and-forget daemon thread
that runs the gist delegate through the sanctioned entry
(``MessageProcessor.process(...).result()``), tagging the run with the trigger
thread's ``(channel, turn_id)`` via ``metadata`` so ``PromptService`` can read
the thread's opening messages from the DB, then upserts the resulting gist.
No carried state across calls.
"""

from __future__ import annotations

import logging
import threading

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

    The trigger thread's identity travels on ``metadata`` (``trigger_channel``/
    ``trigger_turn_id``) — the controller lifts it onto ``mp._trigger_channel``/
    ``mp._trigger_turn_id`` at construction, which is what ``PromptService``
    reads to assemble this delegate's user prompt from the DB."""
    from configs.channels import ThreadGistConfig
    from controllers.message_processor import MessageProcessor

    mp = MessageProcessor.process(
        ThreadGistConfig(),
        metadata={"trigger_channel": trigger_channel, "trigger_turn_id": trigger_turn_id},
    )
    return mp.result() or None


def _run_gist_processor(trigger_channel: str, trigger_turn_id: int) -> None:
    try:
        gist = generate_gist(trigger_channel, trigger_turn_id)
        if gist:
            from configs.channels import ThreadGistConfig
            from controllers.message_processor import MessageProcessor

            # Inert construction only (I2) — no .process(), purely to reach
            # gist_service for the upsert.
            inert = MessageProcessor(ThreadGistConfig())
            inert.gist_service.upsert(trigger_channel, trigger_turn_id, gist)
    except Exception as exc:
        logger.warning("%s processor failed: %s", _LOG_PREFIX, exc)
