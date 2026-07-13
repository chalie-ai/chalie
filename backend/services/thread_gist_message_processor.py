"""Thread-gist daemon fire — launches a delegate MP that gists one thread.

Mirrors ``skill_suggestion_message_processor``: a fire-and-forget daemon thread
that runs the gist delegate through the sanctioned entry
(``MessageProcessor.process(...).result()``), tagging the run with the trigger
thread's ``(channel, turn_id)`` via ``metadata`` so ``PromptService`` can read
the thread's opening messages from the DB, then upserts the resulting gist.
No carried state across calls.

After a successful upsert, broadcasts a ``turn_signal {status: 'updated'}``
frame on the trigger channel so the frontend spine refetches the thread's
collapsed block and renders the fresh label without the user having to open
the thread.
"""

from __future__ import annotations

import logging
import threading

from models.turn_signal import TurnSignal
from services.llm_service import _strip_think_blocks
from services.websocket import Websocket

logger = logging.getLogger(__name__)
_LOG_PREFIX = "[THREAD GIST]"


def maybe_ingest_gist(
    trigger_channel: str, trigger_turn_id: int, trigger_type: str | None
) -> None:
    """Fire the gist delegate MP for one thread (daemon, non-blocking).

    ``trigger_channel`` is the DB storage key (channel value, e.g. ``schedule``);
    ``trigger_type`` is the trigger thread's frontend routing identity (config-type
    value, e.g. ``scheduled``, or ``None`` for an internal channel). The two are NOT
    interchangeable — deriving one from the other is what the caller passes both."""
    if trigger_channel is None or trigger_turn_id is None:
        return
    logger.info("%s firing for %s turn=%s", _LOG_PREFIX, trigger_channel, trigger_turn_id)
    t = threading.Thread(
        target=_run_gist_processor,
        args=(trigger_channel, trigger_turn_id, trigger_type),
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


def _run_gist_processor(
    trigger_channel: str, trigger_turn_id: int, trigger_type: str | None
) -> None:
    try:
        gist = generate_gist(trigger_channel, trigger_turn_id)
        if gist:
            from configs.channels import ThreadGistConfig
            from controllers.message_processor import MessageProcessor

            # Inert construction only (I2) — no .process(), purely to reach
            # gist_service for the upsert. Think-stripping today only happens in
            # the OpenAI provider client, so non-OpenAI providers (or an
            # unclosed think block) would otherwise leak <think>...</think> into
            # this stored, user-facing label — strip here, at the single point
            # the gist becomes persisted data.
            inert = MessageProcessor(ThreadGistConfig())
            inert.gist_service.upsert(trigger_channel, trigger_turn_id, _strip_think_blocks(gist))
            _broadcast_updated(trigger_turn_id, trigger_type)
    except Exception as exc:
        logger.warning("%s processor failed: %s", _LOG_PREFIX, exc)


def _broadcast_updated(trigger_turn_id: int, trigger_type: str | None) -> None:
    """Emit a ``turn_signal {status: 'updated'}`` frame carrying the trigger
    thread's routing identity so the frontend spine refetches its collapsed block.

    The gist delegate MP is internal (``ThreadGistConfig``), which has no
    ``BROADCASTS_STATE`` and no ``type_value`` — it would silently drop any frame
    through ``mp.push_websocket``. So the caller hands down the *trigger* thread's
    own ``type_value()`` (the routing type the user's surface listens on) and we
    build the frame with it, straight to ``Websocket.broadcast``. ``None`` means an
    internal channel with no addressable frontend — nothing to poke."""
    if trigger_type is None:
        return
    frame = TurnSignal(
        status="updated",
        turn_id=trigger_turn_id,
        type=trigger_type,
    )
    Websocket.broadcast(frame)
