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

from configs.channels import ThreadGistConfig
from models.thread_gist import ThreadGist
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


def generate_gist(trigger_channel: str, trigger_turn_id: int, text: str = "") -> str | None:
    """Build + run the gist delegate MP for one thread; return its label or None.

    The trigger thread's identity travels on ``metadata`` (``trigger_channel``/
    ``trigger_turn_id``) — the controller lifts it onto ``mp._trigger_channel``/
    ``mp._trigger_turn_id`` at construction, which is what ``PromptService``
    reads to assemble this delegate's user prompt from the DB. A caller that
    already holds the text — a schedule's prompt, which has no transcript to
    read — passes it as ``text`` and it becomes the prompt body directly.

    Think blocks are stripped here, the one place a raw completion turns into a
    user-facing label, so no provider path can leak chain-of-thought into it. A
    label that reduces to empty — an unclosed think block swallows everything
    after its opener, or the model replied with whitespace alone — is dropped
    loudly rather than returned as a storable label.

    The empty answers are told apart: ``""`` means the delegate answered but with
    nothing usable, ``None`` that it never answered at all (a crashed turn leaves
    no result text). Only the second is worth retrying, which is what lets a
    caller settle a hopeless prompt instead of re-firing it every poll."""
    from controllers.message_processor import MessageProcessor

    mp = MessageProcessor.process(
        ThreadGistConfig(),
        raw_input=text,
        metadata={"trigger_channel": trigger_channel, "trigger_turn_id": trigger_turn_id},
    )
    raw = mp.result()
    label = _strip_think_blocks(raw).strip()
    if not label:
        logger.warning(
            "%s no usable label for %s turn=%s: %r",
            _LOG_PREFIX, trigger_channel, trigger_turn_id, raw[:120],
        )
        return "" if raw else None
    return label


def _run_gist_processor(
    trigger_channel: str, trigger_turn_id: int, trigger_type: str | None
) -> None:
    try:
        gist = generate_gist(trigger_channel, trigger_turn_id)
        if gist:
            ThreadGist(channel=trigger_channel, turn_id=trigger_turn_id, gist=gist).upsert()
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
