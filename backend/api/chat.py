"""
Chat API — HTTP turn-control and rich-card action endpoints.

Routes:
  POST /api/action             — receive a rich-card control click; dispatches via
                             ToolDispatcher.dispatch() and returns the result
                             synchronously in the response body (no WS frame).

  Async-delegate (subagent) lifecycle lives in its own resource group —
  GET /api/subagents and DELETE /api/subagent/<sub_id> (see api/subagents.py).

Send path:
  User messages are sent via POST /api/thread (new thread) or
  POST /api/thread/<turn_id> (reply) — see api/threads.py. Both return 200 with
  {turn_id, channel}: the turn_id is allocated synchronously (atomic input-row
  write) before the response, so the FE holds the handle without waiting on a WS
  signal. Live output still surfaces via WS signals → REST pull.

Design:
  WS is receive-only push (server→client). All client→server requests use HTTP.
  turn_id is the only handle: each send spawns an isolated turn on its own daemon
  thread; the MessageProcessor allocates (new spine turn) or reuses (thread reply)
  a turn_id and carries it end to end — every WS signal stamps it, so the surface
  keys its updates on it and holds no shared "active turn" state. A second message
  to a busy surface is never sent mid-turn: the frontend QUEUES it per turn_id and
  dispatches once that turn's ``done`` arrives. Interrupt is the lone exception —
  DELETE /api/thread/<turn_id> (see api/threads.py) cancels a running turn by
  turn_id, which self-cleans its rows.

  User-channel messages flow through MessageProcessor.process() with the
  UserConfig ProcessorConfig subclass — no MessageProcessor subclass. Live
  output is signal-only, gated by the config's BROADCASTS_STATE: the surface
  holds no event memory and refetches turn blocks over REST. Every lifecycle
  signal (working/updated/done + the per-tool tool_called/tool_done timers) is
  emitted by MessageProcessor itself through its `broadcast` chokepoint,
  including the terminal ``done`` on a FAILED turn (fired from the
  live instance — it owns turn_id). The only signal originating here is the
  channel-wide error toast that failed turn also needs.
"""

import logging
import threading
import time
import uuid
from typing import Sequence

from flask.typing import ResponseReturnValue
from flask_restx import Namespace, Resource

from .auth import require_auth
from .dto import Error, expects, register_dto, responds
from .dto.boundary import error
from .dto.chat import ActionRequest, ActionResult
from services.markup import sanitize
from services.processor_config import ProcessorConfig
from services.websocket_broker import WebSocketBroker

logger = logging.getLogger(__name__)

chat_ns = Namespace("chat", description="Chat turn control and rich-card actions", path="/api")

register_dto(chat_ns, ActionRequest, ActionResult, Error)

_M = chat_ns.models


# ── Background helpers ────────────────────────────────────────────────────────


def _run_chat_background(
    cancel_event: threading.Event,
    raw_input: str,
    config: "ProcessorConfig",
    metadata: dict[str, object],
    request_id: str,
) -> None:
    """Runs the user turn to completion on its own daemon thread. The MessageProcessor
    owns the whole lifecycle (working → updated → done, plus the per-tool timers) and
    adopts the request thread's pre-written input row (turn_id already allocated), so
    it needs nothing from here once started — including the terminal ``done`` on a
    FAILED turn, which the MP
    fires from its live instance (it knows turn_id) before the exception reaches us. We
    add only the channel-wide error toast; a cancelled turn (stop button) stays silent."""
    from services.message_processor import MessageProcessor  # noqa: PLC0415

    try:
        MessageProcessor.process(raw_input, config, metadata, cancel_event=cancel_event)
    except Exception as exc:
        logger.exception("[Chat API] turn error for %s: %s", request_id, exc)
        if not cancel_event.is_set():
            WebSocketBroker().broadcast({"type": "error", "message": "Turn failed unexpectedly", "recoverable": False})


def deliver_async_result(mp: object, result_text: str, cancel_event: threading.Event) -> None:
    """Appends another assistant turn for a finished async delegate — a plain
    MessageProcessor.process() call, independent of any foreground turn (each runs
    on its own thread, keyed by its own turn_id).

    The delegate's ``cancel_event`` is threaded into the synthesis turn so the
    Processes-panel stop control aborts a spiralling delegate at the next chain
    boundary (the processor returns "" and self-cleans when it is set).

    Inherits the originating turn's ``thread_id`` so the synthesised reply lands
    in the same thread the delegate was spawned from.
    """
    from services.message_processor import MessageProcessor  # noqa: PLC0415

    config = getattr(mp, "config", None)
    if config is None:
        logger.warning("[Chat API] async delivery skipped: captured mp has no config")
        return

    synth_config = config.with_hidden_input()
    # Clone the originating metadata but suppress the input row and drop
    # attachments — they were already ingested on the originating turn and must
    # not re-upload on the synthesis turn. Inherit the originating thread id so
    # the synthesised reply lands in the same thread.
    metadata = dict(getattr(mp, "_metadata", None) or {})
    metadata["hidden_input"] = True
    metadata["attachments"] = []
    # A delegate→main continuation appends to the originating turn but is NOT a
    # genuine user fork reply — drop any inherited is_thread_reply so it reads the
    # MAIN spine, never the FORK view.
    metadata.pop("is_thread_reply", None)
    thread_id = getattr(mp, "turn_id", None)
    if thread_id is not None:
        metadata["thread_id"] = thread_id
    # The synthesis turn is a full UserConfig turn, so its created → … → done
    # lifecycle signals come from MessageProcessor itself — nothing to emit here.
    MessageProcessor.process(result_text, synth_config, metadata, cancel_event=cancel_event)


def dispatch_message(
    text: str,
    source: str = "text",
    attachments: list[object] | None = None,
    hidden_input: bool = False,
    channel: str = "user",
    thread_id: "int | None" = None,
) -> "int | None":
    """Spawn an isolated turn for ``text`` and return its ``turn_id``. ``thread_id``
    appends to an existing thread (its input row carries that turn_id so all the
    reply's rows share it); absent, a fresh turn_id is allocated synchronously
    here so the caller can return it inline. There is no cross-surface state — the
    frontend never sends into a busy surface (it queues per turn_id), so a turn
    never has to cancel or combine with another. ``hidden_input`` turns (async
    delivery, disclosures) write no input row and return their inherited thread_id
    (or None).
    """
    return _start_turn(
        text, source, attachments or [], hidden_input, channel=channel, thread_id=thread_id,
    )


def _start_turn(
    text: str, source: str, attachments: list[object], hidden_input: bool = False,
    *, channel: str = "user", thread_id: "int | None" = None,
) -> "int | None":
    from configs.channels import ScheduledConfig, UserConfig  # noqa: PLC0415

    request_id = str(uuid.uuid4())

    try:
        from services.world_state import world_state, Signal  # noqa: PLC0415
        world_state.absorb(Signal(source="http_chat", kind="user_message", payload={"text": text[:200]}))
    except Exception as exc:
        logger.debug("[Chat API] world_state.absorb failed: %s", exc)

    metadata: dict[str, object] = {
        "uuid": request_id,
        "exchange_id": request_id,
        "source": source,
        "attachments": attachments,
        "channel": channel,
        "hidden_input": hidden_input,
    }
    if thread_id is not None:
        metadata["thread_id"] = thread_id
        # Genuine user reply INTO a thread → FORK view. This is
        # the ONLY producer of is_thread_reply; the async-delivery path (which
        # reuses thread_id for a delegate→main continuation) explicitly drops it.
        metadata["is_thread_reply"] = True

    # A reply INTO a schedule thread (channel=schedule) runs under ScheduledConfig
    # so it appends to that schedule's growing thread on its own channel (§13.5);
    # every other turn is an ordinary user-spine turn.
    config = ScheduledConfig() if channel == "schedule" else UserConfig(metadata)
    # We keep a reference to this event only so the error path can stay silent on
    # a cancelled turn; it is also registered by turn_id below for the stop button.
    cancel_event = threading.Event()

    # Allocate the turn_id SYNCHRONOUSLY in the request thread: write the input row
    # now (atomic COALESCE under the writer lock — never max+1 in Python), so the
    # POST can return {turn_id, channel} before the worker even acquires the lane
    # lock. The worker's _setup ADOPTS this row via metadata["input_uid"] instead
    # of writing a second one. A reply already knows its turn_id from the path.
    # Register the cancel handle here too, so DELETE /api/thread/<turn_id> is live
    # the instant the FE holds the id. hidden_input turns write no row (skip).
    landed: "int | None" = thread_id
    if not hidden_input:
        from services.transcript_service import Transcript  # noqa: PLC0415
        pre_uid = Transcript.write_input_row(channel, "user", text, turn_id=thread_id)
        metadata["input_uid"] = pre_uid
        if landed is None:
            landed = Transcript.turn_id_of_row(pre_uid)
        if landed is not None:
            from services.message_processor import _register_canceller  # noqa: PLC0415
            _register_canceller(landed, cancel_event)

    thread = threading.Thread(
        target=_run_chat_background,
        args=(cancel_event, text, config, metadata, request_id),
        daemon=True,
        name=f"chat-{request_id[:8]}",
    )
    thread.start()
    return landed


def _stage_chat_uploads(files: Sequence[object]) -> list[object]:
    """Returns temp paths that _seed_turn_zero feeds to document.upload — which
    ingests by PATH, never bytes, so no file blob ever reaches the act-trail.
    """
    from services.filename_utils import safe_filename  # noqa: PLC0415
    from services.tmp_storage import new_tmp_path  # noqa: PLC0415

    paths: list[object] = []
    for f in files:
        if not f or not getattr(f, 'filename', None):
            continue
        name = safe_filename(getattr(f, 'filename')) or "attachment"
        tmp_path = new_tmp_path(f"{uuid.uuid4().hex[:8]}_{name}")
        getattr(f, 'save')(tmp_path)
        paths.append(tmp_path)
    return paths


# ── Action-button dispatch helpers ───────────────────────────────────────────


class _ActionButtonConfig(ProcessorConfig):
    """Minimal ProcessorConfig for action-button dispatches (no ACT loop runs).

    No ACT loop is executed; the three prompt builders are never invoked —
    they return "" to satisfy the abstract base.
    """

    def __init__(self) -> None:
        super().__init__(
            channel="action_button",
            role="action_button",
            policy_channel=ProcessorConfig.PolicyChannel.CHAT,
            always_available=[],
            skip_transcript=True,
            skip_input_row=True,
            suppress_history=True,
            broadcast_to=None,
            memory_seed=False,
        )

    def get_user_definition(self, mp: object) -> str:
        return ""

    def get_user_prompt(self, mp: object) -> str:
        return ""

    def get_system_prompt(self, mp: object) -> str:
        return ""


class _ActionCtx:
    """Minimal mp-like context for ToolDispatcher in action-button dispatches.

    ``broadcast_to=None`` on the config keeps dispatches silent. ``uid=None``
    signals a non-user channel dispatch. A fresh ``cancel_event`` per instance
    means each action-button dispatch can be independently signalled.
    """

    config = _ActionButtonConfig()
    uid = None

    def __init__(self) -> None:
        self.cancel_event = threading.Event()


# ── HTTP endpoints ────────────────────────────────────────────────────────────


@chat_ns.route("/action")
class ActionResource(Resource):
    @require_auth
    @chat_ns.doc(
        description=(
            "Dispatches a rich-card control click via ToolDispatcher and returns the "
            "result synchronously. A card action is a silent state mutation, not a "
            "conversation turn: it never persists to the transcript and crosses no WS "
            "frame, so the calling card resolves its optimistic update straight from "
            "this HTTP response."
        )
    )
    @chat_ns.expect(_M["ActionRequest"])
    @chat_ns.response(200, "Action result", model=_M["ActionResult"])
    @chat_ns.response(400, "Unknown skill", model=_M["Error"])
    @chat_ns.response(422, "Validation failed", model=_M["Error"])
    @chat_ns.response(500, "Action handler error", model=_M["Error"])
    @responds(ActionResult, code=200)
    @expects(ActionRequest)
    def post(self, dto: ActionRequest) -> ActionResult | ResponseReturnValue:
        """Run the skill and return its result inline — the WS bus stays signal-only."""
        action_start = time.time()
        try:
            from abilities._dispatcher import ToolDispatcher  # noqa: PLC0415

            result_text = ToolDispatcher(_ActionCtx()).dispatch(dto.skill, dto.params)
            if result_text.startswith("Unknown tool:"):
                return error(f"Unknown skill: {dto.skill}", 400)

            return ActionResult(
                content=sanitize(result_text or "Done."),
                duration_ms=int((time.time() - action_start) * 1000),
            )
        except Exception as exc:
            logger.exception("[Chat API] Action handler error: %s", exc)
            return error("Action handler error", 500)
