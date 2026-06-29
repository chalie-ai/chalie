"""
Chat API — HTTP turn-control and rich-card action endpoints.

Routes:
  POST /api/chat/interrupt     — cooperatively cancel the active UMP turn. The
                             cancelled turn deletes its own transcript and
                             tool_call rows. Returns 200 always with JSON body.
  GET  /api/chat/subagents/active — active async delegates for the Processes panel.
  POST /api/chat/subagent/<sub_id>/stop — cooperatively cancel a running async
                             delegate by its sub_id. Returns 200 always.
  POST /api/action             — receive a rich-card control click; dispatches via
                             ToolDispatcher.dispatch() and returns the result
                             synchronously in the response body (no WS frame).

Send path:
  User messages are sent via POST /api/thread (new thread) or
  POST /api/thread/<turn_id> (reply) — see api/threads.py. Both return 201 with
  an empty body; the turn surfaces via WS signals → REST pull.

Design:
  WS is receive-only push (server→client). All client→server requests use HTTP.
  Mid-ACT user messages are handled by the frontend: POST /chat/interrupt
  cancels the active turn (which self-cleans its DB rows), then the frontend
  starts a fresh turn with the combined original+new message text.

  User-channel messages flow through MessageProcessor.process() with the
  UserConfig ProcessorConfig subclass — no MessageProcessor subclass. Live
  output is signal-only, gated by the config's BROADCASTS_STATE: the surface
  holds no event memory and refetches turn blocks over REST. Every lifecycle
  signal (created/working/updated/done + the per-tool tool_called/tool_done
  timers) is emitted by MessageProcessor itself through its `broadcast`
  chokepoint. The only signal originating here is the fallback error + done a
  FAILED turn needs (it never reaches the processor's terminal).
"""

import logging
import threading
import time
import uuid
from typing import Sequence, cast

from flask.typing import ResponseReturnValue
from flask_restx import Namespace, Resource

from .auth import require_auth
from .dto import Error, expects, register_dto, responds
from .dto.boundary import error
from .dto.chat import ActionRequest, ActionResult
from .dto.subagent import ActiveSubagents, Interrupted, SubagentStopResult
from services.log_utils import safe
from services.markup import sanitize
from services.processor_config import ProcessorConfig
from services.websocket_broker import WebSocketBroker

logger = logging.getLogger(__name__)

chat_ns = Namespace("chat", description="Chat turn control and rich-card actions", path="/api")

register_dto(chat_ns, ActionRequest, ActionResult, Interrupted, SubagentStopResult, ActiveSubagents, Error)

_M = chat_ns.models


# ── Active UMP turn tracking ─────────────────────────────────────────────────

_active_ump: "_ActiveTurn | None" = None
_ump_lock = threading.Lock()


def _get_active_ump() -> "_ActiveTurn | None":
    with _ump_lock:
        return _active_ump


def _set_active_ump(proc: "_ActiveTurn") -> None:
    global _active_ump
    with _ump_lock:
        _active_ump = proc


def _clear_active_ump(turn: "_ActiveTurn") -> None:
    global _active_ump
    with _ump_lock:
        if _active_ump is turn:
            _active_ump = None


class _ActiveTurn:
    """Holds the cancel_event (so interrupt/stop endpoints can signal
    cancellation) and the original raw_input (so dispatch_message can combine
    mid-turn messages).
    """

    __slots__ = ("_cancel_event", "_raw_input")

    def __init__(self, cancel_event: threading.Event, raw_input: str) -> None:
        self._cancel_event = cancel_event
        self._raw_input = raw_input

    def cancel(self) -> None:
        self._cancel_event.set()


# ── Background helpers ────────────────────────────────────────────────────────


def _run_chat_background(
    turn: _ActiveTurn,
    cancel_event: threading.Event,
    raw_input: str,
    config: "ProcessorConfig",
    metadata: dict[str, object],
    request_id: str,
) -> None:
    """Runs the user turn to completion. The surface's lifecycle signals
    (created → working → updated → done, plus the per-tool timers) come from
    MessageProcessor itself; only a turn that FAILS before reaching its terminal
    needs the fallback error + done emitted here so the surface still drops its
    working indicator. The active-UMP reference is cleared in ``finally`` (the
    next statement after process() returns) — long before any surface, having
    just received ``done`` over the socket, could round-trip a fresh send."""
    from services.message_processor import MessageProcessor  # noqa: PLC0415

    try:
        MessageProcessor.process(raw_input, config, metadata, cancel_event=cancel_event)
    except Exception as exc:
        logger.exception("[Chat API] UMP error for %s: %s", request_id, exc)
        # A failed turn never reached _end_turn, so no ``done`` fired — emit the
        # error bubble and a terminal ``done`` (keyed by the newest user row, the
        # turn that just failed) so the surface releases its working state. A
        # cancelled turn is owned by its replacement and stays silent.
        if not cancel_event.is_set():
            from services.transcript_service import Transcript  # noqa: PLC0415
            rows = Transcript.get_recent("user", limit=1)
            failed_turn = cast("int | None", rows[-1].get("turn_id")) if rows else None
            broker = WebSocketBroker()
            broker.broadcast({"type": "error", "message": "Turn failed unexpectedly", "recoverable": False})
            broker.broadcast({"type": "done", "turn_id": failed_turn})
    finally:
        _clear_active_ump(turn)


def deliver_async_result(mp: object, result_text: str, cancel_event: threading.Event) -> None:
    """Does NOT register _active_ump and does NOT go through dispatch_message /
    _start_turn, so it never cancels or combines the user's in-flight foreground
    turn — it simply appends another assistant turn.

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
    thread_id: "int | None" = None,
) -> None:
    """If an ACT loop is already in-flight, cancels it, concatenates the original
    message with the new one (separated by two newlines), and starts a fresh turn
    with the combined text. The cancelled turn's DB rows are cleaned up by
    _cleanup_cancelled() in the processor.

    ``thread_id`` appends to an existing thread instead of opening a new one; the
    reply's input row carries the supplied thread_id so all its rows share it.
    A mid-turn interrupt with ``thread_id`` set cancels the active turn and starts
    a new reply to the SAME thread (the thread boundary is preserved).
    """
    attachments = attachments or []

    active = _get_active_ump()
    if active is not None and not hidden_input:
        original = getattr(active, "_raw_input", "") or ""
        active.cancel()
        text = original + "\n\n" + text
        logger.info("[Chat API] Mid-turn message — cancelled active UMP, combined text")

    _start_turn(text, source, attachments, hidden_input, thread_id=thread_id)


def _start_turn(
    text: str, source: str, attachments: list[object], hidden_input: bool = False,
    *, thread_id: "int | None" = None,
) -> str:
    from configs.channels import UserConfig  # noqa: PLC0415

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
        "channel": "user",
        "hidden_input": hidden_input,
    }
    if thread_id is not None:
        metadata["thread_id"] = thread_id
        # Genuine user reply INTO a thread → FORK view. This is
        # the ONLY producer of is_thread_reply; the async-delivery path (which
        # reuses thread_id for a delegate→main continuation) explicitly drops it.
        metadata["is_thread_reply"] = True

    config = UserConfig(metadata)
    cancel_event = threading.Event()
    turn = _ActiveTurn(cancel_event=cancel_event, raw_input=text)
    _set_active_ump(turn)

    thread = threading.Thread(
        target=_run_chat_background,
        args=(turn, cancel_event, text, config, metadata, request_id),
        daemon=True,
        name=f"chat-{request_id[:8]}",
    )
    thread.start()
    return request_id


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


def _interrupt_active_turn() -> Interrupted:
    """Cancel the active UMP turn if one is running; always a 200 ack."""
    proc = _get_active_ump()
    if proc is not None:
        proc.cancel()
        logger.info("[Chat API] Interrupt signal delivered to active UMP turn")
        return Interrupted(interrupted=True)
    return Interrupted(reason="no_active_turn")


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


@chat_ns.route("/chat/interrupt")
class ChatInterruptResource(Resource):
    @require_auth
    @chat_ns.response(200, "OK", model=_M["Interrupted"])
    @responds(Interrupted)
    def post(self) -> Interrupted:
        """The cancelled turn deletes its own transcript and tool_call rows — no data persists for an interrupted turn.

        Always returns HTTP 200.
        """
        return _interrupt_active_turn()


@chat_ns.route("/chat/subagents/active")
class ActiveSubagentsResource(Resource):
    @require_auth
    @chat_ns.response(200, "OK", model=_M["ActiveSubagents"])
    @responds(ActiveSubagents)
    def get(self) -> ActiveSubagents:
        """Hydrates the Processes panel on page load/reconnect, since WS push events
        are missed while the client is disconnected. Each row carries the tool name,
        the model's summary of what the delegate is doing, and when it started.
        """
        from services.async_delegate_runner import async_delegate_runner  # noqa: PLC0415

        return ActiveSubagents(subagents=async_delegate_runner.active())


@chat_ns.route("/chat/subagent/<sub_id>/stop")
class SubagentStopResource(Resource):
    @require_auth
    @chat_ns.param("sub_id", "Async delegate id")
    @chat_ns.response(200, "OK", model=_M["SubagentStopResult"])
    @responds(SubagentStopResult)
    def post(self, sub_id: str) -> SubagentStopResult:
        """Cooperatively cancel a running async delegate.

        Delegates to async_delegate_runner.cancel(). The delegate's cancel_event
        is set; the ACT loop exits at the next iteration boundary.

        Always returns HTTP 200.
        """
        from services.async_delegate_runner import async_delegate_runner  # noqa: PLC0415

        if async_delegate_runner.cancel(sub_id):
            logger.info("[Chat API] Stop signal delivered to delegate %s", safe(sub_id[:8]))
            return SubagentStopResult(cancelled=True)
        return SubagentStopResult(reason="not_found")


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
