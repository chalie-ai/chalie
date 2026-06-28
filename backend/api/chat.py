"""
Chat API — HTTP endpoints for client→server communication.

Routes:
  POST /chat               — receive a user message; always starts a new UMP
                             turn. Returns 202 immediately; response arrives
                             via WebSocketBroker.broadcast().
  POST /chat/interrupt     — cooperatively cancel the active UMP turn. The
                             cancelled turn deletes its own transcript and
                             tool_call rows. Returns 200 always with JSON body.
  POST /action             — receive an action button click; dispatches via
                             ToolDispatcher.dispatch(). Returns 202 immediately;
                             response arrives via WebSocketBroker.broadcast().
  POST /chat/subagent/<sub_id>/stop — cooperatively cancel a running async
                             delegate by its sub_id. Returns 200 always.

Design:
  WS is receive-only push (server→client). All client→server requests use
  HTTP. The /chat endpoint always starts a new UMP turn. Mid-ACT user
  messages are handled by the frontend: POST /chat/interrupt cancels the
  active turn (which self-cleans its DB rows), then the frontend starts a
  fresh turn with the combined original+new message text.

  User-channel messages flow through MessageProcessor.process() with the
  UserConfig ProcessorConfig subclass — no MessageProcessor subclass.
  Live output is gated by broadcast_to='user' on the config: each chain step
  broadcasts its interim assistant text via _broadcast_interim() and
  ToolDispatcher.dispatch() emits live tool events, both fired only when
  broadcast_to is set; the turn's end message is broadcast by
  _broadcast_turn_result() once the chain returns.
"""

import logging
import threading
import time
import uuid
from typing import TYPE_CHECKING, Sequence, cast

from flask import request
from flask.typing import ResponseReturnValue
from flask_restx import Namespace, Resource

from .auth import require_auth
from services.locale_service import CHAT_TIMESTAMP_FMT, format_date
from services.log_utils import safe
from services.markup import sanitize
from services.time_utils import utc_now
from services.websocket_broker import WebSocketBroker
from services.segment_service import SegmentService

if TYPE_CHECKING:
    from services.processor_config import ProcessorConfig

logger = logging.getLogger(__name__)

chat_ns = Namespace("chat", description="Chat operations", path="/")

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
    turn_start: float,
) -> None:
    """Clears the active UMP reference BEFORE broadcasting done so the frontend
    can immediately POST /chat without racing a still-set active_ump."""
    from services.message_processor import MessageProcessor  # noqa: PLC0415

    broker = WebSocketBroker()
    try:
        response = MessageProcessor.process(
            raw_input, config, metadata, cancel_event=cancel_event
        )

        # Cancelled turn — skip broadcast entirely. The replacement turn
        # (started by dispatch_message) owns the WS event stream now.
        if cancel_event.is_set():
            _clear_active_ump(turn)
            return

        # Clear the active UMP BEFORE broadcasting done so the frontend can
        # immediately POST /chat without racing a still-set active_ump.
        _clear_active_ump(turn)
        _broadcast_turn_result(response, request_id, turn_start)

    except Exception as exc:
        logger.exception("[Chat API] UMP error for %s: %s", request_id, exc)
        _clear_active_ump(turn)
        if not cancel_event.is_set():
            broker.broadcast({
                "type": "error",
                "message": str(exc),
                "recoverable": False,
            })
            broker.broadcast({
                "type": "done",
                "duration_ms": int((time.time() - turn_start) * 1000),
            })
    finally:
        _clear_active_ump(turn)


def _broadcast_interim(metadata: dict[str, object], content: str) -> None:
    """Broadcast a chain step's interim assistant text on the user channel.

    Fired from MessageProcessor._emit_interim the moment the model emits a
    tool-bearing step, so the surface shows assistant prose and tool batches
    interleaved live within the turn. The text is already markdown→HTML
    normalised; we sanitise and emit a ``message`` event identical in shape to
    the final turn result but flagged ``interim`` and WITHOUT a trailing
    ``done`` — the chain is still running. The step's tools have not been
    dispatched yet, so the segment is plain text (no rich-media pairing here;
    that lands on the final row, resolved turn-wide).
    """
    broker = WebSocketBroker()
    safe_content = sanitize(content or "")
    exchange_id = (metadata or {}).get("exchange_id") or (metadata or {}).get("uuid") or ""
    broker.broadcast({
        "type": "message",
        "content": safe_content,
        "topic": "user",
        "mode": "UNIFIED",
        "confidence": 1.0,
        "exchange_id": exchange_id,
        "metrics": {},
        "interim": True,
        "segments": [{"type": "text", "content": safe_content}],
        "timestamp": format_date(utc_now(), CHAT_TIMESTAMP_FMT, for_ui=True) or "",
    })


def _broadcast_provider_retry(attempt: int, max_attempts: int) -> None:
    """Notify the user surface that a provider call failed and is being resent.

    Fired from MessageProcessor._send_with_retry before each resend on the user
    channel. The frontend renders this as a transient toast — the turn is still
    in flight, so no error bubble and no ``done``.
    """
    WebSocketBroker().broadcast({
        "type": "provider_retry",
        "message": "The AI provider had a problem — retrying…",
        "attempt": attempt,
        "max_attempts": max_attempts,
    })


def _broadcast_turn_result(response: str, request_id: str, turn_start: float) -> None:
    """Shared by the foreground user turn and the background async-result synthesis
    so both surface a turn through the exact same WS event shape."""
    broker = WebSocketBroker()

    # Pair the final row's rich-media spans with the tools that produced them.
    # By broadcast time the assistant reply is persisted, so the newest channel
    # row is the turn's final assistant row. Resolve TURN-WIDE from it — the span
    # sits on the final row but the tool ran on a step row — via the same shared
    # function the /conversation/recent refresh path uses, so both paths pair
    # span tags with tool_calls identically.
    transcript_ids: list[int] = []
    try:
        from services.transcript_service import Transcript  # noqa: PLC0415
        rows = Transcript.get_recent("user", limit=1)
        if rows:
            from services.database_service import get_shared_db_service  # noqa: PLC0415
            from services.rich_media_parser import resolve_tool_call_transcript_ids  # noqa: PLC0415
            with get_shared_db_service().connection() as conn:
                transcript_ids = resolve_tool_call_transcript_ids(cast(int, rows[-1]["id"]), conn)
    except Exception as exc:
        logger.debug("[Chat API] transcript_id lookup failed: %s", exc)

    content = sanitize(response or "")
    message_evt: dict[str, object] = {
        "type": "message",
        "content": content,
        "topic": "user",
        "mode": "UNIFIED",
        "confidence": 1.0,
        "exchange_id": request_id,
        "metrics": {},
        "timestamp": format_date(utc_now(), CHAT_TIMESTAMP_FMT, for_ui=True) or "",
    }
    message_evt["segments"] = SegmentService.build(content, transcript_ids)

    elapsed_ms = int((time.time() - turn_start) * 1000)
    broker.broadcast(message_evt)
    broker.broadcast({"type": "done", "duration_ms": elapsed_ms})


def deliver_async_result(mp: object, result_text: str, cancel_event: threading.Event) -> None:
    """Does NOT register _active_ump and does NOT go through dispatch_message /
    _start_turn, so it never cancels or combines the user's in-flight foreground
    turn — it simply appends another assistant turn.

    The delegate's ``cancel_event`` is threaded into the synthesis turn so the
    Processes-panel stop control aborts a spiralling delegate at the next chain
    boundary (the processor returns "" and self-cleans when it is set).
    """
    from services.message_processor import MessageProcessor  # noqa: PLC0415

    config = getattr(mp, "config", None)
    if config is None:
        logger.warning("[Chat API] async delivery skipped: captured mp has no config")
        return

    synth_config = config.with_hidden_input()
    # Clone the originating metadata but suppress the input row and drop
    # attachments — they were already ingested on the originating turn and must
    # not re-upload on the synthesis turn.
    metadata = dict(getattr(mp, "_metadata", None) or {})
    metadata["hidden_input"] = True
    metadata["attachments"] = []
    request_id = str(uuid.uuid4())
    turn_start = time.time()

    # The backgrounded result lands as a NEW turn on the channel: the
    # MessageProcessor advances the turn cursor itself. With skip_input_row set
    # (hidden_input), _setup writes no input row and the synthesised reply is the
    # turn's end message — write_assistant_row allocates its own turn_id at write
    # time. Broadcast through the same pipeline as a foreground reply.
    response = MessageProcessor.process(result_text, synth_config, metadata, cancel_event=cancel_event)
    if cancel_event.is_set():
        return
    _broadcast_turn_result(response, request_id, turn_start)


def dispatch_message(
    text: str,
    source: str = "text",
    attachments: list[object] | None = None,
    hidden_input: bool = False,
) -> None:
    """If an ACT loop is already in-flight, cancels it, concatenates the original
    message with the new one (separated by two newlines), and starts a fresh turn
    with the combined text. The cancelled turn's DB rows are cleaned up by
    _cleanup_cancelled() in the processor.
    """
    attachments = attachments or []

    active = _get_active_ump()
    if active is not None and not hidden_input:
        original = getattr(active, "_raw_input", "") or ""
        active.cancel()
        text = original + "\n\n" + text
        logger.info("[Chat API] Mid-turn message — cancelled active UMP, combined text")

    _start_turn(text, source, attachments, hidden_input)


def _start_turn(text: str, source: str, attachments: list[object], hidden_input: bool = False) -> str:
    from configs.channels import UserConfig  # noqa: PLC0415

    request_id = str(uuid.uuid4())
    turn_start = time.time()

    try:
        from services.world_state import world_state, Signal  # noqa: PLC0415
        world_state.absorb(Signal(source="http_chat", kind="user_message", payload={"text": text[:200]}))
    except Exception as exc:
        logger.debug("[Chat API] world_state.absorb failed: %s", exc)

    broker = WebSocketBroker()
    broker.broadcast({"type": "status", "stage": "processing"})

    metadata: dict[str, object] = {
        "uuid": request_id,
        "exchange_id": request_id,
        "source": source,
        "attachments": attachments,
        "channel": "user",
        "hidden_input": hidden_input,
    }

    config = UserConfig(metadata)
    cancel_event = threading.Event()
    turn = _ActiveTurn(cancel_event=cancel_event, raw_input=text)
    _set_active_ump(turn)

    thread = threading.Thread(
        target=_run_chat_background,
        args=(turn, cancel_event, text, config, metadata, request_id, turn_start),
        daemon=True,
        name=f"chat-{request_id[:8]}",
    )
    thread.start()
    return request_id


# ── HTTP endpoints ────────────────────────────────────────────────────────────


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


def _broadcast_user_echo(text: str, echo_id: str) -> None:
    """Echo a just-received user message to every open surface.

    Single user, many surfaces: the user may have Chalie open on several
    devices/tabs at once, all of which must show the same conversation. The
    surface that sent the message has already rendered it optimistically and
    recognises its own ``echo_id`` to ignore this frame; every OTHER open
    surface has no such bubble yet and renders one from this broadcast — so all
    surfaces stay in sync.

    The text is sent verbatim (the client renders a user bubble as escaped
    plain text), so the echoed bubble matches exactly what the sender typed.
    Only user-typed messages enter through ``post_chat`` and reach this echo —
    scheduled / external / async-synthesis turns do not, so no synthetic user
    bubble is ever broadcast.
    """
    WebSocketBroker().broadcast({
        "type": "user_message",
        "content": text or "",
        "echo_id": echo_id,
        "timestamp": format_date(utc_now(), CHAT_TIMESTAMP_FMT, for_ui=True) or "",
    })


def _interrupt_active_turn() -> tuple[dict[str, object], int]:
    """Shared logic for POST /chat/interrupt and POST /chat/stop."""
    proc = _get_active_ump()
    if proc is not None:
        proc.cancel()
        logger.info("[Chat API] Interrupt signal delivered to active UMP turn")
        return {"ok": True, "interrupted": True}, 200
    return {"ok": True, "reason": "no_active_turn"}, 200


@chat_ns.route("/chat")
class ChatResource(Resource):
    @require_auth
    @chat_ns.response(202, "Accepted")
    def post(self) -> ResponseReturnValue:
        """Files are staged to temp paths and ingested via document.upload (by PATH, never bytes) at turn 0.

        Response arrives asynchronously via WebSocketBroker.broadcast().
        """
        text = (request.form.get("text") or "").strip()
        source = request.form.get("source") or "text"
        echo_id = request.form.get("echo_id") or ""
        attachments = _stage_chat_uploads(cast(Sequence[object], request.files.getlist("files")[:10]))

        if not text and not attachments:
            return {"status": "error", "reason": "message required"}, 400

        if not text and attachments:
            text = "[File attached]"

        # Echo the user message to every open surface so they all show it (the
        # sender ignores its own echo via echo_id; peers render the bubble).
        _broadcast_user_echo(text, echo_id)

        dispatch_message(text, source=source, attachments=attachments)
        return {"status": "accepted"}, 202


@chat_ns.route("/chat/interrupt")
class ChatInterruptResource(Resource):
    @require_auth
    @chat_ns.response(200, "OK")
    def post(self) -> ResponseReturnValue:
        """The cancelled turn deletes its own transcript and tool_call rows — no data persists for an interrupted turn.

        Always returns HTTP 200.
        """
        return _interrupt_active_turn()


@chat_ns.route("/chat/stop")
class ChatStopResource(Resource):
    @require_auth
    @chat_ns.response(200, "OK")
    def post(self) -> ResponseReturnValue:
        """Deprecated alias for POST /chat/interrupt. New callers should use POST /chat/interrupt instead."""
        return _interrupt_active_turn()


@chat_ns.route("/chat/subagents/active")
class ActiveSubagentsResource(Resource):
    @require_auth
    @chat_ns.response(200, "OK")
    def get(self) -> ResponseReturnValue:
        """Hydrates the Processes panel on page load/reconnect, since WS push events
        are missed while the client is disconnected. Each row carries the tool name,
        the model's summary of what the delegate is doing, and when it started.
        """
        from services.async_delegate_runner import async_delegate_runner

        return {"subagents": async_delegate_runner.active()}, 200


@chat_ns.route("/chat/subagent/<sub_id>/stop")
class SubagentStopResource(Resource):
    @require_auth
    @chat_ns.response(200, "OK")
    def post(self, sub_id: str) -> ResponseReturnValue:
        """Cooperatively cancel a running async delegate.

        Delegates to async_delegate_runner.cancel(). The delegate's cancel_event
        is set; the ACT loop exits at the next iteration boundary.

        Always returns HTTP 200.

        Response JSON:
            {ok: true, cancelled: true}         — stop signal delivered
            {ok: true, reason: "not_found"}     — sub_id not in active registry
        """
        from services.async_delegate_runner import async_delegate_runner

        if async_delegate_runner.cancel(sub_id):
            logger.info("[Chat API] Stop signal delivered to delegate %s", safe(sub_id[:8]))
            return {"ok": True, "cancelled": True}, 200
        return {"ok": True, "reason": "not_found"}, 200


@chat_ns.route("/action")
class ActionResource(Resource):
    @require_auth
    @chat_ns.response(202, "Accepted")
    def post(self) -> ResponseReturnValue:
        """Response arrives asynchronously via WebSocketBroker.broadcast()."""
        body = request.get_json(silent=True) or {}
        skill = body.get("skill") or ""
        if not skill:
            return {"error": "Missing 'skill' in action payload"}, 400

        action_start = time.time()

        def _run_action() -> None:
            broker = WebSocketBroker()
            try:
                from abilities._dispatcher import ToolDispatcher  # noqa: PLC0415
                from services.processor_config import ProcessorConfig  # noqa: PLC0415

                params = {k: v for k, v in body.items() if k != "skill"}

                broker.broadcast({"type": "status", "stage": "processing"})

                # Build a minimal flat-path context for action-button dispatches.
                # ToolDispatcher requires an mp-like object with config, uid,
                # cancel_event.  broadcast_to=None keeps these
                # dispatches silent (no live WS events for action buttons).
                # ProcessorConfig is abstract, so action-button dispatch needs a
                # concrete subclass; no ACT loop runs here, so the three prompt
                # builders are never invoked — they return "" to satisfy the base.
                class _ActionButtonConfig(ProcessorConfig):
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

                _action_config = _ActionButtonConfig()

                class _ActionCtx:
                    config = _action_config
                    uid = None
                    cancel_event = threading.Event()

                ctx = _ActionCtx()
                result_text = ToolDispatcher(ctx).dispatch(skill, params)

                if result_text.startswith("Unknown tool:"):
                    broker.broadcast({
                        "type": "error",
                        "message": f"Unknown skill: {skill}",
                        "recoverable": True,
                    })
                    broker.broadcast({"type": "done", "duration_ms": 0})
                    return

                elapsed_ms = int((time.time() - action_start) * 1000)
                content = sanitize(result_text or "Done.")

                message_evt: dict[str, object] = {
                    "type": "message",
                    "content": content,
                    "topic": "",
                    "mode": "ACT",
                    "confidence": 0.95,
                    "exchange_id": "",
                    "metrics": {},
                    "timestamp": format_date(utc_now(), CHAT_TIMESTAMP_FMT, for_ui=True) or "",
                }
                message_evt["segments"] = SegmentService.build(content, [])
                broker.broadcast(message_evt)
                broker.broadcast({"type": "done", "duration_ms": elapsed_ms})

            except Exception as exc:
                logger.exception("[Chat API] Action handler error: %s", exc)
                broker.broadcast({
                    "type": "error",
                    "message": str(exc),
                    "recoverable": True,
                })
                broker.broadcast({"type": "done", "duration_ms": 0})

        thread = threading.Thread(target=_run_action, daemon=True, name=f"action-{skill}")
        thread.start()

        return {"status": "accepted"}, 202