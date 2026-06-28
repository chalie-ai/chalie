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
  Live output is signal-only, gated by broadcast_to='user': the surface holds
  no event memory and refetches turn blocks over REST. _broadcast_working()
  binds the live turn, _broadcast_turn_updated() fires at each assistant-row
  write so the surface refetches, tool_invoked carries the transient 'using X'
  line, and _broadcast_done() releases the working state once the chain returns.
"""

import logging
import threading
import time
import uuid
from typing import TYPE_CHECKING, Sequence, cast

from flask import Blueprint, jsonify, request
from flask.typing import ResponseReturnValue

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

chat_bp = Blueprint("chat", __name__)

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
    """Clears the active UMP reference BEFORE broadcasting done so the frontend
    can immediately POST /chat without racing a still-set active_ump."""
    from services.message_processor import MessageProcessor  # noqa: PLC0415

    broker = WebSocketBroker()
    try:
        MessageProcessor.process(raw_input, config, metadata, cancel_event=cancel_event)

        # Cancelled turn — skip broadcast entirely. The replacement turn
        # (started by dispatch_message) owns the WS event stream now.
        if cancel_event.is_set():
            _clear_active_ump(turn)
            return

        # Clear the active UMP BEFORE broadcasting done so the frontend can
        # immediately POST /chat without racing a still-set active_ump.
        _clear_active_ump(turn)
        _broadcast_done()

    except Exception as exc:
        logger.exception("[Chat API] UMP error for %s: %s", request_id, exc)
        _clear_active_ump(turn)
        if not cancel_event.is_set():
            broker.broadcast({
                "type": "error",
                "message": str(exc),
                "recoverable": False,
            })
            _broadcast_done()
    finally:
        _clear_active_ump(turn)


def _broadcast_created(turn_id: int) -> None:
    """Announce a freshly-allocated turn_id for a NEW thread (the non-reply send,
    ``not _forked``). Fired the instant the input row's turn_id is allocated — the
    ONLY way the just-allocated id reaches the surface, since the send response
    carries no id (spec §6.1/§6.2). The surface binds its optimistic main-spine
    placeholder to this id and pulls content on ``turn_updated``."""
    WebSocketBroker().broadcast({"type": "created", "turn_id": turn_id})


def _broadcast_working(turn_id: int) -> None:
    """Signal that processing (re)starts on an EXISTING turn — a reply send
    (``_forked``) or a reactivation. The surface already knows the turn_id (the
    reply path's URL), so this just flips that turn's pill + panel to the working
    state. Signal-only — content arrives on ``turn_updated`` (spec §6.1)."""
    WebSocketBroker().broadcast({"type": "working", "turn_id": turn_id})


def _broadcast_turn_updated(turn_id: int) -> None:
    """Signal that a turn's persisted rows changed — fired at each assistant-row
    write boundary (every chain step and the final synthesis). Signal-only: the
    surface refetches the turn block (coalescing concurrent fires by turn_id) and
    atomically replaces it. The transient 'using X' line rides ``tool_invoked``."""
    WebSocketBroker().broadcast({"type": "turn_updated", "turn_id": turn_id})


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


def _broadcast_done() -> None:
    """Release the turn's working state — the terminal signal, shared by the
    foreground turn and the background async-result synthesis. The content is
    already on the surface via the final ``turn_updated`` → refetch; this only
    flips working off, keyed by the just-finished turn's id (the newest user row)."""
    turn_id: "int | None" = None
    try:
        from services.transcript_service import Transcript  # noqa: PLC0415
        rows = Transcript.get_recent("user", limit=1)
        if rows:
            turn_id = cast("int | None", rows[-1].get("turn_id"))
    except Exception as exc:
        logger.debug("[Chat API] done turn_id lookup failed: %s", exc)
    WebSocketBroker().broadcast({"type": "done", "turn_id": turn_id})


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
    MessageProcessor.process(result_text, synth_config, metadata, cancel_event=cancel_event)
    if cancel_event.is_set():
        return
    _broadcast_done()


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


@chat_bp.route("/chat", methods=["POST"])
@require_auth
def post_chat() -> ResponseReturnValue:
    """Files are staged to temp paths and ingested via document.upload (by PATH, never bytes) at turn 0.

    Response arrives asynchronously via WebSocketBroker.broadcast().
    """
    text = (request.form.get("text") or "").strip()
    source = request.form.get("source") or "text"
    echo_id = request.form.get("echo_id") or ""
    attachments = _stage_chat_uploads(cast(Sequence[object], request.files.getlist("files")[:10]))

    raw_thread_id = request.form.get("thread_id") or ""
    thread_id: "int | None" = None
    if raw_thread_id:
        try:
            thread_id = int(raw_thread_id)
        except (ValueError, TypeError):
            thread_id = None

    if not text and not attachments:
        return jsonify({"status": "error", "reason": "message required"}), 400

    if not text and attachments:
        text = "[File attached]"

    # Echo the user message to every open surface so they all show it (the
    # sender ignores its own echo via echo_id; peers render the bubble).
    _broadcast_user_echo(text, echo_id)

    dispatch_message(text, source=source, attachments=attachments, thread_id=thread_id)
    return jsonify({"status": "accepted"}), 202


@chat_bp.route("/chat/interrupt", methods=["POST"])
@require_auth
def post_chat_interrupt() -> ResponseReturnValue:
    """The cancelled turn deletes its own transcript and tool_call rows — no data persists for an interrupted turn.

    Always returns HTTP 200.
    """
    proc = _get_active_ump()
    if proc is not None:
        proc.cancel()
        logger.info("[Chat API] Interrupt signal delivered to active UMP turn")
        return jsonify({"ok": True, "interrupted": True}), 200
    return jsonify({"ok": True, "reason": "no_active_turn"}), 200


@chat_bp.route("/chat/stop", methods=["POST"])
@require_auth
def post_chat_stop() -> ResponseReturnValue:
    """Deprecated alias for POST /chat/interrupt. New callers should use POST /chat/interrupt instead."""
    return post_chat_interrupt()


@chat_bp.route("/chat/subagents/active", methods=["GET"])
@require_auth
def get_active_subagents() -> ResponseReturnValue:
    """Hydrates the Processes panel on page load/reconnect, since WS push events
    are missed while the client is disconnected. Each row carries the tool name,
    the model's summary of what the delegate is doing, and when it started.
    """
    from services.async_delegate_runner import async_delegate_runner

    return jsonify({"subagents": async_delegate_runner.active()}), 200


@chat_bp.route("/chat/subagent/<sub_id>/stop", methods=["POST"])
@require_auth
def post_subagent_stop(sub_id: str) -> ResponseReturnValue:
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
        return jsonify({"ok": True, "cancelled": True}), 200
    return jsonify({"ok": True, "reason": "not_found"}), 200


@chat_bp.route("/action", methods=["POST"])
@require_auth
def post_action() -> ResponseReturnValue:
    """Response arrives asynchronously via WebSocketBroker.broadcast()."""
    body = request.get_json(silent=True) or {}
    skill = body.get("skill") or ""
    if not skill:
        return jsonify({"error": "Missing 'skill' in action payload"}), 400

    action_start = time.time()

    def _run_action() -> None:
        broker = WebSocketBroker()
        try:
            from abilities._dispatcher import ToolDispatcher  # noqa: PLC0415
            from services.processor_config import ProcessorConfig  # noqa: PLC0415

            params = {k: v for k, v in body.items() if k != "skill"}

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

    return jsonify({"status": "accepted"}), 202
