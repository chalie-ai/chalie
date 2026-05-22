"""
Chat API — HTTP endpoints for client→server communication.

Routes:
  POST /chat               — receive a user message; start a new UMP turn or
                             inject a steer into the active ACT loop. Returns
                             202 immediately; response arrives via
                             WebSocketBroker.broadcast().
  POST /chat/stop          — cooperatively cancel the active UMP turn. Sets
                             the cancel_event on the active processor. Returns
                             200 always with JSON body.
  POST /action             — receive an action button click; dispatch via
                             ActDispatcherService. Returns 202 immediately;
                             response arrives via WebSocketBroker.broadcast().
  POST /chat/subagent/<sub_id>/stop — cooperatively cancel a running async
                             subagent by its sub_id. Returns 200 always.

Design:
  WS is receive-only push (server→client). All client→server requests use
  HTTP. The /chat endpoint owns routing: if a UMP turn is in-flight it
  dispatches a steer tool_call; otherwise it starts a new turn.
"""

import logging
import threading
import time
import uuid

from flask import Blueprint, jsonify, request

from .auth import require_auth
from services.markup import actions_to_xml, sanitize
from services.websocket_broker import WebSocketBroker
from services.segment_service import SegmentService

logger = logging.getLogger(__name__)

chat_bp = Blueprint("chat", __name__)

# ── Active UMP turn tracking ─────────────────────────────────────────────────

_active_ump = None
_ump_lock = threading.Lock()


def _get_active_ump():
    with _ump_lock:
        return _active_ump


def _set_active_ump(proc) -> None:
    global _active_ump
    with _ump_lock:
        _active_ump = proc


# ── Background helpers ────────────────────────────────────────────────────────


def _run_chat_background(
    proc,
    request_id: str,
    turn_start: float,
) -> None:
    """Background thread: process user message via UMP and broadcast response."""
    broker = WebSocketBroker()
    try:
        response = proc.send(request_id=request_id)

        metrics = proc._metrics.snapshot()
        transcript_ids = [proc._uid] if getattr(proc, "_uid", None) is not None else []

        content = sanitize(response or "")
        message_evt = {
            "type": "message",
            "content": content,
            "topic": "user",
            "mode": "UNIFIED",
            "confidence": 1.0,
            "exchange_id": request_id,
            "metrics": metrics,
        }
        message_evt["segments"] = SegmentService.build(content, transcript_ids)
        broker.broadcast(message_evt)

        elapsed_ms = int((time.time() - turn_start) * 1000)
        done_evt = {"type": "done", "duration_ms": elapsed_ms}
        if metrics:
            done_evt["metrics"] = metrics
        broker.broadcast(done_evt)

    except Exception as exc:
        logger.exception("[Chat API] UMP error for %s: %s", request_id, exc)
        partial_metrics = {}
        try:
            if getattr(proc, "_metrics", None) is not None:
                partial_metrics = proc._metrics.snapshot()
                partial_metrics["tokens_total_complete"] = False
        except Exception:
            pass
        broker.broadcast({
            "type": "error",
            "message": str(exc),
            "recoverable": False,
        })
        broker.broadcast({
            "type": "done",
            "duration_ms": int((time.time() - turn_start) * 1000),
            **({"metrics": partial_metrics} if partial_metrics else {}),
        })
    finally:
        _set_active_ump(None)


def dispatch_message(
    text: str,
    source: str = "text",
    attachments: list | None = None,
    hidden_input: bool = False,
    intercept: bool = False,
) -> None:
    """Single chokepoint for all message sources entering the user channel.

    Args:
        text: Message content.
        source: Origin identifier (``"text"``, ``"voice"``, ``"subagent"``,
                ``"scheduled"``, ``"external_agent"``).
        attachments: Optional file paths from POST /upload.
        hidden_input: When True the input row is NOT written to the transcript
                      (the synthesized assistant response still is).
        intercept: When True, steers into the active UMP if one exists.
                   When False, always starts a fresh UMP turn.
    """
    attachments = attachments or []

    if intercept:
        proc = _get_active_ump()
        if proc:
            proc.handle_tool({
                'name': 'steer',
                'input': {'text': text},
                'id': f'steer_{uuid.uuid4().hex[:12]}',
            })
            return

    _start_turn(text, source, attachments, hidden_input)


def _start_turn(text: str, source: str, attachments: list, hidden_input: bool = False) -> str:
    """Start a new UMP turn in a background thread. Returns the new request_id."""
    from services.user_message_processor import UserMessageProcessor

    request_id = str(uuid.uuid4())
    turn_start = time.time()

    try:
        from services.world_state import world_state, Signal
        world_state.absorb(Signal(source="http_chat", kind="user_message", payload={"text": text[:200]}))
    except Exception as exc:
        logger.debug("[Chat API] world_state.absorb failed: %s", exc)

    broker = WebSocketBroker()
    broker.broadcast({"type": "status", "stage": "processing"})

    def _on_narration(text_val, step=0):
        if text_val:
            broker.broadcast({
                "type": "act_narration",
                "text": sanitize(text_val),
                "step": step,
            })

    def _on_tool_event(event):
        if isinstance(event, dict) and event.get("type") in ("act_tool_start", "act_tool_end"):
            broker.broadcast(event)

    metadata = {
        "uuid": request_id,
        "exchange_id": request_id,
        "source": source,
        "attachments": attachments,
        "channel": "user",
        "hidden_input": hidden_input,
    }

    proc = UserMessageProcessor(
        raw_input=text,
        metadata=metadata,
        on_narration=_on_narration,
        on_tool_event=_on_tool_event,
    )
    proc.set_turn_start(turn_start)
    _set_active_ump(proc)

    thread = threading.Thread(
        target=_run_chat_background,
        args=(proc, request_id, turn_start),
        daemon=True,
        name=f"chat-{request_id[:8]}",
    )
    thread.start()
    return request_id


# ── HTTP endpoints ────────────────────────────────────────────────────────────


@chat_bp.route("/chat", methods=["POST"])
@require_auth
def post_chat():
    """Receive a user message and start or steer a UMP turn.

    Body (JSON):
        text (str): Message text.
        source (str, optional): ``"text"`` or ``"voice"``. Default ``"text"``.
        attachments (list[str], optional): Up to 10 tmp_path values from
            POST /upload.

    Response:
        202 Accepted — the message was received.
        Response arrives asynchronously via WebSocketBroker.broadcast().
    """
    body = request.get_json(silent=True) or {}
    text = (body.get("text") or "").strip()
    source = body.get("source") or "text"
    attachments = (body.get("attachments") or [])[:10]

    if not text and not attachments:
        return jsonify({"status": "ignored", "reason": "empty message"}), 202

    if not text and attachments:
        text = "[File attached]"

    dispatch_message(text, source=source, attachments=attachments, intercept=True)
    return jsonify({"status": "accepted"}), 202


@chat_bp.route("/chat/stop", methods=["POST"])
@require_auth
def post_chat_stop():
    """Cooperatively cancel the active UMP turn.

    Sets the cancel_event on the active processor so the ACT loop exits at
    the next iteration boundary. The turn still completes normally (store +
    post_turn) with final_text=''. The frontend receives the normal 'done'
    WS event when the loop actually exits.

    Always returns HTTP 200 — the caller should not treat 200 as confirmation
    that the turn has stopped, only that the stop signal was delivered.

    Response JSON:
        {ok: true, cancelled: true}   — stop signal delivered to active turn
        {ok: true, reason: "no_active_turn"} — nothing was in-flight
    """
    proc = _get_active_ump()
    if proc is not None:
        proc._cancel_event.set()
        logger.info("[Chat API] Stop signal delivered to active UMP turn")
        return jsonify({"ok": True, "cancelled": True}), 200
    return jsonify({"ok": True, "reason": "no_active_turn"}), 200


@chat_bp.route("/chat/subagent/<sub_id>/stop", methods=["POST"])
@require_auth
def post_subagent_stop(sub_id: str):
    """Cooperatively cancel a running async subagent.

    Delegates to cancel_subagent() from the subagent ability module.
    The subagent's cancel_event is set; the ACT loop exits at the next
    iteration boundary.

    Always returns HTTP 200.

    Response JSON:
        {ok: true, cancelled: true}         — stop signal delivered
        {ok: true, reason: "not_found"}     — sub_id not in active registry
    """
    from abilities.subagent import cancel_subagent

    if cancel_subagent(sub_id):
        logger.info("[Chat API] Stop signal delivered to subagent %s", sub_id[:8])
        return jsonify({"ok": True, "cancelled": True}), 200
    return jsonify({"ok": True, "reason": "not_found"}), 200


@chat_bp.route("/action", methods=["POST"])
@require_auth
def post_action():
    """Receive an action button click and dispatch via ActDispatcherService.

    Body (JSON):
        skill (str): The ability name to invoke.
        Any additional keys are passed as parameters to the ability.

    Response:
        202 Accepted — the action was received.
        Response arrives asynchronously via WebSocketBroker.broadcast().
    """
    body = request.get_json(silent=True) or {}
    skill = body.get("skill") or ""
    if not skill:
        return jsonify({"error": "Missing 'skill' in action payload"}), 400

    action_start = time.time()

    def _run_action():
        broker = WebSocketBroker()
        try:
            from services.act_dispatcher_service import ActDispatcherService

            action = {
                "type": skill,
                **{k: v for k, v in body.items() if k != "skill"},
            }

            broker.broadcast({"type": "status", "stage": "processing"})

            dispatcher = ActDispatcherService()
            dispatch_result = dispatcher.dispatch_action("action_button", action)

            if dispatch_result.get("status") == "error":
                result_text = dispatch_result.get("result", "")
                if "Unknown action type" in result_text:
                    broker.broadcast({
                        "type": "error",
                        "message": f"Unknown skill: {skill}",
                        "recoverable": True,
                    })
                    broker.broadcast({"type": "done", "duration_ms": 0})
                    return

            result = dispatch_result.get("result")
            reply_actions = dispatch_result.get("reply_actions")

            if isinstance(result, dict) and "text" in result:
                reply_actions = reply_actions or result.get("reply_actions")
                result = result["text"]

            elapsed_ms = int((time.time() - action_start) * 1000)
            content = sanitize(result or "Done.")
            if reply_actions:
                content += actions_to_xml(reply_actions)

            message_evt = {
                "type": "message",
                "content": content,
                "topic": "",
                "mode": "ACT",
                "confidence": 0.95,
                "exchange_id": "",
                "metrics": {
                    "tokens_total": 0,
                    "tokens_total_complete": False,
                    "tools": {},
                    "response_time_s": round(time.time() - action_start, 3),
                },
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
                "metrics": {
                    "tokens_total": 0,
                    "tokens_total_complete": False,
                    "tools": {},
                    "response_time_s": round(time.time() - action_start, 3),
                },
            })
            broker.broadcast({"type": "done", "duration_ms": 0})

    thread = threading.Thread(target=_run_action, daemon=True, name=f"action-{skill}")
    thread.start()

    return jsonify({"status": "accepted"}), 202
