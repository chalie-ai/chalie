"""
WebSocket endpoint — single bidirectional channel replacing both SSE streams.

Protocol:
  → Client sends:  {"type": "chat", "text": "...", "source": "text|voice"}
  → Client sends:  {"type": "action", "payload": {"skill": "...", ...}}
  → Client sends:  {"type": "act_steer", "text": "..."}
  ← Server sends:  {"type": "status", "stage": "..."}
  ← Server sends:  {"type": "message", "content": "...", ...}
  ← Server sends:  {"type": "act_narration", "text": "...", "step": N}
  ← Server sends:  {"type": "done", "duration_ms": N}
  ← Server sends:  {"type": "drift|task|reminder|escalation|notification", ...}
  ← Server sends:  {"type": "ping"}
"""

import json
import time
import uuid
import logging
import threading

from utils.logger import set_correlation_id

from services.markup import actions_to_xml, sanitize
from services.websocket_broker import WebSocketBroker
from services.segment_service import SegmentService

logger = logging.getLogger(__name__)


def _drain_capability_alerts(store) -> None:
    """Replay persisted capability alert keys via broker and delete them after delivery."""
    try:
        broker = WebSocketBroker()
        alert_keys = store.keys('capability:alert:*')
        for key in alert_keys:
            raw = store.get(key)
            if raw:
                try:
                    data = json.loads(raw)
                    broker.broadcast(data)
                    store.delete(key)
                except Exception as exc:
                    logger.debug("[WS] Failed to send persisted capability alert: %s", exc)
    except Exception as exc:
        logger.debug("[WS] Failed to scan capability alerts: %s", exc)


def _dispatch_ws_message(ws, store, msg, active_request) -> None:
    """Route a parsed WebSocket message to the appropriate handler."""
    msg_type = msg.get('type', '')
    if msg_type == 'chat':
        _handle_chat(ws, store, msg, active_request)
    elif msg_type == 'action':
        _handle_action(msg)
    elif msg_type == 'act_steer':
        _handle_act_steer(store, msg, active_request)
    elif msg_type == 'resume':
        _handle_resume(msg)
    # 'pong' and 'permission_response' are intentionally no-ops here


def _ws_handler(ws) -> None:
    """Handle an individual WebSocket connection lifecycle.

    Authenticates the upgrade request via session cookie, registers the
    connection with WebSocketBroker, drains persisted capability alerts,
    then enters the main receive loop.
    """
    from flask import request as flask_request
    from services.auth_session_service import validate_session

    broker = WebSocketBroker()

    if not validate_session(flask_request):
        try:
            ws.send(json.dumps({"type": "error", "message": "Unauthorized"}))
        except Exception:
            pass
        try:
            ws.close()
        except Exception as exc:
            logger.debug("[WS] Close after auth failure failed: %s", exc)
        return

    request_id = str(uuid.uuid4())
    set_correlation_id(request_id)
    logger.debug("[WS] Connection established", extra={"connection_id": request_id})

    broker.connect(ws)

    from services.memory_client import MemoryClientService
    store = MemoryClientService.create_connection()

    _drain_capability_alerts(store)

    active_request = {'id': None}

    try:
        while True:
            raw = ws.receive(timeout=60)
            if raw is None:
                broker.broadcast({"type": "ping"})
                continue

            try:
                msg = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "[WS-DEBUG] JSON parse failed, raw_len=%d, raw_start=%s",
                    len(raw) if raw else 0,
                    repr(raw[:200]) if raw else 'None',
                )
                continue

            msg_type = msg.get('type', '')
            if msg_type == 'chat':
                logger.info(
                    "[WS-DEBUG] recv chat: raw_len=%d, msg_keys=%s, has_files=%s",
                    len(raw), list(msg.keys()), 'files' in msg,
                )

            _dispatch_ws_message(ws, store, msg, active_request)

    except Exception as exc:
        logger.debug("[WS] Connection closed: %s", exc)
    finally:
        broker.disconnect()


def register_websocket(sock):
    """Register the /ws endpoint on a flask-sock instance."""

    @sock.route('/ws')
    def ws_handler(ws):
        """Delegate to module-level _ws_handler for connection lifecycle."""
        _ws_handler(ws)


def _parse_meta(meta) -> dict:
    """Parse extracted_metadata which may be a JSON string or a dict."""
    if isinstance(meta, str):
        try:
            return json.loads(meta)
        except Exception:
            return {}
    return meta or {}


def _handle_resume(msg):
    """No-op resume handler — WS drop means page reload, no catch-up buffer."""
    logger.debug("[WS] Resume received (no-op — frontend should reload on drop)")


def _handle_action(msg):
    """Handle a deterministic action button click.

    All ability invocations flow through ``ActDispatcherService`` so there is a
    single tool-dispatch path.  The channel ``'action_button'`` is not
    ``'user'``, so the dispatcher never injects ``_rich_media_ordinal`` — the
    ordinal-None contract on ``Ability.execute()`` is automatically satisfied
    without any special-casing here.
    """
    broker = WebSocketBroker()
    payload = msg.get('payload', {})
    skill = payload.get('skill', '')
    if not skill:
        broker.broadcast({"type": "error", "message": "Missing 'skill' in action payload"})
        return

    action_start = time.time()

    broker.broadcast({"type": "status", "stage": "processing"})

    try:
        from services.act_dispatcher_service import ActDispatcherService

        # Build an action dict for the dispatcher: type = skill name, remaining
        # keys are the ability's input parameters.  'skill' itself is a
        # framework key (like 'type') so it is excluded from params.
        action = {
            'type': skill,
            **{k: v for k, v in payload.items() if k != 'skill'},
        }

        start = time.time()
        dispatcher = ActDispatcherService()
        dispatch_result = dispatcher.dispatch_action('action_button', action)

        if dispatch_result.get('status') == 'error':
            result_text = dispatch_result.get('result', '')
            if 'Unknown action type' in result_text:
                broker.broadcast({
                    "type": "error",
                    "message": f"Unknown skill: {skill}",
                    "recoverable": True,
                })
                broker.broadcast({"type": "done", "duration_ms": 0})
                return

        # Dispatcher already unwraps {'text': ..., 'reply_actions': ...} in
        # _build_success_result — result is plain text or dict by this point.
        result = dispatch_result.get('result')
        reply_actions = dispatch_result.get('reply_actions')

        # If the ability returned a dict with a 'text' key that the dispatcher
        # did not unwrap (non-standard ability shape), normalise it here.
        if isinstance(result, dict) and 'text' in result:
            reply_actions = reply_actions or result.get('reply_actions')
            result = result['text']

        elapsed_ms = int((time.time() - start) * 1000)

        # LLM / skill result → HTML content string. Sanitize() is the single
        # chokepoint: it strips every tag/attribute outside our allowlist
        # before the FE ever sees the content. The LLM is told to never emit
        # ``<a>``; the FE auto-linkifies any plain-text URLs it finds.
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
        # Action-button responses come directly from abilities, not the ACT loop,
        # so there are no transcript IDs or tool_calls rows.  SegmentService
        # emits a single plain text segment on empty transcript_ids.
        message_evt["segments"] = SegmentService.build(content, [])
        broker.broadcast(message_evt)
        broker.broadcast({"type": "done", "duration_ms": elapsed_ms})

    except Exception as exc:
        logger.exception("[WS] Action handler error: %s", exc)
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


def _handle_act_steer(store, msg, active_request):
    """Inject user steering text into the active ACT loop via MemoryStore."""
    steer_text = (msg.get('text') or '').strip()
    request_id = active_request.get('id')
    if steer_text and request_id:
        steer_key = f"steer:{request_id}"
        store.rpush(steer_key, steer_text)
        store.expire(steer_key, 120)
        logger.debug("[WS] Steer injected for %s: %s", request_id, steer_text[:60])


def _run_chat_background(
    text: str,
    source: str,
    attachments: list,
    request_id: str,
    turn_start: float,
    bg_error: dict,
    bg_done: threading.Event,
    partial_metrics: dict,
) -> None:
    """Background thread: process user message via UserMessageProcessor and broadcast response."""
    broker = WebSocketBroker()
    proc = None
    try:
        from services.user_message_processor import UserMessageProcessor

        metadata = {
            'uuid': request_id, 'exchange_id': request_id,
            'source': source, 'attachments': attachments, 'channel': 'user',
        }

        def _on_narration(text_val, step=0):
            if text_val:
                broker.broadcast({
                    'type': 'act_narration',
                    'text': sanitize(text_val),
                    'step': step,
                })

        def _on_tool_event(event):
            if isinstance(event, dict) and event.get('type') in ('act_tool_start', 'act_tool_end'):
                broker.broadcast(event)

        proc = UserMessageProcessor(
            raw_input=text,
            metadata=metadata,
            on_narration=_on_narration,
            on_tool_event=_on_tool_event,
        )
        proc.set_turn_start(turn_start)
        response = proc.send(request_id=request_id)

        metrics = proc._metrics.snapshot()
        partial_metrics.update(metrics)

        transcript_ids = [proc._uid] if getattr(proc, '_uid', None) is not None else []

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

    except Exception as exc:
        logger.exception("[WS] UserMessageProcessor error for %s: %s", request_id, exc)
        bg_error['message'] = str(exc)
        try:
            if getattr(proc, '_metrics', None) is not None:
                partial_snap = proc._metrics.snapshot()
                partial_snap['tokens_total_complete'] = False
                partial_metrics.update(partial_snap)
        except Exception as m_err:
            logger.debug("[WS] Partial metrics snapshot failed: %s", m_err)
        broker.broadcast({
            "type": "error",
            "message": str(exc),
            "recoverable": False,
        })
    finally:
        bg_done.set()


def _handle_chat(ws, store, msg, active_request=None):
    """Process a chat message — replaces the POST /chat SSE endpoint."""
    broker = WebSocketBroker()
    text = (msg.get('text') or '').strip()
    attachments = (msg.get('attachments') or [])[:10]
    logger.info(
        "[WS-DEBUG] _handle_chat: msg_keys=%s, attachments_count=%d, msg_len=%d",
        list(msg.keys()), len(attachments), len(str(msg)),
    )

    if not text and not attachments:
        return  # Nothing to process — silently drop

    if not text and attachments:
        text = '[File attached]'

    # Absorb typed signal so WorldState snapshot stays current.
    try:
        from services.world_state import world_state, Signal
        world_state.absorb(Signal(source='ws', kind='user_message', payload={'text': text[:200]}))
    except Exception as ws_err:
        logger.debug("[WS] world_state.absorb failed: %s", ws_err)

    source = msg.get('source', 'text')
    request_id = str(uuid.uuid4())

    if active_request is not None:
        active_request['id'] = request_id

    broker.broadcast({"type": "status", "stage": "processing"})

    turn_start = time.time()
    bg_error: dict = {}
    bg_done = threading.Event()
    partial_metrics: dict = {}

    thread = threading.Thread(
        target=_run_chat_background,
        args=(text, source, attachments, request_id, turn_start,
              bg_error, bg_done, partial_metrics),
        daemon=True,
    )
    thread.start()

    broker.broadcast({"type": "status", "stage": "thinking"})

    # Wait for the background thread to finish, then send done.
    # The background thread broadcasts its own message/error events directly.
    bg_done.wait()

    if active_request is not None:
        active_request['id'] = None

    elapsed_ms = int((time.time() - turn_start) * 1000)
    done_evt = {"type": "done", "duration_ms": elapsed_ms}
    if partial_metrics:
        done_evt["metrics"] = partial_metrics
    broker.broadcast(done_evt)
