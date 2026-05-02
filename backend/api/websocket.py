"""
WebSocket endpoint — single bidirectional channel replacing both SSE streams.

Protocol:
  → Client sends:  {"type": "chat", "text": "...", "source": "text|voice"}
  → Client sends:  {"type": "action", "payload": {"skill": "...", ...}}
  → Client sends:  {"type": "act_steer", "text": "..."}
  → Client sends:  {"type": "resume", "last_seq": N}
  ← Server sends:  {"type": "status", "stage": "...", "seq": N}
  ← Server sends:  {"type": "message", "content": "...", ..., "seq": N}
  ← Server sends:  {"type": "act_narration", "text": "...", "step": N, "seq": N}
  ← Server sends:  {"type": "done", "duration_ms": N, "seq": N}
  ← Server sends:  {"type": "drift|task|reminder|escalation|notification", ..., "seq": N}
  ← Server sends:  {"type": "ping"}
"""

import json
import time
import uuid
import logging
import threading
from collections import deque

from utils.logger import set_correlation_id

from services.markup import actions_to_xml, sanitize

logger = logging.getLogger(__name__)

# Monotonically increasing sequence counter (shared across all connections)
_seq_counter = 0
_seq_lock = threading.Lock()

# Catch-up buffer: last N events for reconnect replay
_CATCHUP_SIZE = 200
_catchup_buffer = deque(maxlen=_CATCHUP_SIZE)
_catchup_lock = threading.Lock()


def _next_seq():
    global _seq_counter
    with _seq_lock:
        _seq_counter += 1
        return _seq_counter


def _buffer_event(event: dict):
    """Store event in catch-up buffer for reconnect replay."""
    with _catchup_lock:
        _catchup_buffer.append(event)


def _get_catchup_events(last_seq: int) -> list:
    """Return all buffered events with seq > last_seq."""
    with _catchup_lock:
        return [e for e in _catchup_buffer if e.get('seq', 0) > last_seq]


def _send_json(ws, data: dict):
    """Send a JSON message, swallowing errors on closed connections."""
    try:
        ws.send(json.dumps(data))
    except Exception as e:
        logger.debug(f"[WS] Send failed (connection likely closed): {e}")


def register_websocket(sock):
    """Register the /ws endpoint on a flask-sock instance."""

    @sock.route('/ws')
    def ws_handler(ws):
        """
        Handle an individual WebSocket connection lifecycle.

        Authenticates the upgrade request via session cookie, then:
        - Subscribes to the ``output:events`` pub/sub channel for
          drift/card/task push events.
        - Drains any buffered notifications queued in ``notifications:recent``.
        - Triggers the first-contact welcome flow when applicable.
        - Spawns a daemon thread (``_drift_sender``) that forwards pub/sub
          events to the client with monotonic sequence numbers.
        - Enters the main receive loop, dispatching incoming messages to
          :func:`_handle_chat`, :func:`_handle_action`, or
          :func:`_handle_resume` based on the ``type`` field.

        Args:
            ws: The flask-sock WebSocket connection object for this connection.
        """
        from flask import request as flask_request
        from services.auth_session_service import validate_session

        # Auth: validate session cookie from the upgrade request
        if not validate_session(flask_request):
            _send_json(ws, {"type": "error", "message": "Unauthorized"})
            # Explicitly close the WebSocket before returning. Without this, flask-sock's
            # Werkzeug integration writes an HTTP 200 response into the already-upgraded TCP
            # connection, causing the browser to see "Invalid frame header".
            try:
                ws.close()
            except Exception as e:
                logger.debug(f"[WS] Close after auth failure failed: {e}")
            return

        # Bind a connection-scoped correlation ID so all log lines emitted
        # during this WebSocket session carry the same traceable identifier.
        request_id = str(uuid.uuid4())
        set_correlation_id(request_id)
        logger.debug("[WS] Connection established", extra={"connection_id": request_id})

        # Subscribe to output:events for drift/card/task push
        from services.memory_client import MemoryClientService
        store = MemoryClientService.create_connection()
        pubsub = store.pubsub()
        pubsub.subscribe('output:events')

        # Drain buffered notifications on connect
        while True:
            item = store.lpop('notifications:recent')
            if not item:
                break
            try:
                data = json.loads(item)
                seq = _next_seq()
                data['seq'] = seq
                _buffer_event(data)
                _send_json(ws, data)
            except Exception as e:
                logger.debug(f"[WS] Failed to drain buffered notification: {e}")

        # Replay any persisted capability alerts so the banner shows after a page refresh.
        # Keys are deleted after delivery — if the capability is still down, the next
        # monitor cycle will recreate the key.
        try:
            alert_keys = store.keys('capability:alert:*')
            for key in alert_keys:
                raw = store.get(key)
                if raw:
                    try:
                        data = json.loads(raw)
                        seq = _next_seq()
                        data['seq'] = seq
                        _send_json(ws, data)
                        store.delete(key)
                    except Exception as e:
                        logger.debug(f"[WS] Failed to send persisted capability alert: {e}")
        except Exception as e:
            logger.debug(f"[WS] Failed to scan capability alerts: {e}")

        # Background thread: push drift/output events to the WebSocket
        ws_open = threading.Event()
        ws_open.set()

        def _drift_sender():
            """Listen to output:events pub/sub and forward to WebSocket."""
            while ws_open.is_set():
                try:
                    msg = pubsub.get_message(timeout=15)
                    if msg and msg['type'] == 'message':
                        try:
                            data = json.loads(msg['data'])
                            seq = _next_seq()
                            data['seq'] = seq
                            _buffer_event(data)
                            _send_json(ws, data)
                        except (json.JSONDecodeError, TypeError):
                            pass
                    else:
                        # Keepalive ping
                        _send_json(ws, {"type": "ping"})
                except Exception as e:
                    logger.debug(f"[WS] Drift sender error: {e}")
                    if not ws_open.is_set():
                        break
                    time.sleep(1)

            try:
                pubsub.unsubscribe('output:events')
                pubsub.close()
            except Exception as e:
                logger.debug(f"[WS] Drift sender pubsub cleanup failed: {e}")

        drift_thread = threading.Thread(
            target=_drift_sender, daemon=True, name="ws-drift"
        )
        drift_thread.start()

        # Track active request for user steering (set by _handle_chat)
        active_request = {'id': None}

        # Main loop: receive client messages
        try:
            while True:
                raw = ws.receive(timeout=60)
                if raw is None:
                    # Client sent close or timeout — send a ping to probe
                    _send_json(ws, {"type": "ping"})
                    continue

                try:
                    msg = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue

                msg_type = msg.get('type', '')

                if msg_type == 'chat':
                    _handle_chat(ws, store, msg, active_request)
                elif msg_type == 'action':
                    _handle_action(ws, msg)
                elif msg_type == 'act_steer':
                    _handle_act_steer(store, msg, active_request)
                elif msg_type == 'resume':
                    _handle_resume(ws, msg)
                elif msg_type == 'pong':
                    pass  # Client keepalive response — no action needed

        except Exception as e:
            logger.debug(f"[WS] Connection closed: {e}")
        finally:
            ws_open.clear()


def _parse_meta(meta) -> dict:
    """Parse extracted_metadata which may be a JSON string or a dict."""
    if isinstance(meta, str):
        try:
            return json.loads(meta)
        except Exception:
            return {}
    return meta or {}


def _poll_until_terminal(svc, doc_id: str, deadline: float) -> str:
    """Poll a document until status is 'ready' / 'failed' or the deadline expires."""
    import time as _time
    doc = svc.get_document(doc_id)
    status = doc.get('status', '') if doc else ''
    while status not in ('ready', 'failed') and _time.monotonic() < deadline:
        _time.sleep(0.2)
        doc = svc.get_document(doc_id)
        if doc:
            status = doc.get('status', '')
    return status


def _image_ocr_tag(doc: dict, image_id: str) -> str:
    """Format a ready-image tag with OCR text (or <none> sentinel)."""
    ocr = (_parse_meta(doc.get('extracted_metadata')).get('ocr_text') or '').strip()
    return f"[image id={image_id} ocr={ocr[:500]}]" if ocr else f"[image id={image_id} ocr=<none>]"


def _resolve_image_tag(svc, image_id: str, deadline: float, request_id: str) -> str:
    """Resolve one image_id to a structured tag. Never silently drops context."""
    doc = svc.get_document(image_id)
    if not doc:
        logger.warning(f"[WS] file_tags image not found image_id={image_id} request_id={request_id}")
        return f"[image id={image_id} status=not_found]"

    status = doc.get('status', '')
    if status not in ('ready', 'failed'):
        status = _poll_until_terminal(svc, image_id, deadline)

    if status == 'ready':
        return _image_ocr_tag(svc.get_document(image_id) or doc, image_id)
    if status == 'failed':
        logger.warning(f"[WS] file_tags image analysis failed image_id={image_id} request_id={request_id}")
        return f"[image id={image_id} status=failed]"
    logger.warning(f"[WS] file_tags image analysis timed out image_id={image_id} status={status} request_id={request_id}")
    return f"[image id={image_id} status=timeout]"


def _format_ready_upload_tag(svc, doc_id: str, original_name: str, fallback_text: str, source_type: str) -> str:
    """Format the tag for a ready recent-upload. Re-reads for the final committed row."""
    final = svc.get_document(doc_id) or {}
    if source_type == 'chat_image':
        return _image_ocr_tag(final, doc_id)
    final_text = (final.get('clean_text') or fallback_text or '').strip()
    if final_text:
        return f"[document id={doc_id} name={original_name} content={final_text[:2000]}]"
    return f"[document id={doc_id} name={original_name} content=<empty>]"


def _fetch_recent_upload_row(db):
    """SELECT the most recent upload/chat_image within the last 120 seconds."""
    with db.connection() as conn:
        return conn.execute(
            """
            SELECT id, original_name, status, clean_text, source_type
            FROM documents
            WHERE source_type IN ('upload', 'chat_image')
              AND deleted_at IS NULL
              AND created_at >= datetime('now', '-120 seconds')
            ORDER BY created_at DESC
            LIMIT 1
            """,
        ).fetchone()


def _resolve_recent_upload(db, svc, request_id: str):
    """Return (tag, doc_id_if_injected) for the recent-upload fallback, or (None, None).

    Only fires when the most recent upload is already 'ready' at lookup time.
    The fallback exists to cover paste/drop where the chat turn races the
    upload XHR — that race is sub-second, so anything still in-flight after
    the row appears here is far more likely to be a stale upload from earlier
    in the conversation than the user's intent for *this* turn. Injecting a
    `status=timeout` or `status=failed` tag in that case derails the model
    into apologising about a missing attachment the user never made.
    """
    try:
        row = _fetch_recent_upload_row(db)
    except Exception as e:
        logger.warning(f"[WS] file_tags recent-upload heuristic error request_id={request_id}: {e}")
        return None, None
    if not row:
        return None, None

    doc_id, original_name, doc_status, clean_text, source_type = row
    if doc_status != 'ready':
        return None, None
    return _format_ready_upload_tag(svc, doc_id, original_name, clean_text, source_type), doc_id


def _resolve_file_tags(image_ids: list, request_id: str) -> list:
    """
    Build file_tags for a chat turn. Returns structured strings to be appended
    to metadata['file_tags'] — consumed by UserMessageProcessor.getUserPrompt.

    Images share a single 10 s deadline so N slow images cannot block a turn
    for N × 10 s. When image_ids is empty, the most recent upload/chat_image
    (within 120 s) falls back in — covers paste/drop where the chat turn is
    sent before the upload XHR completes. Every failure path emits a tag.
    """
    import time as _time
    from services.document_service import DocumentService
    from services.database_service import get_shared_db_service

    db = get_shared_db_service()
    svc = DocumentService(db)

    tags = []
    doc_ids_injected = []

    images_deadline = _time.monotonic() + 10.0
    for image_id in image_ids:
        tags.append(_resolve_image_tag(svc, image_id, images_deadline, request_id))

    if not image_ids:
        tag, injected_id = _resolve_recent_upload(db, svc, request_id)
        if tag:
            tags.append(tag)
            if injected_id:
                doc_ids_injected.append(injected_id)

    logger.info(
        f"[WS] file_tags injected request_id={request_id} "
        f"tags={len(tags)} image_ids={image_ids} doc_ids={doc_ids_injected}"
    )
    return tags


def _handle_resume(ws, msg):
    """Replay missed events on reconnect."""
    last_seq = msg.get('last_seq', 0)
    events = _get_catchup_events(last_seq)
    for event in events:
        _send_json(ws, event)
    logger.debug(f"[WS] Resume: replayed {len(events)} events from seq {last_seq}")


def _handle_action(ws, msg):
    """Handle a deterministic action button click."""
    payload = msg.get('payload', {})
    skill = payload.get('skill', '')
    if not skill:
        _send_json(ws, {"type": "error", "message": "Missing 'skill' in action payload"})
        return

    action_start = time.time()

    seq = _next_seq()
    _send_json(ws, {"type": "status", "stage": "processing", "seq": seq})

    try:
        from abilities._registry import AbilityRegistry
        try:
            ability = AbilityRegistry.get(skill)
        except KeyError:
            seq = _next_seq()
            _send_json(ws, {"type": "error", "message": f"Unknown skill: {skill}", "recoverable": True, "seq": seq})
            seq = _next_seq()
            _send_json(ws, {"type": "done", "duration_ms": 0, "seq": seq})
            return

        start = time.time()
        result = ability.execute('action_button', payload, None)

        # Handle structured results (text + reply_actions)
        reply_actions = None
        if isinstance(result, dict) and 'text' in result:
            reply_actions = result.get('reply_actions')
            result = result['text']

        elapsed_ms = int((time.time() - start) * 1000)

        # LLM / skill result → HTML content string. Sanitize() is the single
        # chokepoint: it strips every tag/attribute outside our allowlist
        # before the FE ever sees the content. The LLM is told to never emit
        # ``<a>``; the FE auto-linkifies any plain-text URLs it finds.
        content = sanitize(result or "Done.")
        if reply_actions:
            content += actions_to_xml(reply_actions)

        seq = _next_seq()
        message_evt = {
            "type": "message",
            "content": content,
            "topic": "",
            "mode": "ACT",
            "confidence": 0.95,
            "exchange_id": "",
            "seq": seq,
            "metrics": {
                "tokens_total": 0,
                "tokens_total_complete": False,
                "tools": {},
                "response_time_s": round(time.time() - action_start, 3),
            },
        }
        _buffer_event(message_evt)
        _send_json(ws, message_evt)

        seq = _next_seq()
        done_evt = {"type": "done", "duration_ms": elapsed_ms, "seq": seq}
        _buffer_event(done_evt)
        _send_json(ws, done_evt)

    except Exception as e:
        logger.error(f"[WS] Action handler error: {e}", exc_info=True)
        seq = _next_seq()
        _send_json(ws, {
            "type": "error",
            "message": str(e),
            "recoverable": True,
            "seq": seq,
            "metrics": {
                "tokens_total": 0,
                "tokens_total_complete": False,
                "tools": {},
                "response_time_s": round(time.time() - action_start, 3),
            },
        })
        seq = _next_seq()
        _send_json(ws, {"type": "done", "duration_ms": 0, "seq": seq})


def _handle_act_steer(store, msg, active_request):
    """Inject user steering text into the active ACT loop via MemoryStore."""
    steer_text = (msg.get('text') or '').strip()
    request_id = active_request.get('id')
    if steer_text and request_id:
        steer_key = f"steer:{request_id}"
        store.rpush(steer_key, steer_text)
        store.expire(steer_key, 120)
        logger.debug(f"[WS] Steer injected for {request_id}: {steer_text[:60]}")


def _handle_chat(ws, store, msg, active_request=None):
    """Process a chat message — replaces the POST /chat SSE endpoint."""
    text = (msg.get('text') or '').strip()
    image_ids = (msg.get('image_ids') or [])[:3]  # max 3 images

    if not text and not image_ids:
        return  # Nothing to process — silently drop

    # If user sent only images with no text, provide a sensible fallback.
    # Image resolution (polling MemoryStore for analysis results) is handled
    # in UserMessageProcessor. The WS handler passes image_ids through in metadata.
    if not text and image_ids:
        text = '[Image attached]'

    # Absorb typed signal so WorldState snapshot stays current.
    try:
        from services.world_state import world_state, Signal
        world_state.absorb(Signal(source='ws', kind='user_message', payload={'text': text[:200]}))
    except Exception as _ws_err:
        logger.debug("[WS] world_state.absorb failed: %s", _ws_err)

    source = msg.get('source', 'text')
    request_id = str(uuid.uuid4())

    # Track active request for user steering
    if active_request is not None:
        active_request['id'] = request_id

    # Subscribe to per-request SSE channel (OutputService publishes here)
    pubsub = store.pubsub()
    sse_channel = f"sse:{request_id}"
    pubsub.subscribe(sse_channel)

    # Send initial status
    seq = _next_seq()
    _send_json(ws, {"type": "status", "stage": "processing", "seq": seq})

    # Measure wall-clock from before thread creation so thread startup overhead
    # doesn't inflate response_time_s beyond what the user experiences.
    turn_start = time.time()

    # Track background thread completion
    bg_error = {}
    bg_done = threading.Event()
    # Shared dict for partial metrics — background thread writes, error path reads.
    partial_metrics: dict = {}

    def _handle_chat_background():
        """Background thread: process user message via UserMessageProcessor and publish response."""
        try:
            from services.user_message_processor import UserMessageProcessor
            from services.output_service import OutputService

            metadata = {
                'uuid': request_id,
                'exchange_id': request_id,
                'source': source,
                'image_ids': image_ids,
                'channel': 'user',
            }

            # Resolve file context tags — waits up to 10 s for image analysis
            # and injects a recent-upload document heuristic when applicable.
            # MUST run before UserMessageProcessor is constructed so that
            # getUserPrompt() picks up file_tags on its first call.
            _file_tags_t0 = time.time()
            metadata['file_tags'] = _resolve_file_tags(image_ids, request_id)
            _file_tags_wait_ms = int((time.time() - _file_tags_t0) * 1000)

            def _on_narration(text, step=0):
                """Publish per-iteration synthesis text to the per-request SSE channel."""
                if not request_id or not text:
                    return
                try:
                    from uuid import uuid4
                    import json as _json
                    narration_id = f"narr_{uuid4().hex[:12]}"
                    # Reuse the outer `store` from ws_chat (line 113) — no need
                    # to create a second MemoryClient connection per narration
                    # (Commit 8 critic P2-1).
                    store.set(f"output:{narration_id}", _json.dumps({
                        'type': 'act_narration',
                        'text': text,
                        'step': step,
                    }), ex=300)
                    store.publish(f"sse:{request_id}", narration_id)
                except Exception as e:
                    logger.debug(f"[WS] Narration publish failed: {e}")

            def _on_tool_event(event):
                """Publish per-tool start/end events to the per-request SSE channel.

                event = {type: 'act_tool_start'|'act_tool_end', call_id, name?, iter?, ms?, ok?}
                Mirrors _on_narration: stores blob to output:{evt_id}, publishes evt_id
                on sse:{request_id}.
                """
                if not request_id or not isinstance(event, dict):
                    return
                evt_type = event.get('type')
                if evt_type not in ('act_tool_start', 'act_tool_end'):
                    return
                try:
                    from uuid import uuid4
                    import json as _json
                    evt_id = f"tool_{uuid4().hex[:12]}"
                    store.set(f"output:{evt_id}", _json.dumps(event), ex=300)
                    store.publish(f"sse:{request_id}", evt_id)
                except Exception as e:
                    logger.debug(f"[WS] Tool event publish failed: {e}")

            proc = UserMessageProcessor(
                raw_input=text,
                metadata=metadata,
                on_narration=_on_narration,
                on_tool_event=_on_tool_event,
            )
            proc.set_turn_start(turn_start)
            try:
                proc._metrics.add_stage_ms('file_tags_wait', _file_tags_wait_ms)
            except Exception:
                pass
            response = proc.send(request_id=request_id)

            metrics = proc._metrics.snapshot()
            partial_metrics.update(metrics)

            output_svc = OutputService()
            output_svc.enqueue_text(
                topic='user',
                response=response,
                mode='UNIFIED',
                confidence=1.0,
                generation_time=0.0,
                original_metadata=metadata,
                metrics=metrics,
            )

            # Store result at output:{request_id} so the fallback path can
            # find it if the pub/sub message was missed.
            try:
                _fb_content = sanitize(response or "")
                fallback_output = {
                    "type": "TEXT",
                    "topic": 'user',
                    "metadata": {
                        "content": _fb_content,
                        "mode": "UNIFIED",
                        "confidence": 1.0,
                        "metadata": metadata,
                        "metrics": metrics,
                    },
                }
                store.setex(f"output:{request_id}", 300, json.dumps(fallback_output))
            except Exception as fb_err:
                logger.debug(f"[WS] Fallback store failed: {fb_err}")
        except Exception as e:
            logger.error(f"[WS] UserMessageProcessor error for {request_id}: {e}", exc_info=True)
            bg_error['message'] = str(e)
            # Capture any metrics accumulated before the failure so the
            # error frame can surface them (spec contract #3).
            try:
                if 'proc' in locals() and getattr(proc, '_metrics', None) is not None:
                    partial_snap = proc._metrics.snapshot()
                    # Mark as incomplete — we failed mid-turn.
                    partial_snap['tokens_total_complete'] = False
                    partial_metrics.update(partial_snap)
            except Exception as m_err:
                logger.debug(f"[WS] Partial metrics snapshot failed: {m_err}")
            try:
                store.publish(sse_channel, json.dumps({"error": str(e)}))
            except Exception as e2:
                logger.debug(f"[WS] Failed to publish error to SSE channel: {e2}")
        finally:
            bg_done.set()

    thread = threading.Thread(target=_handle_chat_background, daemon=True)
    thread.start()

    seq = _next_seq()
    _send_json(ws, {"type": "status", "stage": "thinking", "seq": seq})

    # Listen for pub/sub events until done, disconnected, or cancelled.
    # The loop is unbounded — no wall-clock timeout on the chat request.
    # Exit conditions: (a) 'done' event arrives, (b) background thread sets
    # bg_done and no pub/sub message received, (c) WS disconnects.
    start_time = time.time()
    message_received = False

    while True:
        ps_msg = pubsub.get_message(timeout=1.0)

        if ps_msg and ps_msg['type'] == 'message':
            payload = ps_msg['data']
            if isinstance(payload, bytes):
                payload = payload.decode()

            # Check for error or close signal
            try:
                parsed = json.loads(payload)
                if 'error' in parsed:
                    seq = _next_seq()
                    evt = {"type": "error", "message": parsed['error'], "recoverable": True, "seq": seq}
                    if partial_metrics:
                        evt["metrics"] = partial_metrics
                    _buffer_event(evt)
                    _send_json(ws, evt)
                    seq = _next_seq()
                    done_evt = {"type": "done", "duration_ms": int((time.time() - start_time) * 1000), "seq": seq}
                    _buffer_event(done_evt)
                    _send_json(ws, done_evt)
                    break
                if parsed.get('type') == 'close':
                    seq = _next_seq()
                    done_evt = {"type": "done", "duration_ms": int((time.time() - start_time) * 1000), "seq": seq}
                    _buffer_event(done_evt)
                    _send_json(ws, done_evt)
                    break
            except (json.JSONDecodeError, TypeError):
                pass

            # It's an output_id — fetch the full output
            output_id = payload.strip('"')
            output_data = store.get(f"output:{output_id}")

            if output_data:
                output = json.loads(output_data)

                # Act narration: forward to client as a progress update (not a final message)
                if output.get('type') == 'act_narration':
                    seq = _next_seq()
                    narr_evt = {
                        "type": "act_narration",
                        "text": output.get("text", ""),
                        "step": output.get("step", 0),
                        "seq": seq,
                    }
                    _buffer_event(narr_evt)
                    _send_json(ws, narr_evt)
                    continue  # Keep listening — this isn't the final response

                # Act tool events: forward pill start/end events to client
                if output.get('type') in ('act_tool_start', 'act_tool_end'):
                    seq = _next_seq()
                    pill_evt = {**output, 'seq': seq}
                    _buffer_event(pill_evt)
                    _send_json(ws, pill_evt)
                    continue

                metadata = output.get("metadata", {})
                original_meta = metadata.get("metadata", {})
                seq = _next_seq()
                message_evt = {
                    "type": "message",
                    "content": metadata.get("content", ""),
                    "topic": output.get("topic", ""),
                    "mode": metadata.get("mode", ""),
                    "confidence": metadata.get("confidence", 0),
                    "exchange_id": original_meta.get("exchange_id", ""),
                    "seq": seq,
                }
                _msg_metrics = metadata.get("metrics")
                if _msg_metrics is not None:
                    # Stamp response_time_s at the actual dispatch moment so
                    # the metric reflects user-perceived latency (spec contract).
                    try:
                        _msg_metrics = dict(_msg_metrics)
                        _msg_metrics['response_time_s'] = round(time.time() - turn_start, 3)
                    except Exception:
                        pass
                    message_evt["metrics"] = _msg_metrics
                _buffer_event(message_evt)
                _send_json(ws, message_evt)
                message_received = True

                # Clear active request when response is delivered
                if active_request is not None:
                    active_request['id'] = None

                seq = _next_seq()
                done_evt = {"type": "done", "duration_ms": int((time.time() - start_time) * 1000), "seq": seq}
                _buffer_event(done_evt)
                _send_json(ws, done_evt)
                break

        # Fallback: background thread done but no pub/sub arrived
        if bg_done.is_set() and not message_received:
            time.sleep(0.5)  # Brief grace period
            output_key = f"output:{request_id}"
            fallback_data = store.get(output_key)
            if fallback_data:
                output = json.loads(fallback_data)
                metadata = output.get("metadata", {})
                original_meta = metadata.get("metadata", {})
                seq = _next_seq()
                message_evt = {
                    "type": "message",
                    "content": metadata.get("content", ""),
                    "topic": output.get("topic", ""),
                    "mode": metadata.get("mode", ""),
                    "confidence": metadata.get("confidence", 0),
                    "exchange_id": original_meta.get("exchange_id", ""),
                    "seq": seq,
                }
                _fallback_metrics = metadata.get("metrics")
                if _fallback_metrics is not None:
                    try:
                        _fallback_metrics = dict(_fallback_metrics)
                        _fallback_metrics['response_time_s'] = round(time.time() - turn_start, 3)
                    except Exception:
                        pass
                    message_evt["metrics"] = _fallback_metrics
                _buffer_event(message_evt)
                _send_json(ws, message_evt)
            elif bg_error:
                seq = _next_seq()
                err = {"type": "error", "message": bg_error.get('message', 'Processing failed'), "recoverable": False, "seq": seq}
                if partial_metrics:
                    err["metrics"] = partial_metrics
                _buffer_event(err)
                _send_json(ws, err)
            else:
                seq = _next_seq()
                err = {"type": "error", "message": "No response received", "recoverable": True, "seq": seq}
                if partial_metrics:
                    err["metrics"] = partial_metrics
                _buffer_event(err)
                _send_json(ws, err)

            seq = _next_seq()
            done_evt = {"type": "done", "duration_ms": int((time.time() - start_time) * 1000), "seq": seq}
            _buffer_event(done_evt)
            _send_json(ws, done_evt)
            break
    pubsub.unsubscribe(sse_channel)
    pubsub.close()
