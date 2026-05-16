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
from services.rich_media_parser import parse as _parse_rich_media

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
                    logger.warning("[WS-DEBUG] JSON parse failed, raw_len=%d, raw_start=%s", len(raw) if raw else 0, repr(raw[:200]) if raw else 'None')
                    continue

                msg_type = msg.get('type', '')
                if msg_type == 'chat':
                    logger.info("[WS-DEBUG] recv chat: raw_len=%d, msg_keys=%s, has_files=%s", len(raw), list(msg.keys()), 'files' in msg)

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
                elif msg_type == 'permission_response':
                    pass  # Handled via REST /api/policies/respond

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




def _fetch_tool_calls_for_transcript_ids(transcript_ids: list) -> list[dict]:
    """Fetch all tool_calls rows (including ephemeral=1) for a list of transcript IDs.

    Replaces the old recency-based lookup that:
      - filtered WHERE channel='user' AND role='user', silently missing subagent rows
      - raced DMN which can write user-channel rows between ACT completion and WS emit

    Args:
        transcript_ids: List of transcript.id values from the turn that produced
            the response. Typically a single-element list containing the UMP input
            row ID (``MessageProcessor._uid``).

    Returns:
        Tool_calls rows ordered by created_at, id. Empty list on any error.
    """
    if not transcript_ids:
        return []
    try:
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()
        placeholders = ','.join('?' * len(transcript_ids))
        with db.connection() as conn:
            tc_rows = conn.execute(
                f"SELECT tool_name, params, result, ephemeral, created_at FROM tool_calls "
                f"WHERE transcript_id IN ({placeholders}) "
                f"ORDER BY created_at, id",
                transcript_ids,
            ).fetchall()
        return [
            {
                "tool_name": r[0],
                "params": r[1],
                "result": r[2] or "",
                "ephemeral": r[3],
                "created_at": r[4],
            }
            for r in tc_rows
        ]
    except Exception as exc:
        logger.debug("[WS] _fetch_tool_calls_for_transcript_ids failed: %s", exc)
        return []


def _attach_segments(message_evt: dict, content: str, transcript_ids: list) -> None:
    """Attach the ``segments`` field to a message_evt dict in-place.

    Fetches tool_calls by exact transcript IDs (avoiding the stale recency
    heuristic) then calls RichMediaParser.parse(). Always produces at least
    one segment — for responses with no rich-media spans this is a single
    text segment; the frontend checks for ``segments`` presence, not length.

    When ``transcript_ids`` is empty or None (e.g. legacy callers), a warning
    is logged and a plain text segment is emitted — no silent degradation.

    Args:
        message_evt: The WS event dict being assembled (mutated in-place).
        content: Sanitised assistant text (``transcript.content``).
        transcript_ids: List of transcript row IDs for this turn. Populated
            from ``MessageProcessor._uid`` by the background chat handler.
    """
    if not transcript_ids:
        logger.warning("[WS] _attach_segments: no transcript_ids — emitting plain text segment")
        message_evt["segments"] = [{"type": "text", "content": content}]
        return
    tool_calls = _fetch_tool_calls_for_transcript_ids(transcript_ids)
    segments = _parse_rich_media(content, tool_calls)
    if not segments:
        segments = [{"type": "text", "content": content}]
    message_evt["segments"] = segments


def _handle_resume(ws, msg):
    """Replay missed events on reconnect."""
    last_seq = msg.get('last_seq', 0)
    events = _get_catchup_events(last_seq)
    for event in events:
        _send_json(ws, event)
    logger.debug(f"[WS] Resume: replayed {len(events)} events from seq {last_seq}")


def _handle_action(ws, msg):
    """Handle a deterministic action button click.

    All ability invocations flow through ``ActDispatcherService`` so there is a
    single tool-dispatch path.  The channel ``'action_button'`` is not
    ``'user'``, so the dispatcher never injects ``_rich_media_ordinal`` — the
    ordinal-None contract on ``Ability.execute()`` is automatically satisfied
    without any special-casing here.
    """
    payload = msg.get('payload', {})
    skill = payload.get('skill', '')
    if not skill:
        _send_json(ws, {"type": "error", "message": "Missing 'skill' in action payload"})
        return

    action_start = time.time()

    seq = _next_seq()
    _send_json(ws, {"type": "status", "stage": "processing", "seq": seq})

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
                seq = _next_seq()
                _send_json(ws, {"type": "error", "message": f"Unknown skill: {skill}", "recoverable": True, "seq": seq})
                seq = _next_seq()
                _send_json(ws, {"type": "done", "duration_ms": 0, "seq": seq})
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
        # Action-button responses come directly from abilities, not the ACT loop,
        # so there are no transcript IDs or tool_calls rows. Empty list causes
        # _attach_segments to emit a single plain text segment.
        _attach_segments(message_evt, content, [])
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
    files = (msg.get('files') or [])[:5]
    logger.info("[WS-DEBUG] _handle_chat: msg_keys=%s, files_count=%d, msg_len=%d", list(msg.keys()), len(files), len(str(msg)))

    if not text and not image_ids and not files:
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
                'files': files,
                'channel': 'user',
            }

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
                    # Run narration through the same nh3 chokepoint chat
                    # bubbles use — the frontend renders this via innerHTML
                    # and trusts the backend has already stripped anything
                    # outside the LLM tag allowlist.
                    store.set(f"output:{narration_id}", _json.dumps({
                        'type': 'act_narration',
                        'text': sanitize(text),
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
            response = proc.send(request_id=request_id)

            metrics = proc._metrics.snapshot()
            partial_metrics.update(metrics)

            # Thread the input transcript row ID so the WS handler can fetch
            # tool_calls by exact ID rather than by recency heuristic (Fix 2).
            # proc._uid is the user-input transcript row written at the top of
            # send(); all ACT tool_calls for this turn are stored against it.
            _transcript_ids = [proc._uid] if getattr(proc, '_uid', None) is not None else None

            output_svc = OutputService()
            output_svc.enqueue_text(
                topic='user',
                response=response,
                mode='UNIFIED',
                confidence=1.0,
                generation_time=0.0,
                original_metadata=metadata,
                metrics=metrics,
                transcript_ids=_transcript_ids,
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
                _msg_content = metadata.get("content", "")
                seq = _next_seq()
                message_evt = {
                    "type": "message",
                    "content": _msg_content,
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
                _attach_segments(
                    message_evt,
                    _msg_content,
                    metadata.get("transcript_ids") or [],
                )
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
                _fb_content = metadata.get("content", "")
                seq = _next_seq()
                message_evt = {
                    "type": "message",
                    "content": _fb_content,
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
                _attach_segments(
                    message_evt,
                    _fb_content,
                    metadata.get("transcript_ids") or [],
                )
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
