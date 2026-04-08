"""
Agent-to-agent communication endpoint.

Accepts messages from external agents (e.g. Claude Code) and feeds them
through the full cognitive pipeline (digest_worker → unified_generate).
Chalie processes these exactly like human messages — memory pipeline,
trait extraction, episodic consolidation, the works.

Auth: Bearer token via wrapper_tokens.
"""

import json
import time
import uuid
import logging
import threading

from flask import Blueprint, request, jsonify

from .auth import require_auth

logger = logging.getLogger(__name__)

agent_bp = Blueprint('agent', __name__, url_prefix='/api/agent')


@agent_bp.route('/message', methods=['POST'])
@require_auth
def agent_message():
    """Accept a message from an external agent and run it through the full pipeline.

    Request body:
        {"text": "...", "source": "claude-code"}

    Returns the full response blocks + metadata once processing completes.
    """
    data = request.get_json(silent=True) or {}
    text = (data.get('text') or '').strip()
    source = data.get('source', 'agent')

    if not text:
        return jsonify({"error": "Missing 'text' field"}), 400

    request_id = str(uuid.uuid4())

    # Subscribe to per-request output channel (same as WebSocket flow)
    from services.memory_client import MemoryClientService
    store = MemoryClientService.create_connection()
    pubsub = store.pubsub()
    sse_channel = f"sse:{request_id}"
    pubsub.subscribe(sse_channel)

    # Spawn digest_worker in background thread (same as _handle_chat)
    bg_error = {}
    bg_done = threading.Event()

    def run_digest():
        try:
            from workers.digest_worker import digest_worker
            digest_worker(text, metadata={
                'uuid': request_id,
                'source': source,
            })
        except Exception as e:
            logger.error(f"[Agent] digest_worker error for {request_id}: {e}", exc_info=True)
            bg_error['message'] = str(e)
            try:
                store.publish(sse_channel, json.dumps({"error": str(e)}))
            except Exception:
                pass
        finally:
            bg_done.set()

    thread = threading.Thread(target=run_digest, daemon=True)
    thread.start()

    # Collect response by listening on pub/sub (same pattern as WebSocket)
    start_time = time.time()
    timeout_seconds = 360
    response_blocks = []
    response_meta = {}

    while time.time() - start_time < timeout_seconds:
        ps_msg = pubsub.get_message(timeout=1.0)

        if ps_msg and ps_msg['type'] == 'message':
            payload = ps_msg['data']
            if isinstance(payload, bytes):
                payload = payload.decode()

            # Check for error or close signal
            try:
                parsed = json.loads(payload)
                if 'error' in parsed:
                    pubsub.unsubscribe(sse_channel)
                    pubsub.close()
                    return jsonify({"error": parsed['error']}), 500
                if parsed.get('type') == 'close':
                    break
            except (json.JSONDecodeError, TypeError):
                pass

            # It's an output_id — fetch the full output
            output_id = payload.strip('"')
            output_data = store.get(f"output:{output_id}")

            if output_data:
                output = json.loads(output_data)

                # Skip act narration — only collect final response
                if output.get('type') == 'act_narration':
                    continue

                # Blocks live inside output.metadata.blocks (same as WebSocket handler)
                metadata = output.get('metadata', {})
                response_blocks = metadata.get('blocks', [])
                _ = metadata.get('metadata', {})  # reserved for future use
                response_meta = {
                    'topic': output.get('topic', ''),
                    'mode': metadata.get('mode', ''),
                    'response': metadata.get('response', ''),
                }
                break

    pubsub.unsubscribe(sse_channel)
    pubsub.close()

    elapsed_ms = int((time.time() - start_time) * 1000)

    if not response_blocks:
        return jsonify({"error": "Timeout waiting for response"}), 504

    return jsonify({
        "blocks": response_blocks,
        "meta": response_meta,
        "duration_ms": elapsed_ms,
    })
