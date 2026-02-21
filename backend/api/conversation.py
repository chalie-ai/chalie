"""
Conversation blueprint — /chat (SSE), /conversation/recent, /conversation/summary.
"""

import json
import time
import uuid
import logging
import threading
from flask import Blueprint, request, jsonify, Response

from .auth import require_session
from .sse import sse_event, sse_keepalive, sse_retry, sse_headers

logger = logging.getLogger(__name__)

conversation_bp = Blueprint('conversation', __name__)


@conversation_bp.route('/chat', methods=['POST'])
@require_session
def chat_sse():
    """
    Submit a message and receive the response as an SSE stream.

    Request JSON: {"text": "...", "source": "text|voice", "attachments": []}

    SSE events: status, message, error, done
    """
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Missing 'text' field"}), 400

    source = data.get("source", "text")
    request_id = str(uuid.uuid4())

    def generate():
        from services.redis_client import RedisClientService
        from workers.digest_worker import digest_worker

        redis = RedisClientService.create_connection()
        pubsub = redis.pubsub()
        sse_channel = f"sse:{request_id}"
        pubsub.subscribe(sse_channel)

        # Initial SSE frame
        yield sse_retry()
        yield sse_event("status", {"stage": "processing"})

        # Track background thread completion
        bg_error = {}
        bg_done = threading.Event()

        def run_digest():
            try:
                digest_worker(text, metadata={
                    'uuid': request_id,
                    'source': source,
                })
            except Exception as e:
                logger.error(f"[SSE] digest_worker error for {request_id}: {e}", exc_info=True)
                bg_error['message'] = str(e)
                try:
                    redis.publish(sse_channel, json.dumps({"error": str(e)}))
                except Exception:
                    pass
            finally:
                bg_done.set()

        # Start processing in background thread
        thread = threading.Thread(target=run_digest, daemon=True)
        thread.start()

        yield sse_event("status", {"stage": "thinking"})

        # Listen for pub/sub events with keepalive and timeout
        start_time = time.time()
        timeout_seconds = 90
        keepalive_interval = 15
        status_interval = 20
        last_keepalive = time.time()
        last_status = time.time()
        message_received = False

        while time.time() - start_time < timeout_seconds:
            # Check for pub/sub messages (non-blocking, 1s poll)
            msg = pubsub.get_message(timeout=1.0)

            if msg and msg['type'] == 'message':
                payload = msg['data']
                if isinstance(payload, bytes):
                    payload = payload.decode()

                # Check if it's an error from the background thread
                try:
                    parsed = json.loads(payload)
                    if 'error' in parsed:
                        yield sse_event("error", {
                            "message": parsed['error'],
                            "recoverable": True
                        })
                        yield sse_event("done", {
                            "duration_ms": int((time.time() - start_time) * 1000)
                        })
                        break
                except (json.JSONDecodeError, TypeError):
                    pass

                # It's an output_id — fetch the full output
                output_id = payload.strip('"')
                output_data = redis.get(f"output:{output_id}")

                if output_data:
                    output = json.loads(output_data)
                    metadata = output.get("metadata", {})
                    message_data = {
                        "text": metadata.get("response", ""),
                        "topic": output.get("topic", ""),
                        "mode": metadata.get("mode", ""),
                        "confidence": metadata.get("confidence", 0),
                    }
                    # Include removed_by and removes if present
                    if "removed_by" in metadata:
                        message_data["removed_by"] = metadata["removed_by"]
                    if "removes" in metadata:
                        message_data["removes"] = metadata["removes"]
                    yield sse_event("message", message_data)
                    message_received = True
                    yield sse_event("done", {
                        "duration_ms": int((time.time() - start_time) * 1000)
                    })
                    break

            # Keepalive ping
            now = time.time()
            if now - last_keepalive >= keepalive_interval:
                yield sse_keepalive()
                last_keepalive = now

            # Status update
            if now - last_status >= status_interval:
                yield sse_event("status", {"stage": "still_working"})
                last_status = now

            # Fallback: if background thread is done but no pub/sub arrived
            if bg_done.is_set() and not message_received:
                time.sleep(0.5)  # Brief grace period
                # Try polling the output key directly
                output_key = f"output:{request_id}"
                fallback_data = redis.get(output_key)
                if fallback_data:
                    output = json.loads(fallback_data)
                    metadata = output.get("metadata", {})
                    message_data = {
                        "text": metadata.get("response", ""),
                        "topic": output.get("topic", ""),
                        "mode": metadata.get("mode", ""),
                        "confidence": metadata.get("confidence", 0),
                    }
                    # Include removed_by and removes if present
                    if "removed_by" in metadata:
                        message_data["removed_by"] = metadata["removed_by"]
                    if "removes" in metadata:
                        message_data["removes"] = metadata["removes"]
                    yield sse_event("message", message_data)
                elif bg_error:
                    yield sse_event("error", {
                        "message": bg_error.get('message', 'Processing failed'),
                        "recoverable": False
                    })
                else:
                    yield sse_event("error", {
                        "message": "No response received",
                        "recoverable": True
                    })
                yield sse_event("done", {
                    "duration_ms": int((time.time() - start_time) * 1000)
                })
                break
        else:
            # Timeout
            yield sse_event("error", {
                "message": "Request timed out",
                "recoverable": True
            })
            yield sse_event("done", {
                "duration_ms": int((time.time() - start_time) * 1000)
            })

        pubsub.unsubscribe(sse_channel)
        pubsub.close()

    headers = sse_headers()
    headers["X-Request-ID"] = request_id
    return Response(generate(), mimetype="text/event-stream", headers=headers)


@conversation_bp.route('/conversation/recent', methods=['GET'])
@require_session
def conversation_recent():
    """Return recent conversation from the current thread."""
    try:
        from services.thread_service import get_thread_service
        from services.thread_conversation_service import ThreadConversationService

        ts = get_thread_service()
        thread_id = ts.get_active_thread_id("default", "default")

        if not thread_id:
            return jsonify({"thread_id": None, "exchanges": []}), 200

        tcs = ThreadConversationService()
        exchanges = tcs.get_conversation_history(thread_id)

        formatted = []
        for ex in exchanges:
            prompt = ex.get("prompt", {}) or {}
            response = ex.get("response", {}) or {}
            formatted.append({
                "id": ex.get("id", ""),
                "prompt": prompt.get("message", "") if isinstance(prompt, dict) else "",
                "response": response.get("message", "") if isinstance(response, dict) else "",
                "topic": ex.get("topic", ""),
                "timestamp": ex.get("timestamp", ""),
            })

        return jsonify({"thread_id": thread_id, "exchanges": formatted}), 200

    except Exception as e:
        logger.error(f"[REST API] conversation/recent error: {e}", exc_info=True)
        return jsonify({"error": "Failed to retrieve conversation"}), 500


@conversation_bp.route('/conversation/summary', methods=['GET'])
@require_session
def conversation_summary():
    """Return compressed conversation summaries across time ranges."""
    try:
        from datetime import datetime, timedelta, timezone
        from services.thread_service import get_thread_service
        from services.gist_storage_service import GistStorageService
        from services.database_service import get_shared_db_service
        from services.episodic_retrieval_service import EpisodicRetrievalService
        from services.config_service import ConfigService

        ts = get_thread_service()
        thread_id = ts.get_active_thread_id("default", "default")

        result = {"today": [], "this_week": [], "older_highlights": []}

        # Today's gists from active topic
        if thread_id:
            from services.redis_client import RedisClientService
            redis = RedisClientService.create_connection()
            topic_data = redis.hgetall(f"thread:{thread_id}")
            current_topic = topic_data.get("current_topic", "") if topic_data else ""

            if current_topic:
                gist_service = GistStorageService()
                gists = gist_service.get_latest_gists(current_topic)
                for g in gists:
                    result["today"].append({
                        "content": g.get("content", ""),
                        "type": g.get("type", ""),
                        "timestamp": g.get("created_at", ""),
                    })

        # Episodes for this week and older
        try:
            db = get_shared_db_service()
            episodic_config = ConfigService.resolve_agent_config("episodic-memory")
            retrieval = EpisodicRetrievalService(db, episodic_config)
            episodes = retrieval.retrieve_episodes("conversation summary", limit=10)

            now = datetime.now(timezone.utc)
            week_ago = now - timedelta(days=7)

            for ep in episodes:
                created = ep.get("created_at")
                entry = {
                    "gist": ep.get("gist", ""),
                    "topic": ep.get("topic", ""),
                    "salience": ep.get("salience", 0),
                    "created_at": str(created) if created else "",
                }
                if created and hasattr(created, 'replace'):
                    # Normalize to tz-aware for comparison
                    created_aware = created if created.tzinfo else created.replace(tzinfo=timezone.utc)
                    if created_aware >= week_ago:
                        result["this_week"].append(entry)
                    else:
                        result["older_highlights"].append(entry)
                else:
                    result["this_week"].append(entry)
        except Exception as e:
            logger.warning(f"[REST API] Episode retrieval failed: {e}")

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"[REST API] conversation/summary error: {e}", exc_info=True)
        return jsonify({"error": "Failed to retrieve summary"}), 500
