"""
Proactive blueprint — /events/stream endpoint.
"""

import json
import logging
from flask import Blueprint, request, Response, jsonify

logger = logging.getLogger(__name__)

proactive_bp = Blueprint('proactive', __name__)

OUTPUT_CHANNEL = 'output:events'


@proactive_bp.route('/events/stream', methods=['GET'])
def events_stream():
    """SSE stream for all async output: tool follow-ups, delegate results, drift.

    Uses session cookies for authentication. EventSource automatically sends
    cookies for same-origin requests.
    """
    from services.auth_session_service import validate_session

    if not validate_session(request):
        return Response("Unauthorized", status=401)

    def generate():
        from services.redis_client import RedisClientService

        redis = RedisClientService.create_connection()

        # Drain any buffered notifications missed during a reconnect gap.
        # Events published to output:events while the stream was down are lost
        # from pub/sub; the notifications:recent list catches them up.
        while True:
            item = redis.lpop('notifications:recent')
            if not item:
                break
            try:
                data = json.loads(item)
                event_type = data.get('type', 'message')
                yield f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
            except Exception:
                pass

        pubsub = redis.pubsub()
        pubsub.subscribe(OUTPUT_CHANNEL)

        # Spark: send first-contact welcome if needed (non-blocking)
        try:
            from services.spark_welcome_service import SparkWelcomeService
            SparkWelcomeService().maybe_send_welcome()
        except Exception as e:
            logger.debug(f"Spark welcome check failed (non-fatal): {e}")

        # Initial retry directive
        yield f"retry: 15000\n\n"

        try:
            while True:
                message = pubsub.get_message(timeout=15)
                if message and message['type'] == 'message':
                    try:
                        data = json.loads(message['data'])
                        event_type = data.get('type', 'message')
                        yield f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
                    except (json.JSONDecodeError, TypeError):
                        pass
                else:
                    # Keepalive
                    yield ": keepalive\n\n"
        except GeneratorExit:
            pubsub.unsubscribe(OUTPUT_CHANNEL)
            pubsub.close()

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        },
    )
