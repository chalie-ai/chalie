"""
Proactive blueprint — REST endpoint for recent notifications.

Real-time push is handled by the WebSocket (/ws). This endpoint provides
a REST fallback for fetching buffered notifications (e.g., catch-up on
page load before WebSocket connects).
"""

import json
import logging
from flask import Blueprint, request, jsonify

logger = logging.getLogger(__name__)

proactive_bp = Blueprint('proactive', __name__)


@proactive_bp.route('/events/recent', methods=['GET'])
def recent_events():
    """Return buffered notifications as JSON array.

    Drains the notifications:recent list so events aren't delivered twice.
    The WebSocket handler also drains this list on connect.
    """
    from services.auth_session_service import validate_session

    if not validate_session(request):
        return jsonify({"error": "Unauthorized"}), 401

    from services.redis_client import RedisClientService
    redis = RedisClientService.create_connection()

    events = []
    while True:
        item = redis.lpop('notifications:recent')
        if not item:
            break
        try:
            events.append(json.loads(item))
        except (json.JSONDecodeError, TypeError):
            pass

    return jsonify(events)
