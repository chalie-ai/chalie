"""
Personality blueprint — read and write the 5-axis personality tuple.
"""

import logging

from flask import Blueprint, jsonify, request

from .auth import require_session

logger = logging.getLogger(__name__)

personality_bp = Blueprint('personality', __name__, url_prefix='/settings')


@personality_bp.route('/personality', methods=['GET'])
@require_session
def get_personality():
    """Return the current personality tuple and voice paragraph.

    Response 200::

        {"tuple": [w, m, e, c, h], "voice": "..."}
    """
    try:
        from services.personality.personality_service import get_current_tuple, get_current_voice

        tup = get_current_tuple()
        return jsonify({"tuple": list(tup), "voice": get_current_voice()}), 200
    except Exception as exc:
        logger.error("[REST API] Failed to get personality: %s", exc)
        return jsonify({"error": "Failed to get personality"}), 500


@personality_bp.route('/personality', methods=['PUT'])
@require_session
def set_personality():
    """Update the personality tuple.

    Request body::

        {"tuple": [w, m, e, c, h]}

    Each element must be an integer ∈ {-2, -1, 0, +1, +2}.

    Response 200::

        {"tuple": [w, m, e, c, h], "voice": "..."}

    Response 400::

        {"error": "..."}
    """
    try:
        data = request.get_json(silent=True) or {}
        raw = data.get('tuple')

        if raw is None:
            return jsonify({"error": "Missing required field: tuple"}), 400
        if not isinstance(raw, list) or len(raw) != 5:
            return jsonify({"error": "Field 'tuple' must be a list of exactly 5 integers"}), 400
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in raw):
            return jsonify({"error": "All tuple elements must be integers"}), 400

        from services.personality.personality_service import set_current_tuple

        voice = set_current_tuple(tuple(raw))
        return jsonify({"tuple": raw, "voice": voice}), 200
    except ValueError as exc:
        logger.warning("[REST API] Personality validation error: %s", exc)
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error("[REST API] Failed to set personality: %s", exc)
        return jsonify({"error": "Failed to set personality"}), 500
