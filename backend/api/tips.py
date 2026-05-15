"""
Tips blueprint — Quick Tip dismiss/mute endpoints.
"""

import logging

from flask import Blueprint, jsonify, request

from .auth import require_session

logger = logging.getLogger(__name__)

tips_bp = Blueprint('tips', __name__, url_prefix='/api/tips')


@tips_bp.route('/dismiss', methods=['POST'])
@require_session
def dismiss_tip():
    """Dismiss a specific tip so it won't be shown again.

    Request body::
        {"tip_id": "weather"}

    Response 204 on success.
    """
    try:
        data = request.get_json(silent=True) or {}
        tip_id = data.get('tip_id')
        if not tip_id:
            return jsonify({"error": "tip_id required"}), 400

        from services.quick_tip_service import QuickTipService
        svc = QuickTipService()
        svc.dismiss_tip(tip_id)
        return '', 204
    except Exception as exc:
        logger.error("[TIPS API] dismiss failed: %s", exc)
        return jsonify({"error": "Failed to dismiss tip"}), 500


@tips_bp.route('/mute', methods=['POST'])
@require_session
def mute_tips():
    """Disable all quick tips.

    Sets the quick_tips_enabled setting to false.
    Response 204 on success.
    """
    try:
        from services.quick_tip_service import QuickTipService
        svc = QuickTipService()
        svc.mute_tips()
        return '', 204
    except Exception as exc:
        logger.error("[TIPS API] mute failed: %s", exc)
        return jsonify({"error": "Failed to mute tips"}), 500
