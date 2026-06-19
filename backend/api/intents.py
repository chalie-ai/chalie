"""
Authentication:
  Cookie session (chat UI):  wrapper_id is ``'__chat_ui__'``
  Bearer token (wrappers):   wrapper_id comes from ``g.wrapper_id``
"""

import logging
from typing import TYPE_CHECKING

from flask import Blueprint, g, jsonify, request

from .auth import require_session
from services.intent_service import IntentService

if TYPE_CHECKING:
    from flask.typing import ResponseReturnValue

logger = logging.getLogger(__name__)

_ERR_INTENT_NOT_FOUND = "intent not found"

intents_bp = Blueprint("intents", __name__, url_prefix="/api/intents")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _effective_wrapper_id() -> str:
    wid = getattr(g, "wrapper_id", None)
    return wid if wid else "__chat_ui__"


def _get_intent_service() -> IntentService:
    return IntentService()


# ---------------------------------------------------------------------------
# GET /api/intents — list pending intents
# ---------------------------------------------------------------------------

@intents_bp.route("", methods=["GET"])
@require_session
def list_intents() -> "ResponseReturnValue":
    wrapper_id = _effective_wrapper_id()

    try:
        limit = int(request.args.get("limit", 10))
    except (TypeError, ValueError):
        return jsonify({"error": "limit must be an integer"}), 400

    limit = max(1, min(limit, 100))  # clamp to [1, 100]

    svc = _get_intent_service()
    intents = svc.get_pending(wrapper_id, limit=limit)

    return jsonify({"intents": intents, "wrapper_id": wrapper_id}), 200


# ---------------------------------------------------------------------------
# GET /api/intents/<id> — get single intent
# ---------------------------------------------------------------------------

@intents_bp.route("/<intent_id>", methods=["GET"])
@require_session
def get_intent(intent_id: str) -> "ResponseReturnValue":
    """Returns 404 if the intent does not exist or has expired from the store."""
    svc = _get_intent_service()
    intent = svc.get_intent(intent_id)

    if intent is None:
        return jsonify({"error": _ERR_INTENT_NOT_FOUND}), 404

    return jsonify({"intent": intent}), 200


# ---------------------------------------------------------------------------
# POST /api/intents/<id>/ack — acknowledge receipt
# ---------------------------------------------------------------------------

@intents_bp.route("/<intent_id>/ack", methods=["POST"])
@require_session
def acknowledge_intent(intent_id: str) -> "ResponseReturnValue":
    """Returns 404 if the intent does not exist."""
    wrapper_id = _effective_wrapper_id()
    svc = _get_intent_service()
    ok = svc.acknowledge(intent_id, wrapper_id)

    if not ok:
        return jsonify({"error": _ERR_INTENT_NOT_FOUND}), 404

    return jsonify({"ok": True, "intent_id": intent_id}), 200


# ---------------------------------------------------------------------------
# POST /api/intents/<id>/resolve — report execution result
# ---------------------------------------------------------------------------

@intents_bp.route("/<intent_id>/resolve", methods=["POST"])
@require_session
def resolve_intent(intent_id: str) -> "ResponseReturnValue":
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "request body must be a JSON object"}), 400

    status = body.get("status", "").strip()
    if status not in ("executed", "failed", "skipped"):
        return jsonify({
            "error": "status must be one of: executed, failed, skipped"
        }), 400

    svc = _get_intent_service()
    ok = svc.resolve(intent_id, body)

    if not ok:
        return jsonify({"error": _ERR_INTENT_NOT_FOUND}), 404

    return jsonify({"ok": True, "intent_id": intent_id, "status": status}), 200
