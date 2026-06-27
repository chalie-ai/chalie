"""
Authentication:
  Cookie session (chat UI):  wrapper_id is ``'__chat_ui__'``
  Bearer token (wrappers):   wrapper_id comes from ``g.wrapper_id``
"""

import logging
from typing import TYPE_CHECKING

from flask import g, request
from flask.typing import ResponseReturnValue

from flask_restx import Namespace, Resource
from .auth import require_session
from services.intent_service import IntentService

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_ERR_INTENT_NOT_FOUND = "intent not found"

intents_bp = Namespace("intents", description="Intent management", path="/api/intents")


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

@intents_bp.route("")
@intents_bp.response(200, "List of pending intents")
@intents_bp.response(400, "Invalid limit")
class IntentListResource(Resource):
    @require_session
    def get(self) -> ResponseReturnValue:
        wrapper_id = _effective_wrapper_id()

        try:
            limit = int(request.args.get("limit", 10))
        except (TypeError, ValueError):
            return {"error": "limit must be an integer"}, 400

        limit = max(1, min(limit, 100))

        svc = _get_intent_service()
        intents = svc.get_pending(wrapper_id, limit=limit)

        return {"intents": intents, "wrapper_id": wrapper_id}, 200


# ---------------------------------------------------------------------------
# GET /api/intents/<id> — get single intent
# ---------------------------------------------------------------------------

@intents_bp.route("/<intent_id>")
@intents_bp.response(200, "Intent details")
@intents_bp.response(404, "Intent not found")
@intents_bp.param("intent_id", "str", "Intent identifier")
class IntentDetailResource(Resource):
    @require_session
    def get(self, intent_id: str) -> ResponseReturnValue:
        """Returns 404 if the intent does not exist or has expired."""
        svc = _get_intent_service()
        intent = svc.get_intent(intent_id)

        if intent is None:
            return {"error": _ERR_INTENT_NOT_FOUND}, 404

        return {"intent": intent}, 200


# ---------------------------------------------------------------------------
# POST /api/intents/<id>/ack — acknowledge receipt
# ---------------------------------------------------------------------------

@intents_bp.route("/<intent_id>/ack")
@intents_bp.response(200, "Intent acknowledged")
@intents_bp.response(404, "Intent not found")
@intents_bp.param("intent_id", "str", "Intent identifier")
class IntentAckResource(Resource):
    @require_session
    def post(self, intent_id: str) -> ResponseReturnValue:
        """Acknowledge receipt of an intent."""
        wrapper_id = _effective_wrapper_id()
        svc = _get_intent_service()
        ok = svc.acknowledge(intent_id, wrapper_id)

        if not ok:
            return {"error": _ERR_INTENT_NOT_FOUND}, 404

        return {"ok": True, "intent_id": intent_id}, 200


# ---------------------------------------------------------------------------
# POST /api/intents/<id>/resolve — report execution result
# ---------------------------------------------------------------------------

@intents_bp.route("/<intent_id>/resolve")
@intents_bp.response(200, "Intent resolved")
@intents_bp.response(400, "Invalid request")
@intents_bp.response(404, "Intent not found")
@intents_bp.param("intent_id", "str", "Intent identifier")
class IntentResolveResource(Resource):
    @require_session
    def post(self, intent_id: str) -> ResponseReturnValue:
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return {"error": "request body must be a JSON object"}, 400

        status = body.get("status", "").strip()
        if status not in ("executed", "failed", "skipped"):
            return {
                "error": "status must be one of: executed, failed, skipped"
            }, 400

        svc = _get_intent_service()
        ok = svc.resolve(intent_id, body)

        if not ok:
            return {"error": _ERR_INTENT_NOT_FOUND}, 404

        return {"ok": True, "intent_id": intent_id, "status": status}, 200
