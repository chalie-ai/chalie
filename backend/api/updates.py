"""
Updates API — State mutation endpoints for external wrappers and the chat UI.

Three classified update types, each with its own endpoint:

  POST /api/updates/belief  — set or correct a user trait
  POST /api/updates/memory  — encode an episodic/semantic memory
  POST /api/updates/feedback — report intent execution outcome

All endpoints require authentication (cookie session or bearer token).
Bearer-authenticated requests are additionally checked against the wrapper's
``permissions.update`` list.  Cookie-authenticated requests (the human user)
bypass permission checks entirely.
"""

import json
import logging
import uuid
from typing import TYPE_CHECKING

from flask import g, request

from flask_restx import Namespace, Resource
from .auth import require_auth
from services.log_utils import safe

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

updates_bp = Namespace("updates", description="State mutation endpoints", path="/api/updates")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_update_permission(update_type: str) -> "tuple[dict, int] | None":
    """Return a 403 response if the bearer wrapper lacks the required permission.

    For cookie-authenticated requests (``g.wrapper_id is None``) this function
    always returns ``None`` (permission granted).
    """
    wrapper_id = getattr(g, "wrapper_id", None)
    if wrapper_id is None:
        return None

    from services.wrapper_auth_service import WrapperAuthService
    svc = WrapperAuthService()
    if not svc.check_permission(wrapper_id, "update", update_type):
        return {"error": "Insufficient permissions"}, 403

    return None


def _get_db() -> object:
    from services.database_service import get_shared_db_service
    return get_shared_db_service()


# ---------------------------------------------------------------------------
# POST /api/updates/belief
# ---------------------------------------------------------------------------

@updates_bp.route("/belief")
@updates_bp.response(200, "Belief updated")
@updates_bp.response(400, "Missing key/value")
@updates_bp.response(403, "Insufficient permissions")
@updates_bp.response(422, "Rejected by validation rules")
class BeliefResource(Resource):
    @require_auth
    def post(self):
        """Set or correct a user trait (belief update)."""
        denied = _check_update_permission("belief")
        if denied is not None:
            return denied

        body = request.get_json(silent=True) or {}
        key = (body.get("key") or "").strip()
        value = body.get("value")

        if not key:
            return {"error": "key is required"}, 400
        if value is None or str(value).strip() == "":
            return {"error": "value is required"}, 400

        category = (body.get("category") or "preference").strip()
        if category not in {"core", "preference", "behavioral"}:
            category = "preference"

        try:
            confidence = float(body.get("confidence", 0.8))
        except (TypeError, ValueError):
            confidence = 0.8
        confidence = max(0.0, min(1.0, confidence))

        try:
            from services.data_graph_service import get_data_graph_service
            dgs = get_data_graph_service()
            stored = dgs.store(kind='user_specific', key=key, value=str(value), source='updates_api')
            if not stored:
                logger.info(
                    "[Updates API] belief update rejected: key=%r value=%r",
                    safe(key), safe(value),
                )
                return {"error": "Trait rejected by validation rules"}, 422
        except Exception as exc:
            logger.error("[Updates API] belief update failed: %s", exc)
            return {"error": "Internal error storing belief"}, 500

        logger.info("[Updates API] belief update: key=%r category=%r", key, category)
        return {"ok": True}, 200


# ---------------------------------------------------------------------------
# POST /api/updates/memory
# ---------------------------------------------------------------------------

@updates_bp.route("/memory")
@updates_bp.response(200, "Memory stored")
@updates_bp.response(400, "Missing content")
@updates_bp.response(403, "Insufficient permissions")
class MemoryResource(Resource):
    @require_auth
    def post(self):
        """Encode a memory (episodic/semantic)."""
        denied = _check_update_permission("memory")
        if denied is not None:
            return denied

        body = request.get_json(silent=True) or {}
        content = (body.get("content") or "").strip()
        if not content:
            return {"error": "content is required"}, 400

        topic = (body.get("topic") or "general").strip() or "general"

        try:
            salience = int(body.get("salience", 5))
            salience = max(1, min(10, salience))
        except (TypeError, ValueError):
            salience = 5

        try:
            from services.memory_retrieval import handle_store
            result = handle_store(
                topic,
                {
                    "action": "store",
                    "key": f"memory_{uuid.uuid4().hex[:8]}",
                    "value": content,
                    "kind": "misc",
                },
            )
            if result.status == "error":
                logger.warning(
                    "[Updates API] memory update failed: code=%s body=%s",
                    result.code, result.body,
                )
                return {"error": "Memory encoding failed"}, 422
        except Exception as exc:
            logger.error("[Updates API] memory update failed: %s", exc)
            return {"error": "Internal error encoding memory"}, 500

        logger.info("[Updates API] memory update: topic=%r salience=%d", topic, salience)
        return {"ok": True}, 200


# ---------------------------------------------------------------------------
# POST /api/updates/feedback
# ---------------------------------------------------------------------------

@updates_bp.route("/feedback")
@updates_bp.response(200, "Feedback recorded")
@updates_bp.response(400, "Missing intent_id/outcome")
@updates_bp.response(403, "Insufficient permissions")
class FeedbackResource(Resource):
    @require_auth
    def post(self):
        """Report intent execution outcome (feedback update)."""
        denied = _check_update_permission("feedback")
        if denied is not None:
            return denied

        body = request.get_json(silent=True) or {}
        intent_id = (body.get("intent_id") or "").strip()
        outcome = (body.get("outcome") or "").strip()

        if not intent_id:
            return {"error": "intent_id is required"}, 400
        if not outcome:
            return {"error": "outcome is required"}, 400

        details = (body.get("details") or "").strip()

        try:
            from services.memory_client import MemoryClientService
            from services.time_utils import utc_now

            store = MemoryClientService.create_connection()
            feedback_key = f"intent_feedback:{intent_id}"
            record = {
                "intent_id": intent_id,
                "outcome": outcome,
                "details": details,
                "recorded_at": utc_now().isoformat(),
            }
            store.set(feedback_key, json.dumps(record), ex=86400)
        except Exception as exc:
            logger.error("[Updates API] feedback update failed: %s", exc)
            return {"error": "Internal error recording feedback"}, 500

        logger.info("[Updates API] feedback update: intent_id=%r outcome=%r", intent_id, outcome)
        return {"ok": True}, 200
