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

from flask import Blueprint, g, jsonify, request

from .auth import require_auth
from services.log_utils import safe

logger = logging.getLogger(__name__)

updates_bp = Blueprint("updates", __name__, url_prefix="/api/updates")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_update_permission(update_type: str):
    """Return a 403 response if the bearer wrapper lacks the required permission.

    For cookie-authenticated requests (``g.wrapper_id is None``) this function
    always returns ``None`` (permission granted).

    Args:
        update_type: One of ``"belief"``, ``"memory"``, ``"context"``,
            or ``"feedback"``.

    Returns:
        ``None`` if the caller is permitted, or a ``(response, 403)`` tuple
        that the endpoint should return immediately.
    """
    wrapper_id = getattr(g, "wrapper_id", None)
    if wrapper_id is None:
        # Cookie (human) auth — all updates allowed
        return None

    from services.wrapper_auth_service import WrapperAuthService
    svc = WrapperAuthService()
    if not svc.check_permission(wrapper_id, "update", update_type):
        return jsonify({"error": "Insufficient permissions"}), 403

    return None


def _get_db():
    from services.database_service import get_shared_db_service
    return get_shared_db_service()


# ---------------------------------------------------------------------------
# POST /api/updates/belief
# ---------------------------------------------------------------------------

@updates_bp.route("/belief", methods=["POST"])
@require_auth
def update_belief():
    """Set or correct a user trait (belief update).

    Stores the trait in DataGraphService as kind='user_specific'. The ``key``
    and ``value`` fields are required.

    Body (JSON):
        key (str, required): Trait identifier, e.g. ``"risk_tolerance"``.
        value (str, required): Trait value, e.g. ``"conservative"``.
        category (str): Ignored (kept for API compat). Defaults to ``preference``.
        confidence (float): Ignored (DataGraphService manages retrieval_weight).
            Defaults to ``0.8``.

    Returns:
        200 with ``{ok: true}`` on success.
        400 if ``key`` or ``value`` is missing.
        403 if bearer token lacks the ``update/belief`` permission.
    """
    denied = _check_update_permission("belief")
    if denied is not None:
        return denied

    body = request.get_json(silent=True) or {}
    key = (body.get("key") or "").strip()
    value = body.get("value")

    if not key:
        return jsonify({"error": "key is required"}), 400
    if value is None or str(value).strip() == "":
        return jsonify({"error": "value is required"}), 400

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
            return jsonify({"error": "Trait rejected by validation rules"}), 422
    except Exception as exc:
        logger.error("[Updates API] belief update failed: %s", exc)
        return jsonify({"error": "Internal error storing belief"}), 500

    logger.info("[Updates API] belief update: key=%r category=%r", key, category)
    return jsonify({"ok": True}), 200


# ---------------------------------------------------------------------------
# POST /api/updates/memory
# ---------------------------------------------------------------------------

@updates_bp.route("/memory", methods=["POST"])
@require_auth
def update_memory():
    """Encode a memory (episodic/semantic).

    Stores the provided content as a user trait with category ``behavioral``
    under a deterministic key derived from the topic.  This is a thin surface
    that delegates to the memorize skill path.

    Body (JSON):
        content (str, required): Narrative content to remember.
        topic (str): Topic label.  Defaults to ``"general"``.
        salience (int): 1–10 importance score.  Defaults to ``5``.

    Returns:
        200 with ``{ok: true}`` on success.
        400 if ``content`` is missing.
        403 if bearer token lacks the ``update/memory`` permission.
    """
    denied = _check_update_permission("memory")
    if denied is not None:
        return denied

    body = request.get_json(silent=True) or {}
    content = (body.get("content") or "").strip()
    if not content:
        return jsonify({"error": "content is required"}), 400

    topic = (body.get("topic") or "general").strip() or "general"

    try:
        salience = int(body.get("salience", 5))
        salience = max(1, min(10, salience))
    except (TypeError, ValueError):
        salience = 5

    try:
        # External REST caller — there is no ACT-loop MessageProcessor here, so
        # the memory ability's run() (which reads its channel from the bound
        # processor) does not apply. Call the store primitive directly, passing
        # ``topic`` as the provenance channel for the source tag.
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
            return jsonify({"error": "Memory encoding failed"}), 422
    except Exception as exc:
        logger.error("[Updates API] memory update failed: %s", exc)
        return jsonify({"error": "Internal error encoding memory"}), 500

    logger.info("[Updates API] memory update: topic=%r salience=%d", topic, salience)
    return jsonify({"ok": True}), 200


# ---------------------------------------------------------------------------
# POST /api/updates/feedback
# ---------------------------------------------------------------------------

@updates_bp.route("/feedback", methods=["POST"])
@require_auth
def update_feedback():
    """Report intent execution outcome (feedback update).

    Stores the outcome in MemoryStore for later consumption by the experience
    assimilation pipeline (WS3 intents integration).  The record is keyed by
    ``intent_id`` with a short TTL so stale feedback doesn't accumulate.

    Body (JSON):
        intent_id (str, required): Opaque identifier for the intent that was
            executed.
        outcome (str, required): ``"success"``, ``"failure"``, or any string.
        details (str): Free-form description of what happened.

    Returns:
        200 with ``{ok: true}`` on success.
        400 if ``intent_id`` or ``outcome`` is missing.
        403 if bearer token lacks the ``update/feedback`` permission.
    """
    denied = _check_update_permission("feedback")
    if denied is not None:
        return denied

    body = request.get_json(silent=True) or {}
    intent_id = (body.get("intent_id") or "").strip()
    outcome = (body.get("outcome") or "").strip()

    if not intent_id:
        return jsonify({"error": "intent_id is required"}), 400
    if not outcome:
        return jsonify({"error": "outcome is required"}), 400

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
        # 24-hour TTL — long enough for assimilation workers to consume it
        store.set(feedback_key, json.dumps(record), ex=86400)
    except Exception as exc:
        logger.error("[Updates API] feedback update failed: %s", exc)
        return jsonify({"error": "Internal error recording feedback"}), 500

    logger.info("[Updates API] feedback update: intent_id=%r outcome=%r", intent_id, outcome)
    return jsonify({"ok": True}), 200
