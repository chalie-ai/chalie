"""
Updates API — State mutation endpoints for external wrappers and the chat UI.

Four classified update types, each with its own endpoint:

  POST /api/updates/belief  — set or correct a user trait
  POST /api/updates/memory  — encode an episodic/semantic memory
  POST /api/updates/context — update ambient/session context signals
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
    """Return the shared DatabaseService."""
    from services.database_service import get_shared_db_service
    return get_shared_db_service()


# ---------------------------------------------------------------------------
# POST /api/updates/belief
# ---------------------------------------------------------------------------

@updates_bp.route("/belief", methods=["POST"])
@require_auth
def update_belief():
    """Set or correct a user trait (belief update).

    Delegates directly to :meth:`UserTraitService.store_trait`.  The ``key``
    and ``value`` fields are required; ``category`` and ``confidence`` are
    optional with sensible defaults.

    Body (JSON):
        category (str): One of ``core``, ``preference``, or ``behavioral``.
            Defaults to ``preference``.
        key (str, required): Trait identifier, e.g. ``"risk_tolerance"``.
        value (str, required): Trait value, e.g. ``"conservative"``.
        confidence (float): 0.0–1.0.  Defaults to ``0.8``.

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
        from services.knowledge_service import KnowledgeService
        _DECAY_MAP = {'core': 'permanent', 'behavioral': 'slow'}
        ks = KnowledgeService(_get_db())
        stored = ks.store(
            kind='trait', entity='user', key=key, value=str(value),
            data={'category': category},
            decay_class=_DECAY_MAP.get(category, 'standard'),
            confidence=confidence, source='updates_api',
        )
        if not stored:
            logger.info(
                "[Updates API] belief update rejected by trait validation: key=%r value=%r",
                key, value,
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
        from services.innate_skills.memorize_skill import handle_memorize
        confidence = 0.5 + (salience / 20.0)  # 0.55 – 1.0 range
        result = handle_memorize(
            topic=topic,
            params={
                "traits": [
                    {
                        "key": f"memory_{uuid.uuid4().hex[:8]}",
                        "value": content,
                        "confidence": confidence,
                        "category": "behavioral",
                    }
                ]
            },
        )
        if "Error" in result:
            logger.warning("[Updates API] memory update result: %s", result)
            return jsonify({"error": "Memory encoding failed"}), 422
    except Exception as exc:
        logger.error("[Updates API] memory update failed: %s", exc)
        return jsonify({"error": "Internal error encoding memory"}), 500

    logger.info("[Updates API] memory update: topic=%r salience=%d", topic, salience)
    return jsonify({"ok": True}), 200


# ---------------------------------------------------------------------------
# POST /api/updates/context
# ---------------------------------------------------------------------------

@updates_bp.route("/context", methods=["POST"])
@require_auth
def update_context():
    """Update ambient/session context signals.

    Merges the provided signals into the current client context stored in
    MemoryStore.  Standard keys (``place``, ``energy``) are mapped to the
    expected context structure; arbitrary key-value pairs may be passed via
    ``custom`` and are stored under an ``external_signals`` sub-key.

    Body (JSON):
        place (str): Human-readable place label, e.g. ``"office"``.
        energy (str): Energy level, e.g. ``"high"``, ``"low"``.
        custom (dict): Arbitrary context signals.

    Returns:
        200 with ``{ok: true}`` on success.
        400 if none of ``place``, ``energy``, or ``custom`` are provided.
        403 if bearer token lacks the ``update/context`` permission.
    """
    denied = _check_update_permission("context")
    if denied is not None:
        return denied

    body = request.get_json(silent=True) or {}
    place = body.get("place")
    energy = body.get("energy")
    custom = body.get("custom")

    if place is None and energy is None and custom is None:
        return jsonify({"error": "At least one of place, energy, or custom is required"}), 400

    if custom is not None and not isinstance(custom, dict):
        return jsonify({"error": "custom must be a JSON object"}), 400

    try:
        from services.memory_client import MemoryClientService

        store = MemoryClientService.create_connection()
        STORE_KEY = "client_context:primary"
        TTL = 3600

        raw = store.get(STORE_KEY)
        ctx = json.loads(raw) if raw else {}

        if place is not None:
            ctx["place"] = str(place)
        if energy is not None:
            ctx["energy"] = str(energy)
        if custom:
            existing = ctx.get("external_signals") or {}
            existing.update(custom)
            ctx["external_signals"] = existing

        store.set(STORE_KEY, json.dumps(ctx), ex=TTL)
    except Exception as exc:
        logger.error("[Updates API] context update failed: %s", exc)
        return jsonify({"error": "Internal error updating context"}), 500

    logger.info("[Updates API] context update: place=%r energy=%r custom_keys=%s",
                place, energy, list((custom or {}).keys()))
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
