"""
Wrappers API — CRUD endpoints for external wrapper bearer tokens.

Routes:
  POST   /api/wrappers                    — create a wrapper token (cookie auth only)
  GET    /api/wrappers                    — list active wrappers
  GET    /api/wrappers/<id>              — get wrapper details + capabilities
  PUT    /api/wrappers/<id>/capabilities — update capabilities (bearer, own wrapper only)
  DELETE /api/wrappers/<id>             — revoke wrapper token

The POST route intentionally restricts token creation to cookie-authenticated
sessions: only a human sitting in front of the UI should be able to mint new
wrapper credentials.  All other routes accept both cookie and bearer auth.
"""

import logging

from flask import Blueprint, g, jsonify, request

from .auth import require_auth
from services.log_utils import safe

logger = logging.getLogger(__name__)

_ERR_WRAPPER_NOT_FOUND = "Wrapper not found"

wrappers_bp = Blueprint("wrappers", __name__, url_prefix="/api/wrappers")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_service():
    from services.database_service import get_shared_db_service
    from services.wrapper_auth_service import WrapperAuthService
    return WrapperAuthService(get_shared_db_service())


def _cookie_only(f):
    """Restrict a route to cookie-authenticated requests (no bearer tokens).

    Used on the token creation endpoint so that only humans (not wrappers
    themselves) can mint new credentials.
    """
    from functools import wraps

    @wraps(f)
    def decorated(*args, **kwargs):
        if getattr(g, "wrapper_id", None) is not None:
            return jsonify({"error": "Token creation requires cookie session auth"}), 403
        return f(*args, **kwargs)

    return decorated


# ---------------------------------------------------------------------------
# POST /api/wrappers — create wrapper token
# ---------------------------------------------------------------------------

@wrappers_bp.route("", methods=["POST"])
@require_auth
@_cookie_only
def create_wrapper():
    """Create a new external wrapper token.

    Body (JSON):
        name (str, required): Human-readable label.
        capabilities (dict, optional): Signals/intents the wrapper may emit.
        permissions (dict, optional): API operations the wrapper may call.
        metadata (dict, optional): Version, description, etc.

    Returns:
        201 with ``{wrapper_id, token}`` — ``token`` is shown **once** and not
        recoverable.
        400 if ``name`` is missing or blank.
    """
    body = request.get_json(silent=True) or {}
    name = (body.get("name") or "").strip()
    if not name:
        return jsonify({"error": "name is required"}), 400

    capabilities = body.get("capabilities") or {}
    permissions = body.get("permissions") or {}
    metadata = body.get("metadata") or {}

    # Accept list or dict for capabilities — stored as JSON metadata only,
    # never used in permission-checking logic.
    if not isinstance(capabilities, (dict, list)):
        return jsonify({"error": "capabilities must be a JSON object or array"}), 400
    if not isinstance(permissions, dict):
        return jsonify({"error": "permissions must be a JSON object"}), 400
    if not isinstance(metadata, dict):
        return jsonify({"error": "metadata must be a JSON object"}), 400

    svc = _get_service()
    raw_token, wrapper_id = svc.create_token(
        name=name,
        capabilities=capabilities,
        permissions=permissions,
        metadata=metadata,
    )

    logger.info("[Wrappers API] Created wrapper token: %s name=%r", wrapper_id, name)
    return jsonify({"wrapper_id": wrapper_id, "token": raw_token}), 201


# ---------------------------------------------------------------------------
# GET /api/wrappers — list active wrappers
# ---------------------------------------------------------------------------

@wrappers_bp.route("", methods=["GET"])
@require_auth
def list_wrappers():
    """List all non-revoked wrapper tokens.

    Returns:
        200 with ``{wrappers: [...]}`` where each item contains ``wrapper_id``,
        ``name``, ``capabilities``, ``permissions``, ``metadata``,
        ``last_seen_at``, and ``created_at``.
    """
    svc = _get_service()
    wrappers = svc.list_wrappers()
    return jsonify({"wrappers": wrappers}), 200


# ---------------------------------------------------------------------------
# GET /api/wrappers/<id> — get wrapper details
# ---------------------------------------------------------------------------

@wrappers_bp.route("/<wrapper_id>", methods=["GET"])
@require_auth
def get_wrapper(wrapper_id: str):
    """Get details for a single wrapper token.

    Args:
        wrapper_id: The stable wrapper identifier (path parameter).

    Returns:
        200 with the wrapper dict, or 404 if not found / revoked.
    """
    svc = _get_service()
    wrapper = svc.get_wrapper(wrapper_id)
    if wrapper is None:
        return jsonify({"error": _ERR_WRAPPER_NOT_FOUND}), 404
    return jsonify(wrapper), 200


# ---------------------------------------------------------------------------
# PUT /api/wrappers/<id>/capabilities — update capabilities
# ---------------------------------------------------------------------------

@wrappers_bp.route("/<wrapper_id>/capabilities", methods=["PUT"])
@require_auth
def update_capabilities(wrapper_id: str):
    """Replace the capabilities payload for a wrapper.

    Bearers may only update their **own** wrapper (``g.wrapper_id`` must match).
    Cookie-authenticated requests (human in UI) may update any wrapper.

    Args:
        wrapper_id: The stable wrapper identifier (path parameter).

    Body (JSON):
        capabilities (dict, required): New capabilities payload.

    Returns:
        200 with ``{ok: true}`` on success.
        400 if ``capabilities`` is missing or not an object.
        403 if a bearer token attempts to update another wrapper.
        404 if the wrapper does not exist or is revoked.
    """
    # Bearer auth: can only update own wrapper
    caller_id = getattr(g, "wrapper_id", None)
    if caller_id is not None and caller_id != wrapper_id:
        return jsonify({"error": "Bearer token may only update its own capabilities"}), 403

    body = request.get_json(silent=True) or {}
    capabilities = body.get("capabilities")
    if capabilities is None or not isinstance(capabilities, (dict, list)):
        return jsonify({"error": "capabilities must be a JSON object or array"}), 400

    svc = _get_service()
    updated = svc.update_capabilities(wrapper_id, capabilities)
    if not updated:
        return jsonify({"error": _ERR_WRAPPER_NOT_FOUND}), 404

    return jsonify({"ok": True}), 200


# ---------------------------------------------------------------------------
# DELETE /api/wrappers/<id> — revoke token
# ---------------------------------------------------------------------------

@wrappers_bp.route("/<wrapper_id>", methods=["DELETE"])
@require_auth
def revoke_wrapper(wrapper_id: str):
    """Revoke a wrapper token.

    Args:
        wrapper_id: The stable wrapper identifier (path parameter).

    Returns:
        200 with ``{ok: true}`` on success.
        404 if the wrapper does not exist or is already revoked.
    """
    svc = _get_service()
    revoked = svc.revoke(wrapper_id)
    if not revoked:
        return jsonify({"error": _ERR_WRAPPER_NOT_FOUND}), 404

    logger.info("[Wrappers API] Revoked wrapper: %s", safe(wrapper_id))
    return jsonify({"ok": True}), 200
