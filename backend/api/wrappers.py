import logging
from typing import TYPE_CHECKING, cast

from flask import g, request
from flask.typing import ResponseReturnValue

from flask_restx import Namespace, Resource
from .auth import require_auth, _cookie_only
from services.log_utils import safe

if TYPE_CHECKING:
    from services.wrapper_auth_service import WrapperAuthService

logger = logging.getLogger(__name__)

_ERR_WRAPPER_NOT_FOUND = "Wrapper not found"

wrappers_ns = Namespace("wrappers", description="Wrapper token management", path="/api/wrappers")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_service() -> "WrapperAuthService":
    from services.database_service import get_shared_db_service
    from services.wrapper_auth_service import WrapperAuthService
    return WrapperAuthService(get_shared_db_service())


# ---------------------------------------------------------------------------
# POST /api/wrappers — create wrapper token
# ---------------------------------------------------------------------------

@wrappers_ns.route("")
@wrappers_ns.response(201, "Wrapper created")
@wrappers_ns.response(200, "List of wrappers")
class WrapperRootResource(Resource):
    @require_auth
    @_cookie_only
    def post(self) -> ResponseReturnValue:
        """Create a new external wrapper token."""
        body = request.get_json(silent=True) or {}
        name = (body.get("name") or "").strip()
        if not name:
            return {"error": "name is required"}, 400

        capabilities = body.get("capabilities") or {}
        permissions = body.get("permissions") or {}
        metadata = body.get("metadata") or {}

        if not isinstance(capabilities, (dict, list)):
            return {"error": "capabilities must be a JSON object or array"}, 400
        if not isinstance(permissions, dict):
            return {"error": "permissions must be a JSON object"}, 400
        if not isinstance(metadata, dict):
            return {"error": "metadata must be a JSON object"}, 400

        svc = _get_service()
        raw_token, wrapper_id = svc.create_token(
            name=name,
            capabilities=cast("dict[str, object]", capabilities),
            permissions=cast("dict[str, object]", permissions),
            metadata=cast("dict[str, object]", metadata),
        )

        logger.info("[Wrappers API] Created wrapper token: %s name=%r", wrapper_id, name)
        return {"wrapper_id": wrapper_id, "token": raw_token}, 201

    @require_auth
    def get(self) -> ResponseReturnValue:
        """List all non-revoked wrapper tokens."""
        svc = _get_service()
        wrappers = svc.list_wrappers()
        return {"wrappers": wrappers}, 200


# ---------------------------------------------------------------------------
# GET /api/wrappers/<id> — get wrapper details
# DELETE /api/wrappers/<id> — revoke token
# ---------------------------------------------------------------------------

@wrappers_ns.route("/<wrapper_id>")
@wrappers_ns.response(200, "Wrapper details")
@wrappers_ns.response(404, "Wrapper not found")
@wrappers_ns.param("wrapper_id", "str", "Wrapper identifier")
class WrapperItemResource(Resource):
    @require_auth
    def get(self, wrapper_id: str) -> ResponseReturnValue:
        """Get details for a single wrapper token."""
        svc = _get_service()
        wrapper = svc.get_wrapper(wrapper_id)
        if wrapper is None:
            return {"error": _ERR_WRAPPER_NOT_FOUND}, 404
        return wrapper, 200

    @require_auth
    def delete(self, wrapper_id: str) -> ResponseReturnValue:
        """Revoke a wrapper token."""
        svc = _get_service()
        revoked = svc.revoke(wrapper_id)
        if not revoked:
            return {"error": _ERR_WRAPPER_NOT_FOUND}, 404

        logger.info("[Wrappers API] Revoked wrapper: %s", safe(wrapper_id))
        return {"ok": True}, 200


# ---------------------------------------------------------------------------
# PUT /api/wrappers/<id>/capabilities — update capabilities
# ---------------------------------------------------------------------------

@wrappers_ns.route("/<wrapper_id>/capabilities")
@wrappers_ns.response(200, "Capabilities updated")
@wrappers_ns.response(400, "Invalid capabilities")
@wrappers_ns.response(403, "Forbidden")
@wrappers_ns.response(404, "Wrapper not found")
@wrappers_ns.param("wrapper_id", "str", "Wrapper identifier")
class WrapperCapabilitiesResource(Resource):
    @require_auth
    def put(self, wrapper_id: str) -> ResponseReturnValue:
        """Replace the capabilities payload for a wrapper."""
        caller_id = getattr(g, "wrapper_id", None)
        if caller_id is not None and caller_id != wrapper_id:
            return {"error": "Bearer token may only update its own capabilities"}, 403

        body = request.get_json(silent=True) or {}
        capabilities = body.get("capabilities")
        if capabilities is None or not isinstance(capabilities, (dict, list)):
            return {"error": "capabilities must be a JSON object or array"}, 400

        svc = _get_service()
        updated = svc.update_capabilities(wrapper_id, cast("dict[str, object]", capabilities))
        if not updated:
            return {"error": _ERR_WRAPPER_NOT_FOUND}, 404

        return {"ok": True}, 200


