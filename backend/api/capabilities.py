"""
Capabilities blueprint — /api/capabilities endpoints for managing capability plugins.

Capabilities are self-contained integrations (e.g. CalDAV) that can be connected,
disconnected, and queried through this REST interface.  The blueprint delegates
all business logic to the objects returned by :func:`~capabilities.load_capabilities`
and to :func:`~services.tool_library_service.register_tool` /
:func:`~services.tool_library_service.unregister_tool` for dynamic tool registration.

Authentication
--------------
All routes require an authenticated session or a valid Bearer token (enforced via
the :func:`~api.auth.require_auth` decorator).

Blueprint prefix
----------------
``/api/capabilities``
"""

import logging

from flask import Blueprint, jsonify, request

from .auth import require_auth

logger = logging.getLogger(__name__)

capabilities_bp = Blueprint("capabilities", __name__, url_prefix="/api/capabilities")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_caps() -> dict:
    """Return a fresh dict of all discovered capability instances.

    Delegates to :func:`capabilities.load_capabilities`.  Imported lazily to
    avoid circular imports during application boot.

    Returns:
        dict[str, AbstractCapability]: Capability ``id`` → instance mapping.
    """
    from capabilities import load_capabilities
    return load_capabilities()


def _get_last_sync_at(cap_id: str) -> str | None:
    """Read the ``last_sync_at`` timestamp for a capability from ``tool_configs``.

    Args:
        cap_id: The capability identifier, e.g. ``"caldav"``.

    Returns:
        str | None: ISO 8601 UTC string stored under key
        ``"{cap_id}:last_sync_at"``, or ``None`` if not yet set.
    """
    try:
        from services.database_service import get_shared_db_service
        from services.tool_config_service import ToolConfigService

        db = get_shared_db_service()
        svc = ToolConfigService(db)
        config = svc.get_tool_config(cap_id)
        return config.get(f"{cap_id}:last_sync_at")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[capabilities] Could not read last_sync_at for '%s': %s", cap_id, exc)
        return None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@capabilities_bp.route("", methods=["GET"])
@require_auth
def list_capabilities():
    """List all discovered capabilities with their current connection status.

    Returns a JSON array of objects, one per discovered capability.  Each
    object contains enough information for a UI to render a connection card.

    Returns:
        Response: ``200`` with JSON body::

            {
                "capabilities": [
                    {
                        "id":          str,
                        "name":        str,
                        "version":     str,
                        "connected":   bool,
                        "last_sync_at": str | null,
                        "providers":   list[str]
                    },
                    ...
                ]
            }

        ``500`` on unexpected errors.
    """
    try:
        caps = _load_caps()
        result = []
        for cap_id, cap in caps.items():
            manifest = cap.get_manifest()
            result.append({
                "id": cap_id,
                "name": manifest.get("name", cap_id),
                "version": manifest.get("version", ""),
                "connected": cap.is_connected(),
                "last_sync_at": _get_last_sync_at(cap_id),
                "providers": manifest.get("providers", []),
            })
        return jsonify({"capabilities": result}), 200

    except Exception as exc:
        logger.error("[capabilities] list_capabilities error: %s", exc, exc_info=True)
        return jsonify({"error": "Failed to list capabilities"}), 500


@capabilities_bp.route("/<cap_id>/setup", methods=["POST"])
@require_auth
def setup_capability(cap_id: str):
    """Configure and connect a capability.

    Parses a JSON body with credential fields, calls
    :meth:`~capabilities.base.AbstractCapability.configure` to validate and
    persist credentials, then calls
    :meth:`~capabilities.base.AbstractCapability.connect` to establish the
    connection.  On success, each tool returned by
    :meth:`~capabilities.base.AbstractCapability.get_tools` is registered via
    :func:`~services.tool_library_service.register_tool`.

    Args (URL):
        cap_id: Capability identifier, e.g. ``"caldav"``.

    Request body (JSON):
        ``{"provider": str, "username": str, "password": str}`` — exact fields
        depend on the capability.

    Returns:
        Response:
            - ``200`` ``{"status": "connected"}`` on success.
            - ``400`` ``{"error": "<message>"}`` if credentials are rejected
              (:exc:`ValueError` raised by :meth:`configure`).
            - ``404`` ``{"error": "Capability not found: <cap_id>"}`` for an
              unknown identifier.
            - ``500`` on unexpected errors.
    """
    try:
        caps = _load_caps()
        if cap_id not in caps:
            return jsonify({"error": f"Capability not found: {cap_id}"}), 404

        cap = caps[cap_id]
        credentials = request.get_json(force=True) or {}

        try:
            cap.configure(credentials)
        except ValueError as exc:
            logger.warning("[capabilities] setup '%s' configure failed: %s", cap_id, exc)
            return jsonify({"error": str(exc)}), 400

        connected = cap.connect()
        if not connected:
            return jsonify({"error": "Connection failed — check credentials and try again"}), 400

        # Register dynamic tools exposed by this capability
        from services.tool_library_service import register_tool

        for tool_def in cap.get_tools():
            tool_name = tool_def["name"]
            handler = tool_def["handler"]
            metadata = {k: v for k, v in tool_def.items() if k != "handler"}
            try:
                register_tool(tool_name, handler, metadata)
                logger.info("[capabilities] registered tool '%s' for capability '%s'", tool_name, cap_id)
            except Exception as reg_exc:
                logger.warning(
                    "[capabilities] failed to register tool '%s' for capability '%s': %s",
                    tool_name, cap_id, reg_exc,
                )

        return jsonify({"status": "connected"}), 200

    except ValueError as exc:
        # Catch any ValueError not already handled above (e.g. from connect())
        logger.warning("[capabilities] setup '%s' ValueError: %s", cap_id, exc)
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error("[capabilities] setup_capability('%s') error: %s", cap_id, exc, exc_info=True)
        return jsonify({"error": "Internal error during capability setup"}), 500


@capabilities_bp.route("/<cap_id>/disconnect", methods=["POST"])
@require_auth
def disconnect_capability(cap_id: str):
    """Disconnect a capability and unregister its tools.

    Calls :meth:`~capabilities.base.AbstractCapability.disconnect` then
    removes every tool the capability exposes from the tool registry via
    :func:`~services.tool_library_service.unregister_tool`.

    Args (URL):
        cap_id: Capability identifier, e.g. ``"caldav"``.

    Returns:
        Response:
            - ``200`` ``{"status": "disconnected"}`` on success.
            - ``404`` ``{"error": "Capability not found: <cap_id>"}`` for an
              unknown identifier.
            - ``500`` on unexpected errors.
    """
    try:
        caps = _load_caps()
        if cap_id not in caps:
            return jsonify({"error": f"Capability not found: {cap_id}"}), 404

        cap = caps[cap_id]

        # Unregister tools before disconnecting so we know what tools to remove
        from services.tool_library_service import unregister_tool

        for tool_def in cap.get_tools():
            tool_name = tool_def["name"]
            try:
                unregister_tool(tool_name)
                logger.info("[capabilities] unregistered tool '%s' for capability '%s'", tool_name, cap_id)
            except Exception as unreg_exc:
                logger.warning(
                    "[capabilities] failed to unregister tool '%s' for capability '%s': %s",
                    tool_name, cap_id, unreg_exc,
                )

        cap.disconnect()
        return jsonify({"status": "disconnected"}), 200

    except Exception as exc:
        logger.error("[capabilities] disconnect_capability('%s') error: %s", cap_id, exc, exc_info=True)
        return jsonify({"error": "Internal error during capability disconnect"}), 500


@capabilities_bp.route("/<cap_id>/status", methods=["GET"])
@require_auth
def capability_status(cap_id: str):
    """Return the current status of a specific capability.

    Provides connection state, last synchronisation timestamp, and an error
    count (currently always ``0`` — a future scheduler revision will persist
    per-capability error counts).

    Args (URL):
        cap_id: Capability identifier, e.g. ``"caldav"``.

    Returns:
        Response:
            - ``200`` with JSON body::

                {
                    "id":          str,
                    "connected":   bool,
                    "last_sync_at": str | null,
                    "error_count": int
                }

            - ``404`` ``{"error": "Capability not found: <cap_id>"}`` for an
              unknown identifier.
            - ``500`` on unexpected errors.
    """
    try:
        caps = _load_caps()
        if cap_id not in caps:
            return jsonify({"error": f"Capability not found: {cap_id}"}), 404

        cap = caps[cap_id]

        # Retrieve persisted error count if available (defaults to 0)
        error_count = 0
        try:
            from services.database_service import get_shared_db_service
            from services.tool_config_service import ToolConfigService

            db = get_shared_db_service()
            svc = ToolConfigService(db)
            config = svc.get_tool_config(cap_id)
            raw_count = config.get(f"{cap_id}:error_count")
            if raw_count is not None:
                error_count = int(raw_count)
        except Exception as ec_exc:  # noqa: BLE001
            logger.debug("[capabilities] could not read error_count for '%s': %s", cap_id, ec_exc)

        return jsonify({
            "id": cap_id,
            "connected": cap.is_connected(),
            "last_sync_at": _get_last_sync_at(cap_id),
            "error_count": error_count,
        }), 200

    except Exception as exc:
        logger.error("[capabilities] capability_status('%s') error: %s", cap_id, exc, exc_info=True)
        return jsonify({"error": "Internal error fetching capability status"}), 500
