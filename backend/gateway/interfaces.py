"""
Interface management API — approve scopes, update, remove.

Daemons self-register via POST /register (in gateway.py). These routes
handle the user-facing side: listing apps, approving scopes, and removal.
Called by the chat UI JavaScript.
"""

import logging
import os
import shutil

import requests

from flask import Blueprint, request, jsonify

logger = logging.getLogger(__name__)

_ERR_APP_NOT_FOUND = "App not found"

interfaces_bp = Blueprint("dashboard_apps", __name__)

# Set by app.py at startup
_chalie_client = None
_dashboard_port = 3000
_data_dir = ""
_DAEMON_TIMEOUT = 5


def init_interfaces(chalie_client, dashboard_port: int, data_dir: str = "") -> None:
    """Inject dependencies."""
    global _chalie_client, _dashboard_port, _data_dir
    _chalie_client = chalie_client
    _dashboard_port = dashboard_port
    _data_dir = data_dir


# ── List ───────────────────────────────────────────────────────────────────

@interfaces_bp.route("/api/apps", methods=["GET"])
def list_apps():
    """Return all installed interfaces with status."""
    from . import db

    interfaces = db.list_interfaces()
    for iface in interfaces:
        iface["gateway_url"] = f"http://localhost:{_dashboard_port}/gw/{iface['id']}"
    return jsonify(interfaces), 200


# ── Detail ─────────────────────────────────────────────────────────────────

@interfaces_bp.route("/api/apps/<app_id>", methods=["GET"])
def get_app(app_id: str):
    """Return interface details including scopes and capabilities."""
    from . import db

    iface = db.get_interface(app_id)
    if not iface:
        return jsonify({"error": _ERR_APP_NOT_FOUND}), 404

    # Fetch live capabilities from daemon if reachable
    capabilities = []
    try:
        resp = requests.get(
            f"http://{iface['host']}:{iface['port']}/capabilities",
            timeout=_DAEMON_TIMEOUT,
        )
        if resp.ok:
            capabilities = resp.json()
    except Exception:
        pass

    iface["capabilities"] = capabilities
    iface["gateway_url"] = f"http://localhost:{_dashboard_port}/gw/{iface['id']}"
    return jsonify(iface), 200


# ── Approve scopes ─────────────────────────────────────────────────────────

@interfaces_bp.route("/api/apps/<app_id>/scopes", methods=["PUT"])
def approve_scopes(app_id: str):
    """Approve or update scopes for an interface.

    Body: {approved_scopes: {context: {...}, signals: {...}, messages: {...}}}

    Setting approved_scopes activates the interface — gateway starts allowing
    requests through for the approved types.
    """
    from . import db

    iface = db.get_interface(app_id)
    if not iface:
        return jsonify({"error": _ERR_APP_NOT_FOUND}), 404

    body = request.get_json(silent=True) or {}
    approved_scopes = body.get("approved_scopes")
    if approved_scopes is None:
        return jsonify({"error": "approved_scopes is required"}), 400

    # Update scopes and set status to online (user approved)
    db.update_interface(app_id, approved_scopes=approved_scopes, status="online")

    logger.info("[Interfaces] Scopes approved for '%s'", iface["name"])
    return jsonify({"ok": True, "status": "online"}), 200


# ── Refresh capabilities ──────────────────────────────────────────────────

@interfaces_bp.route("/api/apps/<app_id>/refresh", methods=["POST"])
def refresh_app(app_id: str):
    """Re-fetch capabilities from daemon and update Chalie."""
    from . import db

    iface = db.get_interface(app_id)
    if not iface:
        return jsonify({"error": _ERR_APP_NOT_FOUND}), 404

    if iface.get("chalie_interface_id"):
        result = _chalie_client.refresh_capabilities(iface["chalie_interface_id"])
        return jsonify(result), 200

    return jsonify({"error": "Interface not registered with Chalie"}), 400


# ── Remove ─────────────────────────────────────────────────────────────────

@interfaces_bp.route("/api/apps/<app_id>", methods=["DELETE"])
def remove_app(app_id: str):
    """Uninstall an interface."""
    from . import db

    iface = db.get_interface(app_id)
    if not iface:
        return jsonify({"error": _ERR_APP_NOT_FOUND}), 404

    # Unregister from Chalie
    if iface.get("chalie_interface_id"):
        _chalie_client.unregister_interface(iface["chalie_interface_id"])

    # Clean up metadata row (CASCADE should handle it, but be explicit)
    db.delete_meta(app_id)

    # Clean up data folder
    data_path = os.path.join(_data_dir, "interfaces", app_id)
    if os.path.isdir(data_path):
        shutil.rmtree(data_path, ignore_errors=True)

    db.delete_interface(app_id)
    logger.info("[Interfaces] Removed '%s'", iface["name"])
    return jsonify({"ok": True}), 200
