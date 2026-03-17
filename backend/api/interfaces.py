"""
Interface management API — pairing, listing, refresh, and removal.

The /pair endpoint is authenticated by pairing key (not cookie or bearer).
All other endpoints require cookie session auth (admin operations).
"""

from flask import Blueprint, request, jsonify
import logging

from api.auth import require_session

logger = logging.getLogger(__name__)

interfaces_bp = Blueprint('interfaces', __name__)


@interfaces_bp.route('/api/interfaces/pairing-key', methods=['POST'])
@require_session
def generate_pairing_key():
    """Generate a one-time pairing key for interface setup.

    Returns the key + Chalie's host:port for the user to enter into the interface.
    """
    from services.interface_registry_service import InterfaceRegistryService
    svc = InterfaceRegistryService()
    result = svc.generate_pairing_key()
    return jsonify(result), 200


@interfaces_bp.route('/api/interfaces/pair', methods=['POST'])
def pair_interface():
    """Complete pairing handshake. Authenticated by pairing key, not cookie.

    Body: {pairing_key, name, host, port}
    Returns: {interface_id, signal_token, status, capabilities}
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body required"}), 400

    pairing_key = data.get("pairing_key")
    name = data.get("name")
    host = data.get("host")
    port = data.get("port")

    if not all([pairing_key, name, host, port]):
        return jsonify({"error": "Missing required fields: pairing_key, name, host, port"}), 400

    try:
        port = int(port)
    except (TypeError, ValueError):
        return jsonify({"error": "Port must be a number"}), 400

    if not (1 <= port <= 65535):
        return jsonify({"error": "Port must be between 1 and 65535"}), 400

    from services.interface_registry_service import InterfaceRegistryService
    svc = InterfaceRegistryService()

    try:
        result = svc.pair(pairing_key, name, host, port)
        return jsonify(result), 201
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"[INTERFACE API] Pairing failed: {e}")
        return jsonify({"error": "Pairing failed"}), 500


@interfaces_bp.route('/api/interfaces', methods=['GET'])
@require_session
def list_interfaces():
    """List all paired interfaces with status and capability count."""
    from services.interface_registry_service import InterfaceRegistryService
    svc = InterfaceRegistryService()
    return jsonify(svc.list_interfaces()), 200


@interfaces_bp.route('/api/interfaces/<interface_id>', methods=['GET'])
@require_session
def get_interface(interface_id):
    """Get interface details including capabilities."""
    from services.interface_registry_service import InterfaceRegistryService
    svc = InterfaceRegistryService()
    iface = svc.get_interface(interface_id)
    if not iface:
        return jsonify({"error": "Interface not found"}), 404

    # Include tool list
    tools = svc.get_interface_tools(interface_id)
    iface["tools"] = tools
    return jsonify(iface), 200


@interfaces_bp.route('/api/interfaces/<interface_id>/refresh', methods=['POST'])
@require_session
def refresh_capabilities(interface_id):
    """Re-fetch capabilities from an interface."""
    from services.interface_registry_service import InterfaceRegistryService
    svc = InterfaceRegistryService()
    iface = svc.get_interface(interface_id)
    if not iface:
        return jsonify({"error": "Interface not found"}), 404

    capabilities = svc.fetch_capabilities(interface_id)
    return jsonify({"capabilities": [c.get("name") for c in capabilities], "count": len(capabilities)}), 200


@interfaces_bp.route('/api/interfaces/<interface_id>', methods=['DELETE'])
@require_session
def remove_interface(interface_id):
    """Unpair and remove an interface."""
    from services.interface_registry_service import InterfaceRegistryService
    svc = InterfaceRegistryService()
    iface = svc.get_interface(interface_id)
    if not iface:
        return jsonify({"error": "Interface not found"}), 404

    svc.remove(interface_id)
    return jsonify({"ok": True}), 200
