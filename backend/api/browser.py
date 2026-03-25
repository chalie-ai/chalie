"""Browser tool REST API — credential vault and snapshot management."""

import logging
from flask import Blueprint, jsonify, request

from .auth import require_session

logger = logging.getLogger(__name__)

browser_bp = Blueprint("browser", __name__, url_prefix="/api/browser")


# -- Credentials ---------------------------------------------------------------

@browser_bp.route("/credentials", methods=["GET"])
@require_session
def list_credentials():
    """List stored browser credentials (no decrypted values)."""
    try:
        from tools.browser.credentials import CredentialVault
        from services.database_service import get_shared_db_service
        vault = CredentialVault(get_shared_db_service())
        creds = vault.list_credentials(account_id=1)
        return jsonify({"credentials": creds})
    except ImportError:
        return jsonify({"error": "Browser tool not available"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@browser_bp.route("/credentials", methods=["POST"])
@require_session
def store_credential():
    """Store a browser credential (encrypted)."""
    try:
        from tools.browser.credentials import CredentialVault
        from services.database_service import get_shared_db_service
    except ImportError:
        return jsonify({"error": "Browser tool not available"}), 503

    data = request.get_json(silent=True) or {}
    domain = (data.get("domain") or "").strip()
    label = (data.get("label") or "").strip()
    cred_type = (data.get("credential_type") or "").strip()
    cred_data = data.get("data")

    if not domain or not label or not cred_type:
        return jsonify({"error": "domain, label, and credential_type required"}), 400
    if cred_type not in ("password", "cookie", "header"):
        return jsonify({"error": "credential_type must be: password, cookie, header"}), 400
    if not cred_data or not isinstance(cred_data, dict):
        return jsonify({"error": "data must be a JSON object"}), 400

    try:
        vault = CredentialVault(get_shared_db_service())
        row_id = vault.store(account_id=1, domain=domain, label=label,
                             credential_type=cred_type, data=cred_data)
        return jsonify({"id": row_id, "status": "stored"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@browser_bp.route("/credentials/<int:credential_id>", methods=["DELETE"])
@require_session
def delete_credential(credential_id):
    """Delete a stored credential."""
    try:
        from tools.browser.credentials import CredentialVault
        from services.database_service import get_shared_db_service
        vault = CredentialVault(get_shared_db_service())
        deleted = vault.delete(account_id=1, credential_id=credential_id)
        if deleted:
            return jsonify({"status": "deleted"})
        return jsonify({"error": "Not found"}), 404
    except ImportError:
        return jsonify({"error": "Browser tool not available"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -- Snapshots -----------------------------------------------------------------

@browser_bp.route("/snapshots", methods=["GET"])
@require_session
def list_snapshots():
    """List active monitoring snapshots."""
    try:
        from tools.browser.credentials import SnapshotStore
        from services.database_service import get_shared_db_service
        store = SnapshotStore(get_shared_db_service())
        snapshots = store.list_snapshots(account_id=1)
        return jsonify({"snapshots": snapshots})
    except ImportError:
        return jsonify({"error": "Browser tool not available"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@browser_bp.route("/snapshots/<snapshot_key>", methods=["DELETE"])
@require_session
def delete_snapshot(snapshot_key):
    """Delete a monitoring snapshot."""
    try:
        from tools.browser.credentials import SnapshotStore
        from services.database_service import get_shared_db_service
        store = SnapshotStore(get_shared_db_service())
        deleted = store.delete(account_id=1, snapshot_key=snapshot_key)
        if deleted:
            return jsonify({"status": "deleted"})
        return jsonify({"error": "Not found"}), 404
    except ImportError:
        return jsonify({"error": "Browser tool not available"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -- Status --------------------------------------------------------------------

@browser_bp.route("/status", methods=["GET"])
@require_session
def browser_status():
    """Return browser pool health status."""
    try:
        from tools.browser.pool import get_pool
        pool = get_pool()
        return jsonify(pool.status())
    except ImportError:
        return jsonify({"available": False, "reason": "playwright not installed"})
    except Exception as e:
        return jsonify({"available": False, "reason": str(e)})
