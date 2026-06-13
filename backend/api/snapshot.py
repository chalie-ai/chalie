"""Snapshot blueprint — POST /api/snapshot/export, POST /api/snapshot/import.

Exposes the whole-instance Time-Machine over HTTP: export streams a complete
clone of the instance as a single ``.zip``; import stages an uploaded clone and
requests an internal restart so the staged restore is applied at the next boot
(``SnapshotService.apply_pending`` runs before the DB is opened — see
``run.py``). Both routes are session-gated like the other admin blueprints.

Auto-registered by ``api.__init__._register_blueprints`` (top-level
``snapshot_bp`` honouring its own ``url_prefix``) — no ``__init__`` edit needed.
Depends on ``services.snapshot_service.SnapshotService`` (engine) and
``services.app_update_service.AppUpdateService`` (restart).
"""

import logging
import tempfile
from pathlib import Path

from flask import Blueprint, jsonify, request, send_file
from werkzeug.utils import secure_filename

from .auth import require_session

logger = logging.getLogger(__name__)

snapshot_bp = Blueprint("snapshot", __name__, url_prefix="/api/snapshot")

# Per-route ceiling for the import upload. A real snapshot is gigabytes, so the
# global 50 MB MAX_CONTENT_LENGTH (api/__init__.py) would 413 it; setting the
# per-request limit to a large concrete value lifts the cap for THIS route only
# (None would fall through to the global config under Flask 3.1.x).
_SNAPSHOT_MAX_UPLOAD_BYTES = 50 * 1024 ** 3  # 50 GiB — bounds disk-fill, not real snapshots
_DEFAULT_UPLOAD_NAME = "snapshot.zip"


@snapshot_bp.route("/export", methods=["POST"])
@require_session
def snapshot_export():
    """Export the whole instance and stream the resulting zip as a download."""
    try:
        body = request.get_json(silent=True) or {}
        password = body.get("password") or None

        from services.snapshot_service import SnapshotService
        zip_path = SnapshotService().export(password=password)

        return send_file(
            str(zip_path),
            mimetype="application/zip",
            as_attachment=True,
            download_name=zip_path.name,
        )
    except Exception as e:
        logger.exception(f"[REST API] snapshot/export error: {e}")
        return jsonify({"error": "Snapshot export failed"}), 500


@snapshot_bp.route("/import", methods=["POST"])
@require_session
def snapshot_import():
    """Stage an uploaded snapshot and request a restart to apply it.

    Restore is destructive (full wipe-and-replace at next boot); the staging
    step runs the schema-downgrade guard and verifies checksums, raising loudly
    on a bad password, corrupt zip, or unsafe schema before anything is staged.
    """
    # FIRST statement: lift the global 50 MB cap for this route only, before any
    # lazy access to request.files / request.form parses the body (§5.1).
    request.max_content_length = _SNAPSHOT_MAX_UPLOAD_BYTES

    if "file" not in request.files:
        return jsonify({"error": "No snapshot file uploaded"}), 400

    uploaded = request.files["file"]
    password = request.form.get("password") or None

    tmp_dir = Path(tempfile.mkdtemp(prefix="chalie-snapshot-upload-"))
    try:
        safe_name = secure_filename(uploaded.filename or _DEFAULT_UPLOAD_NAME)
        upload_path = tmp_dir / safe_name
        uploaded.save(str(upload_path))

        from services.snapshot_service import SnapshotService
        SnapshotService().stage_import(upload_path, password)
    except Exception as e:
        logger.exception(f"[REST API] snapshot/import staging error: {e}")
        return jsonify({"error": f"Snapshot import failed: {e}"}), 400
    finally:
        import shutil
        shutil.rmtree(str(tmp_dir), ignore_errors=True)

    # Staging succeeded — restart so apply_pending() runs before the DB opens.
    from services.app_update_service import AppUpdateService
    AppUpdateService.request_restart()
    return jsonify({"ok": True, "restarting": True}), 200
