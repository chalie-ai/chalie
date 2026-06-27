"""Snapshot blueprint — POST /api/snapshot/export, POST /api/snapshot/import.

Exposes the whole-instance Time-Machine over HTTP: export streams a complete
clone of the instance as a single ``.zip``; import stages an uploaded clone and
requests an internal restart so the staged restore is applied at the next boot
(``SnapshotService.apply_pending`` runs before the DB is opened — see
``run.py``). Both routes are session-gated like the other admin blueprints.

Auto-registered by ``api.__init__._register_blueprints`` (top-level
``snapshot_bp`` honouring its own ``url_prefix``) — no ``__init__`` edit needed.
Depends on ``services.snapshot_service.SnapshotService`` (engine) and
``services.restart_service.request_restart`` (restart).
"""

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from flask import request, send_file
from flask.typing import ResponseReturnValue
from werkzeug.utils import secure_filename

from flask_restx import Namespace, Resource
from .auth import require_session

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

snapshot_bp = Namespace("snapshot", description="Snapshot export/import", path="/api/snapshot")

# Per-route ceiling for the import upload. A real snapshot is gigabytes, so the
# global 50 MB MAX_CONTENT_LENGTH (api/__init__.py) would 413 it; setting the
# per-request limit to a large concrete value lifts the cap for THIS route only
# (None would fall through to the global config under Flask 3.1.x).
_SNAPSHOT_MAX_UPLOAD_BYTES = 50 * 1024 ** 3  # 50 GiB — bounds disk-fill, not real snapshots
_DEFAULT_UPLOAD_NAME = "snapshot.zip"


@snapshot_bp.route("/export")
@snapshot_bp.response(200, "Snapshot exported")
@snapshot_bp.response(500, "Export failed")
class SnapshotExportResource(Resource):
    @require_session
    def post(self) -> ResponseReturnValue:
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
            return {"error": "Snapshot export failed"}, 500


@snapshot_bp.route("/import")
@snapshot_bp.response(200, "Snapshot imported")
@snapshot_bp.response(400, "Import failed")
class SnapshotImportResource(Resource):
    @require_session
    def post(self) -> ResponseReturnValue:
        """Stage an uploaded snapshot and request a restart to apply it.

        Restore is destructive (full wipe-and-replace at next boot); the staging
        step runs the schema-downgrade guard and verifies checksums, raising loudly
        on a bad password, corrupt zip, or unsafe schema before anything is staged.
        """
        request.max_content_length = _SNAPSHOT_MAX_UPLOAD_BYTES

        if "file" not in request.files:
            return {"error": "No snapshot file uploaded"}, 400

        uploaded = request.files["file"]
        password = request.form.get("password") or None

        from services.snapshot_service import SnapshotError, SnapshotService

        tmp_dir = Path(tempfile.mkdtemp(prefix="chalie-snapshot-upload-"))
        try:
            safe_name = secure_filename(uploaded.filename or _DEFAULT_UPLOAD_NAME)
            upload_path = tmp_dir / safe_name
            uploaded.save(str(upload_path))
            SnapshotService().stage_import(upload_path, password)
        except SnapshotError as e:
            logger.warning(f"[REST API] snapshot/import rejected: {e}")
            return {"error": str(e)}, 400
        except Exception as e:
            logger.exception(f"[REST API] snapshot/import staging error: {e}")
            return {"error": "Snapshot import failed"}, 400
        finally:
            import shutil
            shutil.rmtree(str(tmp_dir), ignore_errors=True)

        from services.restart_service import request_restart
        request_restart()
        return {"ok": True, "restarting": True}, 200
