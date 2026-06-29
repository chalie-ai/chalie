"""Snapshot namespace — POST /api/snapshot/export, POST /api/snapshot/import.

Exposes the whole-instance Time-Machine over HTTP: export streams a complete
clone of the instance as a single ``.zip``; import stages an uploaded clone and
requests an internal restart so the staged restore is applied at the next boot
(``SnapshotService.apply_pending`` runs before the DB is opened — see ``run.py``).
Both routes are non-CRUD actions (no resource rows) and session-gated.

The ``password`` fields are the snapshot encryption passphrase — write-only: they
live only on the inbound request DTOs and are never returned or logged.
"""

import logging
import shutil
import tempfile
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, ParamSpec, TypeVar

from flask import request, send_file
from flask.typing import ResponseReturnValue
from werkzeug.utils import secure_filename
from flask_restx import Namespace, Resource

from .auth import require_session
from .dto import Error, expects, register_dto, responds
from .dto.snapshot import SnapshotExportRequest, SnapshotImportRequest, SnapshotImportResult

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_ERR_EXPORT = "Snapshot export failed"
_ERR_IMPORT = "Snapshot import failed"

_P = ParamSpec("_P")
_R = TypeVar("_R")

snapshot_ns = Namespace("snapshot", description="Snapshot export/import", path="/api/snapshot")

register_dto(
    snapshot_ns,
    SnapshotExportRequest,
    SnapshotImportRequest,
    SnapshotImportResult,
    Error,
)

_S = snapshot_ns.models

# Per-route ceiling for the import upload. A real snapshot is gigabytes, so the
# global 50 MB MAX_CONTENT_LENGTH (api/__init__.py) would 413 it; setting the
# per-request limit to a large concrete value lifts the cap for THIS route only
# (None would fall through to the global config under Flask 3.1.x).
_SNAPSHOT_MAX_UPLOAD_BYTES = 50 * 1024 ** 3  # 50 GiB — bounds disk-fill, not real snapshots
_DEFAULT_UPLOAD_NAME = "snapshot.zip"


def _error(message: str, status: int) -> ResponseReturnValue:
    """Build a uniform non-2xx ``Error`` body carrying its own status code."""
    return Error(error=message).model_dump(mode="json"), status


def _lift_upload_limit(func: Callable[_P, _R]) -> Callable[_P, _R]:
    """Lift the per-request ``max_content_length`` before the boundary parses multipart.

    Must sit above ``@expects(..., source="form")`` so the cap is raised before
    ``request.files`` is materialized (Flask enforces the limit at parse time,
    which happens inside the boundary's form read — earlier than the handler body).
    """
    @wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        request.max_content_length = _SNAPSHOT_MAX_UPLOAD_BYTES
        return func(*args, **kwargs)
    return wrapper


@snapshot_ns.route("/export")
@snapshot_ns.produces(["application/zip"])
class SnapshotExportResource(Resource):
    @require_session
    @snapshot_ns.expect(_S["SnapshotExportRequest"])
    @snapshot_ns.response(200, "Snapshot ZIP (binary stream)")
    @snapshot_ns.response(500, _ERR_EXPORT, model=_S["Error"])
    @responds(code=200)
    @expects(SnapshotExportRequest)
    def post(self, dto: SnapshotExportRequest) -> ResponseReturnValue:
        """Export a complete instance clone as a downloadable ZIP."""
        try:
            from services.snapshot_service import SnapshotService
            zip_path = SnapshotService().export(password=dto.password)
            # send_file returns a werkzeug Response — the foundation's @responds
            # passes it through untouched (binary passthrough, not a JSON DTO).
            return send_file(
                str(zip_path),
                mimetype="application/zip",
                as_attachment=True,
                download_name=zip_path.name,
            )
        except Exception as e:
            logger.exception(f"[REST API] snapshot/export error: {e}")
            return _error(_ERR_EXPORT, 500)


@snapshot_ns.route("/import")
class SnapshotImportResource(Resource):
    @require_session
    @snapshot_ns.expect(_S["SnapshotImportRequest"])
    @snapshot_ns.response(200, "Snapshot staged, restart requested", model=_S["SnapshotImportResult"])
    @snapshot_ns.response(409, "Bad password / corrupt zip / unsafe schema", model=_S["Error"])
    @snapshot_ns.response(422, "Validation failed", model=_S["Error"])
    @snapshot_ns.response(500, _ERR_IMPORT, model=_S["Error"])
    @responds(SnapshotImportResult, code=200)
    @_lift_upload_limit
    @expects(SnapshotImportRequest, source="form")
    def post(self, dto: SnapshotImportRequest) -> SnapshotImportResult | ResponseReturnValue:
        """Stage an uploaded snapshot and request a restart to apply it.

        Restore is destructive (full wipe-and-replace at next boot); the staging
        step runs the schema-downgrade guard and verifies checksums, raising
        loudly on a bad password, corrupt zip, or unsafe schema before anything
        is staged.
        """
        from services.snapshot_service import SnapshotError, SnapshotService

        tmp_dir = Path(tempfile.mkdtemp(prefix="chalie-snapshot-upload-"))
        try:
            safe_name = secure_filename(dto.file.filename or _DEFAULT_UPLOAD_NAME)
            upload_path = tmp_dir / safe_name
            dto.file.save(str(upload_path))
            SnapshotService().stage_import(upload_path, dto.password)
        except SnapshotError as e:
            logger.warning(f"[REST API] snapshot/import rejected: {e}")
            return _error(str(e), 409)
        except RuntimeError as e:
            # pyzipper raises RuntimeError for a wrong AES passphrase — a client
            # error (bad password), not an internal failure. Surfaced as 409 to
            # match the other SnapshotError rejections (the service does not wrap
            # this one as SnapshotError — see stage_import._extract_all).
            logger.warning(f"[REST API] snapshot/import rejected (bad password): {e}")
            return _error(str(e), 409)
        except Exception as e:
            logger.exception(f"[REST API] snapshot/import staging error: {e}")
            return _error(_ERR_IMPORT, 500)
        finally:
            shutil.rmtree(str(tmp_dir), ignore_errors=True)

        from services.restart_service import request_restart
        request_restart()
        return SnapshotImportResult(restarting=True)