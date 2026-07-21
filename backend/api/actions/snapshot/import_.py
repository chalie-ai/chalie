"""Snapshot import action — POST /api/snapshot/import.

Stages an uploaded snapshot and requests a restart to apply it. The staged
restore is applied at the next boot (``SnapshotService.apply_pending`` runs
before the DB is opened — see ``run.py``). Restore is destructive (full
wipe-and-replace); the staging step runs the schema-downgrade guard and
verifies checksums, raising loudly on a bad password, corrupt zip, or unsafe
schema before anything is staged.

The ``password`` field is the snapshot encryption passphrase — write-only: it
lives only on the inbound request and is never returned or logged.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import ClassVar

import pyzipper
from flask import request
from flask.typing import ResponseReturnValue
from werkzeug.utils import secure_filename

from api.action import Action
from api.endpoint import DocumentedResponse
from api.request import Request
from api.response.snapshot import SnapshotImportResult
from exceptions import ConflictError, EndpointError, SnapshotError
from services.snapshot_service import SnapshotService

# Per-route ceiling for the import upload. A real snapshot is gigabytes, so the
# global 50 MB MAX_CONTENT_LENGTH (api/__init__.py) would 413 it; setting the
# per-request limit to a large concrete value lifts the cap for THIS route only
# (None would fall through to the global config under Flask 3.1.x).
_SNAPSHOT_MAX_UPLOAD_BYTES = 50 * 1024 ** 3  # 50 GiB — bounds disk-fill, not real snapshots
_DEFAULT_UPLOAD_NAME = "snapshot.zip"


class SnapshotImportAction(Action):
    """Stage an uploaded snapshot and request a restart to apply it."""

    cookie_only_methods: ClassVar[frozenset[str]] = frozenset({"post"})
    _post_may_create: ClassVar[bool] = False
    response_dto = {"post": DocumentedResponse(SnapshotImportResult)}

    def slug(self) -> str:
        return "snapshot"

    def verb(self) -> str:
        return "import"

    def post(self, id: int | str, data: Request | None) -> ResponseReturnValue:
        """Stage an uploaded snapshot and request a restart to apply it.

        Restore is destructive (full wipe-and-replace at next boot); the staging
        step runs the schema-downgrade guard and verifies checksums, raising
        loudly on a bad password, corrupt zip, or unsafe schema before anything
        is staged.
        """
        # Lift the per-request upload cap before werkzeug parses multipart —
        # request.files below is what triggers the form read.
        request.max_content_length = _SNAPSHOT_MAX_UPLOAD_BYTES

        file = request.files.get("file")
        if file is None:
            raise EndpointError("No file provided")
        password = request.form.get("password") or None

        tmp_dir = Path(tempfile.mkdtemp(prefix="chalie-snapshot-upload-"))
        try:
            safe_name = secure_filename(file.filename or _DEFAULT_UPLOAD_NAME)
            upload_path = tmp_dir / safe_name
            file.save(str(upload_path))
            SnapshotService().stage_import(upload_path, password)
        except SnapshotError as e:
            raise ConflictError(str(e)) from e
        except pyzipper.BadZipFile as e:
            # pyzipper's vendored BadZipFile — NOT the stdlib zipfile class, so
            # it must be caught by this name. Raised when the upload is not a
            # zip at all: a client error, surfaced as the 409 the route has
            # always documented for a corrupt archive.
            raise ConflictError(f"Not a valid snapshot archive: {e}") from e
        except RuntimeError as e:
            # pyzipper raises RuntimeError for a wrong AES passphrase — a client
            # error (bad password), not an internal failure. Surfaced as 409 to
            # match the other SnapshotError rejections (the service does not wrap
            # this one as SnapshotError — see stage_import._extract_all).
            raise ConflictError(str(e)) from e
        finally:
            shutil.rmtree(str(tmp_dir), ignore_errors=True)

        from services.restart_service import request_restart
        request_restart()
        return SnapshotImportResult(restarting=True).single(), 200
