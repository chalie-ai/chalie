"""Snapshot export action — POST /api/snapshot/export.

Streams a complete instance clone as a downloadable ZIP file. The success body
is a raw binary zip stream (not the JSON envelope).

The ``password`` field is the snapshot encryption passphrase — write-only: it
lives only on the inbound request DTO and is never returned or logged.
"""

from __future__ import annotations

import shutil
from typing import ClassVar, cast

from flask import after_this_request, send_file
from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse
from api.request import Request
from api.request.snapshot import SnapshotExportRequest
from services.snapshot_service import SnapshotService


class SnapshotExportAction(Action):
    """Export the entire instance as a downloadable ZIP archive."""

    cookie_only_methods: ClassVar[frozenset[str]] = frozenset({"post"})
    _post_may_create: ClassVar[bool] = False
    request_dto = SnapshotExportRequest
    response_dto = {"post": DocumentedResponse()}

    def slug(self) -> str:
        return "snapshot"

    def verb(self) -> str:
        return "export"

    def post(self, id: int | str, data: Request | None) -> ResponseReturnValue:
        """Export a complete instance clone as a downloadable ZIP.

        Returns a binary ZIP stream (not JSON). The workspace is cleaned up
        after the response is sent.
        """
        dto = cast(SnapshotExportRequest, data)
        zip_path = SnapshotService().export(password=dto.password)
        workspace = zip_path.parent

        @after_this_request
        def _cleanup(response: object) -> object:
            shutil.rmtree(str(workspace), ignore_errors=True)
            return response

        return send_file(
            str(zip_path),
            mimetype="application/zip",
            as_attachment=True,
            download_name=zip_path.name,
        )
