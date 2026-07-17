"""Watched folders endpoint — migrated from the legacy ``documents`` namespace.

Filesystem directories Chalie monitors for new/changed/deleted files, auto-
ingesting them through the document pipeline. Promoted from the nested
``/api/documents/watched-folders`` to a top-level resource.

Covers the CRUD routes the legacy module exposed for the folder collection:
- GET    /api/watched-folders/all   → get_all (paginated listing)
- POST   /api/watched-folders/<id>  → post (create when id=-1, update otherwise)
- DELETE /api/watched-folders/<id>  → delete

The legacy module had no get-by-id route, so ``get`` is intentionally left
unoverridden (stays a 405 stub). The legacy PUT /api/documents/watched-folders/<id>
is mapped to POST with id != -1. ``/scan`` and ``/browse`` move to Action verbs
under ``api/actions/watched_folders/``.

Deliberate contract changes from the legacy routes: the URL promotion above,
and a missing ``folder_path`` on create now 400s (via :class:`EndpointError`)
instead of the legacy's 422.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import ClassVar, cast

from flask import request
from flask.typing import ResponseReturnValue

from api.endpoint import DocumentedResponse, Endpoint
from exceptions import EndpointError, NotFoundError
from api.request import Request
from api.request.watched_folder import WatchedFolderRequest
from api.response.response import Response
from api.response.watched_folder import WatchedFolder
from services.folder_watcher_service import FolderWatcherService


class WatchedFolders(Endpoint):
    """CRUD endpoint for watched filesystem folders."""

    id_type: ClassVar[type[int] | type[str]] = str
    request_dto: ClassVar[type[Request] | None] = WatchedFolderRequest
    response_dto = {
        "get_all": DocumentedResponse(WatchedFolder, listing=True),
        "post": DocumentedResponse(
            WatchedFolder,
            extras=(
                (403, "Permission denied"),
                (404, "Not found"),
                (409, "This folder is already being watched"),
            ),
        ),
        "delete": DocumentedResponse(),
    }

    def slug(self) -> str:
        return "watched-folders"

    def _service(self) -> FolderWatcherService:
        return FolderWatcherService()

    def get_all(self, page: int = 1, limit: int = 20) -> ResponseReturnValue:
        """List all watched folders, oldest first (by creation time)."""
        folders = [self._to_response(row) for row in self._service().get_all_folders()]
        start = (page - 1) * limit
        page_slice = folders[start : start + limit]
        return WatchedFolder.listing(page_slice, page=page, limit=limit, total=len(folders))

    def post(self, id: int | str, data: Request | None) -> ResponseReturnValue:
        """Register a folder (create sentinel) or apply a partial update.

        Args:
            id: ``-1`` to create, otherwise the folder id to update.
            data: WatchedFolderRequest — create requires folder_path; update
                applies only the non-None fields (folder_path is immutable).

        Returns:
            WatchedFolder, 201 on create / 200 on update.

        Raises:
            EndpointError: folder_path missing on create.
            NotFoundError: folder id not found on update.
        """
        dto = cast(WatchedFolderRequest, data)
        if self.is_create(id):
            return self._create(dto)
        return self._update(cast(str, id), dto)

    def delete(self, id: int | str) -> ResponseReturnValue:
        """Remove a watched folder, optionally soft-deleting its documents.

        Raises:
            NotFoundError: If folder not found.

        Returns:
            204 No Content on success (no body).
        """
        delete_documents = request.args.get("delete_documents", "false").lower() == "true"
        if not self._service().delete_folder(cast(str, id), delete_documents=delete_documents):
            raise NotFoundError("Not found")
        return "", 204

    def _create(self, dto: WatchedFolderRequest) -> ResponseReturnValue:
        if dto.folder_path is None:
            raise EndpointError("folder_path is required")

        try:
            folder = self._service().create_folder(
                folder_path=dto.folder_path,
                label=dto.label,
                file_patterns=dto.file_patterns,
                ignore_patterns=dto.ignore_patterns,
                recursive=dto.recursive if dto.recursive is not None else True,
                scan_interval=dto.scan_interval if dto.scan_interval is not None else 300,
            )
        except ValueError as e:
            return Response.failure(str(e)), 422
        except PermissionError as e:
            return Response.failure(str(e)), 403
        except sqlite3.IntegrityError:
            return Response.failure("This folder is already being watched"), 409
        return self._to_response(folder).single(), 201

    def _update(self, folder_id: str, dto: WatchedFolderRequest) -> ResponseReturnValue:
        svc = self._service()
        if not svc.get_folder(folder_id):
            raise NotFoundError("Not found")

        updates = {
            k: v
            for k, v in {
                "label": dto.label,
                "file_patterns": dto.file_patterns,
                "ignore_patterns": dto.ignore_patterns,
                "recursive": dto.recursive,
                "scan_interval": dto.scan_interval,
                "enabled": dto.enabled,
            }.items()
            if v is not None
        }
        updated = svc.update_folder(folder_id, **updates)
        if updated is None:
            raise NotFoundError("Not found")
        return self._to_response(updated).single()

    def _to_response(self, row: dict[str, object]) -> WatchedFolder:
        return WatchedFolder(
            id=cast(str, row["id"]),
            folder_path=cast(str, row["folder_path"]),
            label=cast("str | None", row.get("label")),
            source_type=cast(str, row["source_type"]),
            enabled=cast(bool, row.get("enabled")),
            file_patterns=cast("list[str]", row.get("file_patterns") or []),
            ignore_patterns=cast("list[str]", row.get("ignore_patterns") or []),
            recursive=cast(bool, row.get("recursive")),
            scan_interval=cast(int, row["scan_interval"]),
            source_config=cast("dict[str, object]", row.get("source_config") or {}),
            created_at=cast(datetime, row["created_at"]),
            updated_at=cast(datetime, row["updated_at"]),
        )
