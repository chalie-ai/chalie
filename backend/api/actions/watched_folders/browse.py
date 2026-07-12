"""Watched-folder browse action — nested resource under /api/watched-folders/browse.

Covers:
- POST /api/watched-folders/browse  → post (list readable sub-dirs of a host path)

Only the sentinel/id-less form is meaningful — an id-addressed call 404s,
since directory browsing is not addressed by a folder id.
"""

from __future__ import annotations

from typing import cast

from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse
from exceptions import NotFoundError
from api.request import Request
from api.request.watched_folder import WatchedFolderBrowseRequest
from api.response.response import Response
from api.response.watched_folder import WatchedFolderBrowse
from services.folder_watcher_service import FolderWatcherService


class WatchedFoldersBrowse(Action):
    """Action listing readable sub-directories at a host filesystem path."""

    request_dto = WatchedFolderBrowseRequest
    response_dto = {
        "post": DocumentedResponse(
            WatchedFolderBrowse,
            extras=((403, "Permission denied"), (404, "Not found")),
        ),
    }

    def slug(self) -> str:
        return "watched-folders"

    def verb(self) -> str:
        return "browse"

    def post(self, id: int | str, data: Request | None) -> ResponseReturnValue:
        if not self.is_create(id):
            raise NotFoundError("Not found")
        dto = cast(WatchedFolderBrowseRequest, data)
        try:
            result = FolderWatcherService().browse_directory(dto.path)
        except ValueError as e:
            return Response.failure(str(e)), 422
        except PermissionError as e:
            return Response.failure(str(e)), 403
        return WatchedFolderBrowse(
            current=cast(str, result["current"]),
            parent=cast("str | None", result.get("parent")),
            directories=cast("list[str]", result["directories"]),
        ).single()
