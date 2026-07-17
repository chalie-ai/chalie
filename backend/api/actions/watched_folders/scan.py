"""Watched-folder scan action — nested resource under /api/watched-folders/scan.

Covers:
- POST /api/watched-folders/scan/<id>  → post (request an out-of-schedule scan)

The base auto-documents 404 for get/delete handlers but not for post, so this
class declares it explicitly via ``extras`` to keep swagger truthful.
"""

from __future__ import annotations

from typing import ClassVar, cast

from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse
from exceptions import NotFoundError
from api.request import Request
from services.folder_watcher_service import FolderWatcherService

_NOT_FOUND = "Not found"


class WatchedFoldersScan(Action):
    """Action requesting an out-of-schedule immediate scan for a watched folder."""

    id_type: ClassVar[type[int] | type[str]] = str
    response_dto = {"post": DocumentedResponse(extras=((404, _NOT_FOUND),))}

    def slug(self) -> str:
        return "watched-folders"

    def verb(self) -> str:
        return "scan"

    def post(self, id: int | str, data: Request | None) -> ResponseReturnValue:
        if self.is_create(id):
            raise NotFoundError(_NOT_FOUND)
        svc = FolderWatcherService()
        folder_id = cast(str, id)
        if not svc.get_folder(folder_id):
            raise NotFoundError(_NOT_FOUND)
        svc.trigger_scan(folder_id)
        return "", 204
