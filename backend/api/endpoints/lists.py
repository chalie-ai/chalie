"""Lists endpoint — migrated from the legacy lists namespace.

Covers the CRUD routes that the legacy module exposed:
- GET    /api/lists/all        → get_all
- GET    /api/lists/<id>       → get
- POST   /api/lists/<id>       → post (create when id=-1, update otherwise)
- DELETE /api/lists/<id>       → delete

PUT routes from the legacy module are mapped to POST with id != -1.
"""

from __future__ import annotations

from datetime import datetime
from typing import cast

from flask.typing import ResponseReturnValue

from api.endpoint import Endpoint, EndpointError, NotFoundError
from api.request import Request
from api.request.lists import ListRequest
from api.response.lists import ListResponse
from services.list_service import ListService


class Lists(Endpoint):
    """CRUD endpoint for lists."""

    request_dto = ListRequest

    def slug(self) -> str:
        return "lists"

    def get_all(self, page: int = 1, limit: int = 20) -> ResponseReturnValue:
        items = [self._to_response(row) for row in ListService().get_all_lists()]
        total = len(items)
        start = (page - 1) * limit
        return ListResponse.listing(items[start : start + limit], page, limit, total)

    def get(self, id: int | str) -> ResponseReturnValue:
        row = ListService().get_list(cast(str, id))
        if row is None:
            raise NotFoundError("Not found")
        return self._to_response(row).single()

    def post(self, id: int | str, data: Request | None) -> ResponseReturnValue:
        svc = ListService()
        dto = cast(ListRequest, data)
        if self.is_create(id):
            if dto.name is None:
                raise EndpointError("name is required")
            try:
                list_id = svc.create_list(dto.name, list_type=dto.list_type or "checklist")
            except ValueError:
                return ListResponse.failure("A list with this name already exists."), 409
            row = svc.get_list(list_id)
            if row is None:
                raise NotFoundError("Not found")
            return self._to_response(row).single(), 201
        try:
            row = svc.update_list(cast(str, id), name=dto.name, list_type=dto.list_type)
        except ValueError:
            return ListResponse.failure("A list with this name already exists."), 409
        if row is None:
            raise NotFoundError("Not found")
        return self._to_response(row).single()

    def delete(self, id: int | str) -> ResponseReturnValue:
        if not ListService().delete_list(cast(str, id)):
            raise NotFoundError("Not found")
        return "", 204

    def _to_response(self, row: dict[str, object]) -> ListResponse:
        return ListResponse(
            id=cast(str, row["id"]),
            name=cast(str, row["name"]),
            list_type=cast(str, row["list_type"]),
            item_count=cast(int, row["item_count"]),
            checked_count=cast(int, row["checked_count"]),
            created_at=cast(datetime, row["created_at"]),
            updated_at=cast(datetime, row["updated_at"]),
        )
