"""List items action — nested resource under /api/lists/items.

Covers:
- GET    /api/lists/items/<list_id>     → get (list items)
- POST   /api/lists/items/<list_id>     → post (add item)
- POST   /api/lists/items/<list_id>:<item_id>  → update item (composite id)
- DELETE /api/lists/items/<list_id>:<item_id>  → delete item (composite id)

Composite id convention: when the route segment contains a colon it is
interpreted as ``<list_id>:<item_id>`` and split on the FIRST colon only, so
list_id and item_id are free to contain colons themselves. When there is no
colon the segment is treated as a plain list_id (the create/add path).
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar, cast

from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse
from exceptions import EndpointError, NotFoundError
from api.request import Request
from api.request.lists import ItemRequest
from api.response.lists import ListItemResponse
from services.list_service import ListService


class ListItems(Action):
    """Action for list-item operations.

    Add-item uses the plain ``<list_id>`` segment; update and delete use the
    composite ``<list_id>:<item_id>`` form documented in the module docstring.
    """

    request_dto = ItemRequest
    # Unlike most Action verbs, add-item creates a resource (the new list
    # item), so its POST genuinely returns 201.
    _post_may_create: ClassVar[bool] = True
    response_dto = {
        "get": DocumentedResponse(ListItemResponse, listing=True),
        "post": DocumentedResponse(ListItemResponse),
    }

    def slug(self) -> str:
        return "lists"

    def verb(self) -> str:
        return "items"

    def get(self, id: int | str) -> ResponseReturnValue:
        items = ListService().get_items(cast(str, id))
        if items is None:
            raise NotFoundError("Not found")
        response_items = [self._to_response(it) for it in items]
        return ListItemResponse.listing(
            response_items, 1, max(len(response_items), 1), len(response_items)
        )

    def post(self, id: int | str, data: Request | None) -> ResponseReturnValue:
        dto = cast(ItemRequest, data)
        if ":" in str(id):
            return self._update(str(id), dto)
        if dto.content is None:
            raise EndpointError("content is required")
        item = ListService().add_item(cast(str, id), dto.content)
        if item is None:
            raise NotFoundError("Not found")
        return self._to_response(item).single(), 201

    def delete(self, id: int | str) -> ResponseReturnValue:
        id_str = str(id)
        if ":" not in id_str:
            raise EndpointError("Composite id must be <list_id>:<item_id>")
        list_id, item_id = id_str.split(":", 1)
        if not ListService().delete_item(list_id, item_id):
            raise NotFoundError("Not found")
        return "", 204

    def _update(self, composite_id: str, dto: ItemRequest) -> ResponseReturnValue:
        """Handle update via composite id path (POST /lists/items/<list_id>:<item_id>)."""
        list_id, item_id = composite_id.split(":", 1)
        item = ListService().update_item(
            list_id,
            item_id,
            content=dto.content,
            checked=dto.checked,
            position=dto.position,
        )
        if item is None:
            raise NotFoundError("Not found")
        return self._to_response(item).single()

    def _to_response(self, item: dict[str, object]) -> ListItemResponse:
        return ListItemResponse(
            id=cast(str, item["id"]),
            content=cast(str, item["content"]),
            checked=cast(bool, item["checked"]),
            position=cast(int, item["position"]),
            added_at=cast(datetime, item["added_at"]),
            updated_at=cast(datetime, item["updated_at"]),
        )
