"""Lists namespace — clean nested id-addressed CRUD on Pydantic DTOs.

Two resources (lists, items), each pure CRUD, each DTO-typed through the
foundation boundary decorators (``@expects``/``@responds``). This is the
reference namespace: the shape here is the template every other namespace copies.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, cast

from flask.typing import ResponseReturnValue
from flask_restx import Namespace, Resource

from .auth import require_session
from .dto import Error, expects, register_dto, responds
from .dto.boundary import error
from .dto.list import List, ListCreate, ListUpdate
from .dto.list_item import ItemCreate, ItemUpdate, ListItem

if TYPE_CHECKING:
    from services.list_service import ListService


lists_ns = Namespace("lists", description="List operations", path="/api/lists")

register_dto(lists_ns, List, ListCreate, ListUpdate, ListItem, ItemCreate, ItemUpdate, Error)

_L = lists_ns.models

_NOT_FOUND = "Not found"
_NAME_TAKEN = "A list with this name already exists."


def _get_list_service() -> "ListService":
    from services.database_service import get_shared_db_service
    from services.list_service import ListService
    return ListService(get_shared_db_service())


def _list_dto(row: dict[str, object]) -> List:
    return List(
        id=cast(str, row['id']),
        name=cast(str, row['name']),
        list_type=cast(str, row['list_type']),
        created_at=cast(datetime, row['created_at']),
        updated_at=cast(datetime, row['updated_at']),
        item_count=cast(int, row['item_count']),
        checked_count=cast(int, row['checked_count']),
    )


def _item_dto(row: dict[str, object]) -> ListItem:
    return ListItem(
        id=cast(str, row['id']),
        content=cast(str, row['content']),
        checked=cast(bool, row['checked']),
        position=cast(int, row['position']),
        added_at=cast(datetime, row['added_at']),
        updated_at=cast(datetime, row['updated_at']),
    )


# ---------------------------------------------------------------------------
# Lists resource
# ---------------------------------------------------------------------------

@lists_ns.route("")
class ListsResource(Resource):
    @require_session
    @lists_ns.response(200, "All lists", model=_L["List"])
    @responds(List, code=200)
    def get(self) -> list[List]:
        return [_list_dto(row) for row in _get_list_service().get_all_lists()]

    @require_session
    @lists_ns.expect(_L["ListCreate"])
    @lists_ns.response(201, "Created", model=_L["List"])
    @lists_ns.response(409, "A list with this name already exists", model=_L["Error"])
    @lists_ns.response(422, "Validation failed", model=_L["Error"])
    @responds(List, code=201)
    @expects(ListCreate)
    def post(self, dto: ListCreate) -> List | ResponseReturnValue:
        svc = _get_list_service()
        try:
            list_id = svc.create_list(dto.name, list_type=dto.list_type)
        except ValueError:
            return error(_NAME_TAKEN, 409)
        return _list_dto(cast("dict[str, object]", svc.get_list(list_id)))


@lists_ns.route("/<list_id>")
class ListResource(Resource):
    @require_session
    @lists_ns.param("list_id", "List id")
    @lists_ns.response(200, "The list", model=_L["List"])
    @lists_ns.response(404, _NOT_FOUND, model=_L["Error"])
    @responds(List, code=200)
    def get(self, list_id: str) -> List | ResponseReturnValue:
        lst = _get_list_service().get_list(list_id)
        if lst is None:
            return error(_NOT_FOUND, 404)
        return _list_dto(lst)

    @require_session
    @lists_ns.param("list_id", "List id")
    @lists_ns.expect(_L["ListUpdate"])
    @lists_ns.response(200, "Updated list", model=_L["List"])
    @lists_ns.response(404, _NOT_FOUND, model=_L["Error"])
    @lists_ns.response(409, "A list with this name already exists", model=_L["Error"])
    @lists_ns.response(422, "Validation failed", model=_L["Error"])
    @responds(List, code=200)
    @expects(ListUpdate)
    def put(self, list_id: str, dto: ListUpdate) -> List | ResponseReturnValue:
        svc = _get_list_service()
        try:
            lst = svc.update_list(list_id, name=dto.name, list_type=dto.list_type)
        except ValueError:
            return error(_NAME_TAKEN, 409)
        if lst is None:
            return error(_NOT_FOUND, 404)
        return _list_dto(lst)

    @require_session
    @lists_ns.param("list_id", "List id")
    @lists_ns.response(204, "Deleted")
    @lists_ns.response(404, _NOT_FOUND, model=_L["Error"])
    @responds(code=204)
    def delete(self, list_id: str) -> None | ResponseReturnValue:
        if not _get_list_service().delete_list(list_id):
            return error(_NOT_FOUND, 404)
        return None


# ---------------------------------------------------------------------------
# Items sub-resource (addressed by item id)
# ---------------------------------------------------------------------------

@lists_ns.route("/<list_id>/items")
class ListItemsResource(Resource):
    @require_session
    @lists_ns.param("list_id", "List id")
    @lists_ns.response(200, "All items", model=_L["ListItem"])
    @lists_ns.response(404, _NOT_FOUND, model=_L["Error"])
    @responds(ListItem, code=200)
    def get(self, list_id: str) -> list[ListItem] | ResponseReturnValue:
        items = _get_list_service().get_items(list_id)
        if items is None:
            return error(_NOT_FOUND, 404)
        return [_item_dto(it) for it in items]

    @require_session
    @lists_ns.param("list_id", "List id")
    @lists_ns.expect(_L["ItemCreate"])
    @lists_ns.response(201, "Created item", model=_L["ListItem"])
    @lists_ns.response(404, _NOT_FOUND, model=_L["Error"])
    @lists_ns.response(422, "Validation failed", model=_L["Error"])
    @responds(ListItem, code=201)
    @expects(ItemCreate)
    def post(self, list_id: str, dto: ItemCreate) -> ListItem | ResponseReturnValue:
        item = _get_list_service().add_item(list_id, dto.content)
        if item is None:
            return error(_NOT_FOUND, 404)
        return _item_dto(item)


@lists_ns.route("/<list_id>/items/<item_id>")
class ListItemResource(Resource):
    @require_session
    @lists_ns.param("list_id", "List id")
    @lists_ns.param("item_id", "Item id")
    @lists_ns.expect(_L["ItemUpdate"])
    @lists_ns.response(200, "Updated item", model=_L["ListItem"])
    @lists_ns.response(404, _NOT_FOUND, model=_L["Error"])
    @lists_ns.response(422, "Validation failed", model=_L["Error"])
    @responds(ListItem, code=200)
    @expects(ItemUpdate)
    def put(self, list_id: str, item_id: str, dto: ItemUpdate) -> ListItem | ResponseReturnValue:
        item = _get_list_service().update_item(
            list_id, item_id,
            content=dto.content, checked=dto.checked, position=dto.position,
        )
        if item is None:
            return error(_NOT_FOUND, 404)
        return _item_dto(item)

    @require_session
    @lists_ns.param("list_id", "List id")
    @lists_ns.param("item_id", "Item id")
    @lists_ns.response(204, "Deleted")
    @lists_ns.response(404, _NOT_FOUND, model=_L["Error"])
    @responds(code=204)
    def delete(self, list_id: str, item_id: str) -> None | ResponseReturnValue:
        if not _get_list_service().delete_item(list_id, item_id):
            return error(_NOT_FOUND, 404)
        return None
