import logging
from datetime import datetime
from typing import TYPE_CHECKING, cast

from flask import request
from flask.typing import ResponseReturnValue
from flask_restx import Namespace, Resource

from .auth import require_session

if TYPE_CHECKING:
    from services.list_service import ListService

logger = logging.getLogger(__name__)

_ERR_INTERNAL = "Internal server error"
_ERR_NOT_FOUND = "Not found"

lists_bp = Namespace("lists", description="List operations", path="/lists")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_list_service() -> "ListService":
    from services.database_service import get_shared_db_service
    from services.list_service import ListService
    return ListService(get_shared_db_service())


def _serialize_dt(val: object) -> object:
    if isinstance(val, datetime):
        return val.isoformat()
    return val


def _serialize_list(lst: "dict[str, object]") -> "dict[str, object]":
    out = dict(lst)
    for field in ("updated_at", "created_at"):
        if field in out:
            out[field] = _serialize_dt(out[field])
    return out


def _serialize_item(item: "dict[str, object]") -> "dict[str, object]":
    out = dict(item)
    for field in ("added_at", "updated_at"):
        if field in out:
            out[field] = _serialize_dt(out[field])
    return out


def _validate_name(name: "str | None") -> "tuple[str | None, str | None]":
    name = (name or "").strip()
    if not name:
        return None, "name is required"
    if len(name) > 200:
        return None, "name must be 200 characters or fewer"
    return name, None


def _validate_items(items: object) -> "tuple[list[str] | None, str | None]":
    if not isinstance(items, list) or not items:
        return None, "items must be a non-empty array"
    cleaned = []
    for item in items:
        if not isinstance(item, str) or not item.strip():
            return None, "each item must be a non-empty string"
        if len(item.strip()) > 500:
            return None, "each item must be 500 characters or fewer"
        cleaned.append(item.strip())
    return cleaned, None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@lists_bp.route("")
class ListsResource(Resource):
    @require_session
    @lists_bp.response(200, "Success")
    @lists_bp.response(500, "Internal server error")
    def get(self) -> ResponseReturnValue:
        try:
            svc = _get_list_service()
            lists = svc.get_all_lists()
            return {"items": [_serialize_list(lst) for lst in lists]}
        except Exception as e:
            logger.error(f"[LISTS API] get_lists error: {e}")
            return {"error": _ERR_INTERNAL}, 500

    @require_session
    @lists_bp.response(201, "Created")
    @lists_bp.response(400, "Bad request")
    @lists_bp.response(409, "Conflict")
    @lists_bp.response(500, "Internal server error")
    def post(self) -> ResponseReturnValue:
        data = request.get_json(silent=True) or {}
        name, err = _validate_name(data.get("name"))
        if err:
            return {"error": err}, 400

        list_type = (data.get("list_type") or "checklist").strip()

        try:
            svc = _get_list_service()
            list_id = svc.create_list(cast(str, name), list_type=list_type)
            lst = svc.get_list(list_id)
            cast("dict[str, object]", lst)["items"] = [_serialize_item(cast("dict[str, object]", i)) for i in cast("list[object]", cast("dict[str, object]", lst).get("items", []))]
            return {"item": _serialize_list(cast("dict[str, object]", lst))}, 201
        except ValueError as e:
            return {"error": str(e)}, 409
        except Exception as e:
            logger.error(f"[LISTS API] create_list error: {e}")
            return {"error": _ERR_INTERNAL}, 500


@lists_bp.route("/<list_id>")
class ListResource(Resource):
    @require_session
    @lists_bp.param("list_id", "string", "List id")
    @lists_bp.response(200, "Success")
    @lists_bp.response(404, "Not found")
    @lists_bp.response(500, "Internal server error")
    def get(self, list_id: str) -> ResponseReturnValue:
        try:
            svc = _get_list_service()
            lst = svc.get_list(list_id)
            if lst is None:
                return {"error": _ERR_NOT_FOUND}, 404
            lst = dict(lst)
            lst["items"] = [_serialize_item(cast("dict[str, object]", i)) for i in cast("list[object]", lst.get("items", []))]
            return {"item": _serialize_list(lst)}
        except Exception as e:
            logger.error(f"[LISTS API] get_list error: {e}")
            return {"error": _ERR_INTERNAL}, 500

    @require_session
    @lists_bp.param("list_id", "string", "List id")
    @lists_bp.response(200, "Success")
    @lists_bp.response(404, "Not found")
    @lists_bp.response(500, "Internal server error")
    def delete(self, list_id: str) -> ResponseReturnValue:
        try:
            svc = _get_list_service()
            ok = svc.delete_list(list_id)
            if not ok:
                return {"error": _ERR_NOT_FOUND}, 404
            return {"ok": True}
        except Exception as e:
            logger.error(f"[LISTS API] delete_list error: {e}")
            return {"error": _ERR_INTERNAL}, 500


@lists_bp.route("/<list_id>/rename")
class ListRenameResource(Resource):
    @require_session
    @lists_bp.param("list_id", "string", "List id")
    @lists_bp.response(200, "Success")
    @lists_bp.response(400, "Bad request")
    @lists_bp.response(404, "Not found")
    @lists_bp.response(500, "Internal server error")
    def put(self, list_id: str) -> ResponseReturnValue:
        data = request.get_json(silent=True) or {}
        name, err = _validate_name(data.get("name"))
        if err:
            return {"error": err}, 400

        try:
            svc = _get_list_service()
            ok = svc.rename_list(list_id, cast(str, name))
            if not ok:
                return {"error": "Not found or name already in use"}, 404
            return {"ok": True}
        except Exception as e:
            logger.error(f"[LISTS API] rename_list error: {e}")
            return {"error": _ERR_INTERNAL}, 500


@lists_bp.route("/<list_id>/items")
class ListItemsResource(Resource):
    @require_session
    @lists_bp.param("list_id", "string", "List id")
    @lists_bp.response(200, "Success")
    @lists_bp.response(400, "Bad request")
    @lists_bp.response(404, "Not found")
    @lists_bp.response(500, "Internal server error")
    def post(self, list_id: str) -> ResponseReturnValue:
        data = request.get_json(silent=True) or {}
        items, err = _validate_items(data.get("items"))
        if err:
            return {"error": err}, 400

        try:
            svc = _get_list_service()
            if svc.get_list(list_id) is None:
                return {"error": _ERR_NOT_FOUND}, 404
            added = svc.add_items(list_id, cast("list[str]", items))
            return {"added": added}
        except Exception as e:
            logger.error(f"[LISTS API] add_items error: {e}")
            return {"error": _ERR_INTERNAL}, 500

    @require_session
    @lists_bp.param("list_id", "string", "List id")
    @lists_bp.response(200, "Success")
    @lists_bp.response(404, "Not found")
    @lists_bp.response(500, "Internal server error")
    def delete(self, list_id: str) -> ResponseReturnValue:
        try:
            svc = _get_list_service()
            count = svc.clear_list(list_id)
            if count == -1:
                return {"error": _ERR_NOT_FOUND}, 404
            return {"cleared": count}
        except Exception as e:
            logger.error(f"[LISTS API] clear_items error: {e}")
            return {"error": _ERR_INTERNAL}, 500


@lists_bp.route("/<list_id>/items/batch")
class ListItemsBatchResource(Resource):
    @require_session
    @lists_bp.param("list_id", "string", "List id")
    @lists_bp.response(200, "Success")
    @lists_bp.response(400, "Bad request")
    @lists_bp.response(404, "Not found")
    @lists_bp.response(500, "Internal server error")
    def delete(self, list_id: str) -> ResponseReturnValue:
        data = request.get_json(silent=True) or {}
        items, err = _validate_items(data.get("items"))
        if err:
            return {"error": err}, 400

        try:
            svc = _get_list_service()
            if svc.get_list(list_id) is None:
                return {"error": _ERR_NOT_FOUND}, 404
            removed = svc.remove_items(list_id, cast("list[str]", items))
            return {"removed": removed}
        except Exception as e:
            logger.error(f"[LISTS API] remove_items error: {e}")
            return {"error": _ERR_INTERNAL}, 500


@lists_bp.route("/<list_id>/items/check")
class ListItemsCheckResource(Resource):
    @require_session
    @lists_bp.param("list_id", "string", "List id")
    @lists_bp.response(200, "Success")
    @lists_bp.response(400, "Bad request")
    @lists_bp.response(404, "Not found")
    @lists_bp.response(500, "Internal server error")
    def put(self, list_id: str) -> ResponseReturnValue:
        data = request.get_json(silent=True) or {}
        items, err = _validate_items(data.get("items"))
        if err:
            return {"error": err}, 400

        try:
            svc = _get_list_service()
            if svc.get_list(list_id) is None:
                return {"error": _ERR_NOT_FOUND}, 404
            checked = svc.check_items(list_id, cast("list[str]", items))
            return {"checked": checked}
        except Exception as e:
            logger.error(f"[LISTS API] check_items error: {e}")
            return {"error": _ERR_INTERNAL}, 500


@lists_bp.route("/<list_id>/items/uncheck")
class ListItemsUncheckResource(Resource):
    @require_session
    @lists_bp.param("list_id", "string", "List id")
    @lists_bp.response(200, "Success")
    @lists_bp.response(400, "Bad request")
    @lists_bp.response(404, "Not found")
    @lists_bp.response(500, "Internal server error")
    def put(self, list_id: str) -> ResponseReturnValue:
        data = request.get_json(silent=True) or {}
        items, err = _validate_items(data.get("items"))
        if err:
            return {"error": err}, 400

        try:
            svc = _get_list_service()
            if svc.get_list(list_id) is None:
                return {"error": _ERR_NOT_FOUND}, 404
            unchecked = svc.uncheck_items(list_id, cast("list[str]", items))
            return {"unchecked": unchecked}
        except Exception as e:
            logger.error(f"[LISTS API] uncheck_items error: {e}")
            return {"error": _ERR_INTERNAL}, 500