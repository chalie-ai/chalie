"""
ListAbility — Manage deterministic lists via the ACT loop.

Strict CRUD interface. Existing lists are addressed by ``id``; ``name`` is
used only for ``create`` and ``rename``. ``list_all`` returns ids to reuse.

Actions: list_all, create, view, add, check, remove, clear, rename, delete

Rich-media rendering:
  When ``_rich_media_ordinal`` is present (user channel only), actions that
  return a list (create, add, check, view) emit a structured JSON payload +
  instruction trailer so the frontend renders a checklist card.
"""

import json
import logging
from typing import Optional

from abilities._ability import Ability
from services.innate_skills._tag import tag as _skill_tag

logger = logging.getLogger(__name__)

_RICH_MEDIA_INSTRUCTION = (
    "You MUST present this result by wrapping your synthesis in <span id='{tag}'>your synthesis here</span>. "
    "The span will render as a checklist card; without it, the user sees only "
    "plain text. Write a brief acknowledgement. Example: "
    "\"<span id='{tag}'>Added milk and eggs to your grocery list.</span>\""
)


class ListAbility(Ability):
    NAME = "list"
    SEARCH_TOOLTIP = "list and checklist manager"
    SUMMARY = "Create and manage named lists — shopping, to-do, chores — addressed by id."
    EXAMPLES = [
        "create a grocery list and add milk, eggs, and bread",
        "show me all my lists",
        "add bananas to list abc12345",
        "check off milk on list abc12345",
        "rename list abc12345 to weekend chores",
        "delete list abc12345",
    ]
    INPUT_SCHEMA = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "list_all", "create", "view",
                    "add", "check", "remove",
                    "clear", "rename", "delete",
                ],
                "description": (
                    "list_all: return all lists. "
                    "create: make a new list. "
                    "view: read a list. "
                    "add: append items. "
                    "check: toggle checked state per item. "
                    "remove: remove specific items. "
                    "clear: empty a list but keep it. "
                    "rename: change a list's name. "
                    "delete: delete a list entirely."
                ),
            },
            "id": {
                "type": "string",
                "description": (
                    "List id (8-char hex). Required for view/add/check/remove/clear/rename/delete. "
                    "Reuse the id returned by create or list_all."
                ),
            },
            "name": {
                "type": "string",
                "description": "List name. Required for create and rename.",
            },
            "items": {
                "type": "array",
                "items": {},
                "description": (
                    "create/add/remove: string array, e.g. [\"milk\",\"eggs\"]. "
                    "check: object array, e.g. [{\"content\":\"milk\",\"checked\":true}]. "
                    "Always a real JSON array, never a serialised string. "
                    "Optional for create."
                ),
            },
        },
        "required": ["action"],
    }

    def run(self, params: dict) -> dict | str:
        action = params.get("action", "list_all")
        ordinal = params.get("_rich_media_ordinal")

        try:
            from services.list_service import ListService
            from services.database_service import get_shared_db_service

            db = get_shared_db_service()
            service = ListService(db)
            body = _dispatch(service, action, params)
        except Exception as e:
            logger.exception(f"[LIST SKILL] Error: {e}")
            body = _fail(str(e))

        if ordinal is not None and action in ("create", "add", "check", "view"):
            rich = _try_serialise_rich(body, ordinal, action)
            if rich is not None:
                return rich

        return {"text": _skill_tag("list", body, action=action)}

    @classmethod
    def enrich_rich_payload(cls, payload: dict, row: dict) -> dict:
        """Re-fetch the live list so checkbox state mutated via the silent-action
        channel is visible on conversation refresh.

        ``tool_calls.result`` is a frozen snapshot from the LLM-call moment; the
        FE check-action writes directly to the ``lists`` table via
        ``ListService.check_items``. Without this hook, refresh would replay the
        stale snapshot and visually un-tick boxes the user already ticked.
        Single read path: both the live action handler and the refresh path go
        through ``ListService.get_list``.
        """
        list_id = payload.get("id") if isinstance(payload, dict) else None
        if not list_id:
            return payload
        try:
            from services.database_service import get_shared_db_service
            from services.list_service import ListService

            service = ListService(get_shared_db_service())
            fresh = _list_json(service, list_id)
        except Exception as exc:
            logger.warning("list.enrich_rich_payload: live re-fetch failed for %r: %s", list_id, exc)
            return payload
        return fresh if fresh else payload


def _dispatch(service, action: str, params: dict) -> str:
    if action == "list_all":
        return _handle_list_all(service)
    if action == "create":
        return _handle_create(service, params)
    if action == "view":
        return _handle_view(service, params)
    if action == "add":
        return _handle_add(service, params)
    if action == "check":
        return _handle_check(service, params)
    if action == "remove":
        return _handle_remove(service, params)
    if action == "clear":
        return _handle_clear(service, params)
    if action == "rename":
        return _handle_rename(service, params)
    if action == "delete":
        return _handle_delete(service, params)
    valid = "list_all, create, view, add, check, remove, clear, rename, delete"
    return _fail(f"Unknown action '{action}'. Use: {valid}")


def _success(list_data: Optional[dict]) -> str:
    return json.dumps({"status": "success", "list": list_data})


def _success_message(message: str) -> str:
    return json.dumps({"status": "success", "message": message})


def _fail(message: str) -> str:
    return json.dumps({"status": "fail", "message": message})


def _list_json(service, list_id: str) -> Optional[dict]:
    lst = service.get_list(list_id)
    if not lst:
        return None
    return {
        "id": lst["id"],
        "name": lst["name"],
        "items": [
            {"content": item["content"], "checked": bool(item["checked"])}
            for item in lst.get("items", [])
        ],
    }


def _normalize_string_items(params: dict) -> list:
    items = params.get("items", [])
    if isinstance(items, str):
        try:
            parsed = json.loads(items)
            items = parsed if isinstance(parsed, list) else [items]
        except ValueError:
            items = [items]
    return [i.strip() for i in items if isinstance(i, str) and i.strip()]


def _require_id(params: dict) -> tuple[Optional[str], Optional[str]]:
    list_id = (params.get("id") or "").strip()
    if not list_id:
        return None, _fail("'id' is required. Use the id returned by create or list_all.")
    return list_id, None


def _require_name(params: dict) -> tuple[Optional[str], Optional[str]]:
    name = (params.get("name") or "").strip()
    if not name:
        return None, _fail("'name' is required.")
    return name, None


def _require_items(params: dict) -> tuple[Optional[list], Optional[str]]:
    raw = params.get("items")
    if not isinstance(raw, list) or not raw:
        return None, _fail("'items' must be a non-empty array.")
    return raw, None


def _handle_list_all(service) -> str:
    lists = service.get_all_lists()
    payload = [
        {
            "id": lst["id"],
            "name": lst["name"],
            "item_count": lst["item_count"],
            "checked_count": lst["checked_count"],
        }
        for lst in lists
    ]
    return json.dumps({"status": "success", "lists": payload})


def _handle_create(service, params: dict) -> str:
    name, err = _require_name(params)
    if err:
        return err

    items = _normalize_string_items(params)

    try:
        list_id = service.create_list(name)
    except ValueError as e:
        return _fail(str(e))

    if items:
        added = service.add_items(list_id, items)
        if added < len(items):
            logger.info(
                "[LIST SKILL] create: added %d/%d items to list '%s' (id=%s) — some skipped (dedupe or empty)",
                added, len(items), name, list_id,
            )

    return _success(_list_json(service, list_id))


def _handle_view(service, params: dict) -> str:
    list_id, err = _require_id(params)
    if err:
        return err

    lst_data = _list_json(service, list_id)
    if lst_data is None:
        return _fail(f"List with id: {list_id} not found.")

    return _success(lst_data)


def _handle_add(service, params: dict) -> str:
    list_id, err = _require_id(params)
    if err:
        return err

    _, err = _require_items(params)
    if err:
        return err

    if service.get_list(list_id) is None:
        return _fail(f"List with id: {list_id} not found.")

    cleaned = _normalize_string_items(params)
    if not cleaned:
        return _fail("'items' must contain at least one non-empty string.")

    added = service.add_items(list_id, cleaned)
    if added == 0:
        return _fail("No items added (all duplicates or empty).")

    return _success(_list_json(service, list_id))


def _handle_check(service, params: dict) -> str:
    list_id, err = _require_id(params)
    if err:
        return err

    items, err = _require_items(params)
    if err:
        return err

    lst = service.get_list(list_id)
    if lst is None:
        return _fail(f"List with id: {list_id} not found.")

    to_check, to_uncheck = _split_check_items(items)
    if not to_check and not to_uncheck:
        return _fail("Each item must have 'content' (string) and 'checked' (bool).")

    checked_count = service.check_items(list_id, to_check) if to_check else 0
    unchecked_count = service.uncheck_items(list_id, to_uncheck) if to_uncheck else 0

    if checked_count == 0 and unchecked_count == 0:
        return _fail(f"No matching items found in list '{lst['name']}'.")

    return _success(_list_json(service, list_id))


def _split_check_items(raw_items: list) -> tuple[list, list]:
    to_check, to_uncheck = [], []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        content = (item.get("content") or "").strip()
        if not content:
            continue
        if "checked" not in item:
            continue
        (to_check if item["checked"] else to_uncheck).append(content)
    return to_check, to_uncheck


def _handle_remove(service, params: dict) -> str:
    list_id, err = _require_id(params)
    if err:
        return err

    _, err = _require_items(params)
    if err:
        return err

    lst = service.get_list(list_id)
    if lst is None:
        return _fail(f"List with id: {list_id} not found.")

    cleaned = _normalize_string_items(params)
    if not cleaned:
        return _fail("'items' must contain at least one non-empty string.")

    removed = service.remove_items(list_id, cleaned)
    if removed == 0:
        return _fail(f"No matching items found in list '{lst['name']}'.")

    return _success(_list_json(service, list_id))


def _handle_clear(service, params: dict) -> str:
    list_id, err = _require_id(params)
    if err:
        return err

    count = service.clear_list(list_id)
    if count == -1:
        return _fail(f"List with id: {list_id} not found.")

    return _success(_list_json(service, list_id))


def _handle_rename(service, params: dict) -> str:
    list_id, err = _require_id(params)
    if err:
        return err

    new_name, err = _require_name(params)
    if err:
        return err

    success = service.rename_list(list_id, new_name)
    if not success:
        return _fail(f"Rename failed — list with id: {list_id} not found, or name '{new_name}' is already in use.")

    return _success(_list_json(service, list_id))


def _handle_delete(service, params: dict) -> str:
    list_id, err = _require_id(params)
    if err:
        return err

    success = service.delete_list(list_id)
    if not success:
        return _fail(f"List with id: {list_id} not found.")

    return _success_message(f"List with id: {list_id} was deleted successfully")


def _try_serialise_rich(body: str, ordinal: int, action: str) -> str | None:
    try:
        parsed = json.loads(body)
    except (ValueError, TypeError):
        return None
    if parsed.get("status") != "success" or parsed.get("list") is None:
        return None
    payload = parsed["list"]
    tag = f"list_{ordinal}"
    data_json = json.dumps(payload)
    instruction = _RICH_MEDIA_INSTRUCTION.format(tag=tag)
    return _skill_tag("list", f"{data_json}\n\n{instruction}", action=action)
