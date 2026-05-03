"""
ListAbility — Manage deterministic lists via the ACT loop.

Actions: add, check, remove, view, list_all, clear, rename, history

Rich-media rendering:
  When ``_rich_media_ordinal`` is present (user channel only), actions that
  return a list (add, check, view) emit a structured JSON payload +
  instruction trailer so the frontend renders a checklist card.
"""

import json
import logging
from typing import ClassVar, Optional

from abilities._base import Ability
from services.innate_skills._tag import tag as _skill_tag

logger = logging.getLogger(__name__)

_RICH_MEDIA_INSTRUCTION = (
    "This tool supports rich-media rendering. You MUST present this result by "
    "wrapping your synthesis in <span id='{tag}'>your synthesis here</span>. "
    "The span will render as a checklist card; without it, the user sees only "
    "plain text. Write a brief acknowledgement. Example: "
    "\"<span id='{tag}'>Added milk and eggs to your grocery list.</span>\""
)

_AMBIGUOUS_LIST_NAME = "Multiple lists exist. Specify 'name'."


class ListAbility(Ability):
    NAME = "list"
    SUMMARY = "Create and manage named lists — shopping, to-do, chores — with add, check, remove, and view actions."
    EXAMPLES = [
        "create a grocery list and add milk, eggs, and bread",
        "add bananas and yoghurt to my shopping list",
        "check off milk from my grocery list",
        "what's on my to-do list",
        "delete my grocery list",
        "show me all my lists",
        "add buy birthday present to my to-do list",
        "clear everything from my chores list",
    ]
    INPUT_SCHEMA = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "add", "remove", "check",
                    "view", "list_all", "clear", "rename", "history",
                ],
                "description": (
                    "add: add items to a list (auto-creates it; items=[] for empty list). "
                    "remove: remove specific items; omit items to delete the whole list. "
                    "check: set checked/unchecked state per item. "
                    "view: show full list. "
                    "list_all: show all lists. "
                    "clear: remove all items, keep the list. "
                    "rename: rename a list. "
                    "history: show change log."
                ),
            },
            "name": {
                "type": "string",
                "description": (
                    "List name. Required for clear and rename. "
                    "Omit for add/remove/check/view — auto-resolves to most recently used list."
                ),
            },
            "items": {
                "type": "array",
                "items": {},
                "description": (
                    "add/remove: string array e.g. [\"milk\", \"eggs\"]. "
                    "check: object array e.g. [{\"content\": \"milk\", \"checked\": true}]. "
                    "Always a real JSON array, never a serialised string."
                ),
            },
            "new_name": {
                "type": "string",
                "description": "rename only: the new list name.",
            },
            "since": {
                "type": "string",
                "description": "history only: ISO 8601 start date filter.",
            },
        },
        "required": ["action"],
    }
    TIMEOUT = 10

    _DEFAULT_LIST_NAME: ClassVar[str] = "Shopping List"

    def execute(self, channel: str, params: dict, telemetry: dict | None) -> dict | str:
        action = params.get("action", "list_all")
        ordinal = params.get("_rich_media_ordinal")

        try:
            from services.list_service import ListService
            from services.database_service import get_shared_db_service

            db = get_shared_db_service()
            service = ListService(db)
            body = _dispatch(service, action, params, self._DEFAULT_LIST_NAME)
        except Exception as e:
            logger.error(f"[LIST SKILL] Error: {e}", exc_info=True)
            body = _fail(str(e))

        if ordinal is not None and action in ("add", "check", "view"):
            rich = _try_serialise_rich(body, action, ordinal)
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
        name = payload.get("name") if isinstance(payload, dict) else None
        if not name:
            return payload
        try:
            from services.database_service import get_shared_db_service
            from services.list_service import ListService

            service = ListService(get_shared_db_service())
            fresh = _list_json(service, name)
        except Exception as exc:
            logger.warning("list.enrich_rich_payload: live re-fetch failed for %r: %s", name, exc)
            return payload
        return fresh if fresh else payload


def _dispatch(service, action: str, params: dict, default_list_name: str) -> str:
    if action == "add":
        return _handle_add(service, params, default_list_name)
    elif action == "remove":
        return _handle_remove(service, params, default_list_name)
    elif action == "check":
        return _handle_check(service, params, default_list_name)
    elif action == "view":
        return _handle_view(service, params, default_list_name)
    elif action in ("list_all", "list"):
        return _handle_list_all(service)
    elif action == "clear":
        return _handle_clear(service, params)
    elif action == "rename":
        return _handle_rename(service, params)
    elif action == "history":
        return _handle_history(service, params)
    else:
        valid = "add, remove, check, view, list_all, clear, rename, history"
        return _fail(f"Unknown action '{action}'. Use: {valid}")


def _success(list_data: Optional[dict]) -> str:
    return json.dumps({"status": "success", "list": list_data})


def _fail(message: str) -> str:
    return json.dumps({"status": "fail", "message": message})


def _list_json(service, name: str) -> Optional[dict]:
    lst = service.get_list(name)
    if not lst:
        return None
    return {
        "name": lst["name"],
        "items": [
            {"content": item["content"], "checked": bool(item["checked"])}
            for item in lst.get("items", [])
        ],
    }


def _resolve_name(service, params: dict, default_list_name: str) -> Optional[str]:
    name = params.get("name", "").strip()
    if name:
        return name

    lists = service.get_all_lists()
    if not lists:
        return default_list_name

    if len(lists) == 1:
        return lists[0]["name"]

    most_recent = service.get_most_recent_list()
    if most_recent:
        return most_recent["name"]

    return None


def _normalize_items(params: dict) -> list:
    items = params.get("items", [])
    if isinstance(items, str):
        try:
            parsed = json.loads(items)
            items = parsed if isinstance(parsed, list) else [items]
        except ValueError:
            items = [items]
    return [i for i in items if isinstance(i, str) and i.strip()]


def _handle_add(service, params: dict, default_list_name: str) -> str:
    items = _normalize_items(params)

    name = params.get("name", "").strip()
    if not name:
        lists = service.get_all_lists()
        if not lists:
            name = default_list_name
        elif len(lists) == 1:
            name = lists[0]["name"]
        else:
            most_recent = service.get_most_recent_list()
            name = most_recent["name"] if most_recent else default_list_name

    if not items:
        try:
            service.create_list(name)
        except ValueError:
            pass
        return _success(_list_json(service, name))

    added = service.add_items(name, items, auto_create=True)

    if added == 0 and not service.get_list(name):
        return _fail(f"Failed to add items to '{name}'.")

    return _success(_list_json(service, name))


def _handle_remove(service, params: dict, default_list_name: str) -> str:
    items = _normalize_items(params)

    name = _resolve_name(service, params, default_list_name)
    if not name:
        return _fail(_AMBIGUOUS_LIST_NAME)

    if not items:
        success = service.delete_list(name)
        if not success:
            return _fail(f"List '{name}' not found.")
        return _success(None)

    service.remove_items(name, items)
    lst_data = _list_json(service, name)
    if lst_data is None:
        return _fail(f"List '{name}' not found.")

    return _success(lst_data)


def _split_check_items(raw_items: list) -> tuple[list, list]:
    """Partition raw items into (to_check, to_uncheck) by their `checked` flag."""
    to_check, to_uncheck = [], []
    for item in raw_items:
        if isinstance(item, dict):
            content = item.get("content", "").strip()
            if not content:
                continue
            (to_check if item.get("checked", True) else to_uncheck).append(content)
        elif isinstance(item, str) and item.strip():
            to_check.append(item.strip())
    return to_check, to_uncheck


def _handle_check(service, params: dict, default_list_name: str) -> str:
    raw_items = params.get("items", [])
    if not isinstance(raw_items, list):
        return _fail("'items' must be an array of {content, checked} objects.")

    name = _resolve_name(service, params, default_list_name)
    if not name:
        return _fail(_AMBIGUOUS_LIST_NAME)

    to_check, to_uncheck = _split_check_items(raw_items)
    if not to_check and not to_uncheck:
        return _fail("No valid items provided. Each item must have 'content' and 'checked'.")

    if to_check:
        service.check_items(name, to_check)
    if to_uncheck:
        service.uncheck_items(name, to_uncheck)

    lst_data = _list_json(service, name)
    if lst_data is None:
        return _fail(f"List '{name}' not found.")

    return _success(lst_data)


def _handle_view(service, params: dict, default_list_name: str) -> str:
    name = _resolve_name(service, params, default_list_name)
    if not name:
        return _fail(_AMBIGUOUS_LIST_NAME)

    lst_data = _list_json(service, name)
    if lst_data is None:
        return _fail(f"List '{name}' not found.")

    return _success(lst_data)


def _handle_list_all(service) -> str:
    lists = service.get_all_lists()
    if not lists:
        return "[LIST] No lists found."

    lines = ["[LIST] All lists:"]
    for lst in lists:
        count = lst["item_count"]
        checked = lst["checked_count"]
        count_str = f"{count} items" + (f", {checked} checked" if checked else "")
        lines.append(f"  · {lst['name']} ({count_str})")
    return "\n".join(lines)


def _handle_clear(service, params: dict) -> str:
    name = params.get("name", "").strip()
    if not name:
        return "[LIST] 'name' is required to clear a list."

    count = service.clear_list(name)
    if count == -1:
        return f"[LIST] List '{name}' not found."
    return f"[LIST] Cleared {count} item(s) from '{name}'."


def _handle_rename(service, params: dict) -> str:
    name = params.get("name", "").strip()
    new_name = params.get("new_name", "").strip()
    if not name or not new_name:
        return "[LIST] 'name' and 'new_name' are required to rename a list."

    success = service.rename_list(name, new_name)
    if success:
        return f"[LIST] Renamed '{name}' → '{new_name}'."
    return f"[LIST] Failed to rename '{name}' — list not found or new name already in use."


def _handle_history(service, params: dict) -> str:
    name = params.get("name", "").strip() or None
    since_str = params.get("since", "").strip() or None

    since = None
    if since_str:
        from datetime import datetime, timezone
        from services.time_utils import parse_utc
        _SENTINEL = datetime.min.replace(tzinfo=timezone.utc)
        since = parse_utc(since_str)
        if since == _SENTINEL:
            return "[LIST] Invalid 'since' format. Use ISO 8601 (e.g. '2026-01-01T00:00:00Z')."

    events = service.get_history(name, since=since, limit=30)
    if not events:
        target = f"'{name}'" if name else "any list"
        return f"[LIST] No history found for {target}."

    from services.time_formatter_service import TimeFormatterService
    lines = ["[LIST] History:"]
    for ev in events:
        ts_str = TimeFormatterService.local(ev["created_at"]) or str(ev["created_at"])
        content_part = f" — {ev['item_content']}" if ev["item_content"] else ""
        lines.append(f"  [{ts_str}] {ev['event_type']}{content_part}")

    return "\n".join(lines)


def _try_serialise_rich(body: str, action: str, ordinal: int) -> str | None:
    try:
        parsed = json.loads(body)
    except (ValueError, TypeError):
        return None
    if parsed.get("status") != "success" or parsed.get("list") is None:
        return None
    tag = f"list_{ordinal}"
    payload = parsed["list"]
    data_json = json.dumps(payload)
    instruction = _RICH_MEDIA_INSTRUCTION.format(tag=tag)
    return f"{data_json}\n\n{instruction}"
