"""
List Skill — Manage deterministic lists via the ACT loop.

Actions: add, check, remove, view, list_all, clear, rename, history
"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

TOOL_SCHEMA = {
    "name": "list",
    "description": (
        "Create and manage named lists (shopping, to-do, chores). "
        "Prefer over memory for any list-like data. Not for reminders — use schedule for those."
    ),
    "input_schema": {
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
    },
}

_DEFAULT_LIST_NAME = "Shopping List"


def handle_list(channel: str, params: dict) -> str:
    """
    Manage user lists.

    Actions:
    - add:      Add items to a list (auto-creates if needed; items=[] creates empty list)
    - remove:   Remove items from a list; omit items to delete the entire list
    - check:    Set checked state per item [{content, checked}]
    - view:     Show full list contents
    - list_all: Show all active lists summary
    - clear:    Remove all items from a list
    - rename:   Rename a list
    - history:  Show change log

    Args:
        channel: Current conversation channel
        params: Action parameters dict

    Returns:
        JSON string for add/remove/check/view; formatted string for others
    """
    action = params.get('action', 'list_all')

    try:
        from services.list_service import ListService
        from services.database_service import get_shared_db_service

        db = get_shared_db_service()
        service = ListService(db)
        return _dispatch(service, action, params, channel)

    except Exception as e:
        logger.error(f"[LIST SKILL] Error: {e}", exc_info=True)
        return _fail(str(e))


def _dispatch(service, action: str, params: dict, channel: str) -> str:
    if action == 'add':
        return _handle_add(service, params, channel)
    elif action == 'remove':
        return _handle_remove(service, params, channel)
    elif action == 'check':
        return _handle_check(service, params, channel)
    elif action == 'view':
        return _handle_view(service, params, channel)
    elif action in ('list_all', 'list'):
        return _handle_list_all(service, channel)
    elif action == 'clear':
        return _handle_clear(service, params, channel)
    elif action == 'rename':
        return _handle_rename(service, params, channel)
    elif action == 'history':
        return _handle_history(service, params)
    else:
        valid = 'add, remove, check, view, list_all, clear, rename, history'
        return _fail(f"Unknown action '{action}'. Use: {valid}")


# ─────────────────────────────────────────────
# JSON helpers
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# Name resolution
# ─────────────────────────────────────────────

def _resolve_name(service, params: dict) -> Optional[str]:
    """
    Resolve list name from params, falling back to most-recent or default.

    Returns name string, or None if ambiguous (multiple lists, no recent).
    """
    name = params.get('name', '').strip()
    if name:
        return name

    lists = service.get_all_lists()
    if not lists:
        return _DEFAULT_LIST_NAME

    if len(lists) == 1:
        return lists[0]['name']

    most_recent = service.get_most_recent_list()
    if most_recent:
        return most_recent['name']

    return None


def _normalize_items(params: dict) -> list:
    items = params.get('items', [])
    if isinstance(items, str):
        # LLM occasionally serialises the array as a JSON string — parse it back
        try:
            parsed = json.loads(items)
            items = parsed if isinstance(parsed, list) else [items]
        except (json.JSONDecodeError, ValueError):
            items = [items]
    return [i for i in items if isinstance(i, str) and i.strip()]


# ─────────────────────────────────────────────
# Handlers
# ─────────────────────────────────────────────

def _handle_add(service, params: dict, channel: str) -> str:
    items = _normalize_items(params)

    name = params.get('name', '').strip()
    if not name:
        lists = service.get_all_lists()
        if not lists:
            name = _DEFAULT_LIST_NAME
        elif len(lists) == 1:
            name = lists[0]['name']
        else:
            most_recent = service.get_most_recent_list()
            name = most_recent['name'] if most_recent else _DEFAULT_LIST_NAME

    if not items:
        # Create empty list
        try:
            service.create_list(name)
        except ValueError:
            pass  # Already exists — that's fine
        return _success(_list_json(service, name))

    added = service.add_items(name, items, auto_create=True)

    if added == 0 and not service.get_list(name):
        return _fail(f"Failed to add items to '{name}'.")

    return _success(_list_json(service, name))


def _handle_remove(service, params: dict, channel: str) -> str:
    items = _normalize_items(params)

    name = _resolve_name(service, params)
    if not name:
        return _fail("Multiple lists exist. Specify 'name'.")

    if not items:
        # Delete entire list
        success = service.delete_list(name)
        if not success:
            return _fail(f"List '{name}' not found.")
        return _success(None)

    service.remove_items(name, items)
    lst_data = _list_json(service, name)
    if lst_data is None:
        return _fail(f"List '{name}' not found.")

    return _success(lst_data)


def _handle_check(service, params: dict, channel: str) -> str:
    raw_items = params.get('items', [])
    if not isinstance(raw_items, list):
        return _fail("'items' must be an array of {content, checked} objects.")

    name = _resolve_name(service, params)
    if not name:
        return _fail("Multiple lists exist. Specify 'name'.")

    to_check = []
    to_uncheck = []
    for item in raw_items:
        if isinstance(item, dict):
            content = item.get('content', '').strip()
            checked = item.get('checked', True)
            if content:
                (to_check if checked else to_uncheck).append(content)
        elif isinstance(item, str) and item.strip():
            to_check.append(item.strip())

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


def _handle_view(service, params: dict, channel: str) -> str:
    name = _resolve_name(service, params)
    if not name:
        return _fail("Multiple lists exist. Specify 'name'.")

    lst_data = _list_json(service, name)
    if lst_data is None:
        return _fail(f"List '{name}' not found.")

    return _success(lst_data)


def _handle_list_all(service, channel: str) -> str:
    lists = service.get_all_lists()
    if not lists:
        return "[LIST] No lists found."

    lines = ["[LIST] All lists:"]
    for lst in lists:
        count = lst['item_count']
        checked = lst['checked_count']
        count_str = f"{count} items" + (f", {checked} checked" if checked else "")
        lines.append(f"  · {lst['name']} ({count_str})")
    return "\n".join(lines)


def _handle_clear(service, params: dict, channel: str) -> str:
    name = params.get('name', '').strip()
    if not name:
        return "[LIST] 'name' is required to clear a list."

    count = service.clear_list(name)
    if count == -1:
        return f"[LIST] List '{name}' not found."
    return f"[LIST] Cleared {count} item(s) from '{name}'."


def _handle_rename(service, params: dict, channel: str) -> str:
    name = params.get('name', '').strip()
    new_name = params.get('new_name', '').strip()
    if not name or not new_name:
        return "[LIST] 'name' and 'new_name' are required to rename a list."

    success = service.rename_list(name, new_name)
    if success:
        return f"[LIST] Renamed '{name}' → '{new_name}'."
    return f"[LIST] Failed to rename '{name}' — list not found or new name already in use."


def _handle_history(service, params: dict) -> str:
    name = params.get('name', '').strip() or None
    since_str = params.get('since', '').strip() or None

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

    lines = ["[LIST] History:"]
    for ev in events:
        ts = ev['created_at']
        ts_str = ts.strftime('%Y-%m-%d %H:%M') if hasattr(ts, 'strftime') else str(ts)
        content_part = f" — {ev['item_content']}" if ev['item_content'] else ""
        lines.append(f"  [{ts_str}] {ev['event_type']}{content_part}")

    return "\n".join(lines)
