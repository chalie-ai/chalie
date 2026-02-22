"""
List Skill — Manage deterministic lists via the ACT loop.

Actions: create, add, remove, check, uncheck, view, list_all, clear, delete, rename, history
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_LIST_NAME = "Shopping List"


def handle_list(topic: str, params: dict) -> str:
    """
    Manage user lists.

    Actions:
    - create:   Create a new named list
    - add:      Add items to a list (auto-creates if needed)
    - remove:   Remove items from a list
    - check:    Check off items
    - uncheck:  Uncheck items
    - view:     Show full list contents
    - list_all: Show all active lists summary
    - clear:    Remove all items from a list
    - delete:   Soft-delete an entire list
    - rename:   Rename a list
    - history:  Show change log

    Args:
        topic: Current conversation topic
        params: Action parameters dict

    Returns:
        Formatted result string
    """
    action = params.get('action', 'list_all')

    try:
        from services.list_service import ListService
        from services.database_service import DatabaseService, get_merged_db_config

        db_config = get_merged_db_config()
        db = DatabaseService(db_config)
        try:
            service = ListService(db)
            return _dispatch(service, action, params, topic)
        finally:
            db.close_pool()

    except Exception as e:
        logger.error(f"[LIST SKILL] Error: {e}", exc_info=True)
        return f"[LIST] Error: {e}"


def _dispatch(service, action: str, params: dict, topic: str) -> str:
    if action == 'create':
        return _handle_create(service, params, topic)
    elif action == 'add':
        return _handle_add(service, params, topic)
    elif action == 'remove':
        return _handle_remove(service, params, topic)
    elif action == 'check':
        return _handle_check(service, params, topic)
    elif action == 'uncheck':
        return _handle_uncheck(service, params, topic)
    elif action == 'view':
        return _handle_view(service, params, topic)
    elif action == 'list_all':
        return _handle_list_all(service, topic)
    elif action == 'clear':
        return _handle_clear(service, params, topic)
    elif action == 'delete':
        return _handle_delete(service, params, topic)
    elif action == 'rename':
        return _handle_rename(service, params, topic)
    elif action == 'history':
        return _handle_history(service, params)
    else:
        valid = 'create, add, remove, check, uncheck, view, list_all, clear, delete, rename, history'
        return f"[LIST] Unknown action '{action}'. Use: {valid}"


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

    # Use most recently updated
    most_recent = service.get_most_recent_list()
    if most_recent:
        return most_recent['name']

    return None


def _handle_create(service, params: dict, topic: str) -> str:
    name = params.get('name', '').strip()
    if not name:
        return "[LIST] 'name' is required to create a list."

    try:
        list_id = service.create_list(name)
        try:
            from services.list_card_service import ListCardService
            ListCardService().emit_create_card(topic, name)
        except Exception as card_err:
            logger.warning(f"[LIST SKILL] Card emit failed (non-fatal): {card_err}")
        return f"[LIST] Created list '{name}' (id={list_id})."
    except ValueError as e:
        return f"[LIST] {e}"


def _handle_add(service, params: dict, topic: str) -> str:
    items = params.get('items', [])
    if isinstance(items, str):
        items = [items]
    items = [i for i in items if i and i.strip()]

    if not items:
        return "[LIST] 'items' is required to add to a list."

    name = params.get('name', '').strip()

    if not name:
        # Auto-resolve or create default
        lists = service.get_all_lists()
        if not lists:
            name = _DEFAULT_LIST_NAME
        elif len(lists) == 1:
            name = lists[0]['name']
        else:
            most_recent = service.get_most_recent_list()
            name = most_recent['name'] if most_recent else _DEFAULT_LIST_NAME

    added = service.add_items(name, items, auto_create=True)

    if added == 0:
        return f"[LIST] All items already on '{name}' (deduped)."
    skipped = len(items) - added
    if added > 0:
        try:
            from services.list_card_service import ListCardService
            ListCardService().emit_add_card(topic, name, items[:added], skipped)
        except Exception as card_err:
            logger.warning(f"[LIST SKILL] Card emit failed (non-fatal): {card_err}")
    msg = f"[LIST] Added {added} item(s) to '{name}'."
    if skipped > 0:
        msg += f" {skipped} skipped (already present)."
    return msg


def _handle_remove(service, params: dict, topic: str) -> str:
    items = params.get('items', [])
    if isinstance(items, str):
        items = [items]
    items = [i for i in items if i and i.strip()]

    if not items:
        return "[LIST] 'items' is required to remove from a list."

    name = _resolve_name(service, params)
    if not name:
        return "[LIST] Multiple lists exist. Specify 'name'."

    removed = service.remove_items(name, items)
    if removed > 0:
        try:
            from services.list_card_service import ListCardService
            ListCardService().emit_remove_card(topic, name, items[:removed])
        except Exception as card_err:
            logger.warning(f"[LIST SKILL] Card emit failed (non-fatal): {card_err}")
    return f"[LIST] Removed {removed} item(s) from '{name}'."


def _handle_check(service, params: dict, topic: str) -> str:
    items = params.get('items', [])
    if isinstance(items, str):
        items = [items]
    items = [i for i in items if i and i.strip()]

    if not items:
        return "[LIST] 'items' is required to check off."

    name = _resolve_name(service, params)
    if not name:
        return "[LIST] Multiple lists exist. Specify 'name'."

    count = service.check_items(name, items)
    if count > 0:
        try:
            from services.list_card_service import ListCardService
            ListCardService().emit_check_card(topic, name, items[:count], True)
        except Exception as card_err:
            logger.warning(f"[LIST SKILL] Card emit failed (non-fatal): {card_err}")
    return f"[LIST] Checked {count} item(s) on '{name}'."


def _handle_uncheck(service, params: dict, topic: str) -> str:
    items = params.get('items', [])
    if isinstance(items, str):
        items = [items]
    items = [i for i in items if i and i.strip()]

    if not items:
        return "[LIST] 'items' is required to uncheck."

    name = _resolve_name(service, params)
    if not name:
        return "[LIST] Multiple lists exist. Specify 'name'."

    count = service.uncheck_items(name, items)
    if count > 0:
        try:
            from services.list_card_service import ListCardService
            ListCardService().emit_check_card(topic, name, items[:count], False)
        except Exception as card_err:
            logger.warning(f"[LIST SKILL] Card emit failed (non-fatal): {card_err}")
    return f"[LIST] Unchecked {count} item(s) on '{name}'."


def _handle_view(service, params: dict, topic: str) -> str:
    name = params.get('name', '').strip()
    if not name:
        name = _resolve_name(service, params)
    if not name:
        return "[LIST] Multiple lists exist. Specify 'name'."

    lst = service.get_list(name)
    if not lst:
        return f"[LIST] List '{name}' not found."

    items = lst.get('items', [])
    total = len(items)
    checked = sum(1 for i in items if i['checked'])

    try:
        from services.list_card_service import ListCardService
        ListCardService().emit_view_card(topic, lst['name'], items, checked, total)
    except Exception as card_err:
        logger.warning(f"[LIST SKILL] Card emit failed (non-fatal): {card_err}")

    if not items:
        return f"[LIST] '{lst['name']}' is empty."

    lines = [f"[LIST] {lst['name']}:"]
    for item in items:
        status = "✓" if item['checked'] else "·"
        lines.append(f"  {status} {item['content']}")

    lines.append(f"  ({checked}/{total} checked)")
    return "\n".join(lines)


def _handle_list_all(service, topic: str) -> str:
    lists = service.get_all_lists()
    if lists:
        try:
            from services.list_card_service import ListCardService
            ListCardService().emit_list_all_card(topic, lists)
        except Exception as card_err:
            logger.warning(f"[LIST SKILL] Card emit failed (non-fatal): {card_err}")

    if not lists:
        return "[LIST] No lists found."

    lines = ["[LIST] All lists:"]
    for lst in lists:
        count = lst['item_count']
        checked = lst['checked_count']
        count_str = f"{count} items" + (f", {checked} checked" if checked else "")
        lines.append(f"  · {lst['name']} ({count_str})")
    return "\n".join(lines)


def _handle_clear(service, params: dict, topic: str) -> str:
    name = params.get('name', '').strip()
    if not name:
        return "[LIST] 'name' is required to clear a list."

    count = service.clear_list(name)
    if count == -1:
        return f"[LIST] List '{name}' not found."
    if count > 0:
        try:
            from services.list_card_service import ListCardService
            ListCardService().emit_clear_card(topic, name, count)
        except Exception as card_err:
            logger.warning(f"[LIST SKILL] Card emit failed (non-fatal): {card_err}")
    return f"[LIST] Cleared {count} item(s) from '{name}'."


def _handle_delete(service, params: dict, topic: str) -> str:
    name = params.get('name', '').strip()
    if not name:
        return "[LIST] 'name' is required to delete a list."

    success = service.delete_list(name)
    if success:
        try:
            from services.list_card_service import ListCardService
            ListCardService().emit_delete_card(topic, name)
        except Exception as card_err:
            logger.warning(f"[LIST SKILL] Card emit failed (non-fatal): {card_err}")
        return f"[LIST] Deleted list '{name}'."
    return f"[LIST] List '{name}' not found."


def _handle_rename(service, params: dict, topic: str) -> str:
    name = params.get('name', '').strip()
    new_name = params.get('new_name', '').strip()
    if not name or not new_name:
        return "[LIST] 'name' and 'new_name' are required to rename a list."

    success = service.rename_list(name, new_name)
    if success:
        try:
            from services.list_card_service import ListCardService
            ListCardService().emit_rename_card(topic, name, new_name)
        except Exception as card_err:
            logger.warning(f"[LIST SKILL] Card emit failed (non-fatal): {card_err}")
        return f"[LIST] Renamed '{name}' → '{new_name}'."
    return f"[LIST] Failed to rename '{name}' — list not found or new name already in use."


def _handle_history(service, params: dict) -> str:
    name = params.get('name', '').strip() or None
    since_str = params.get('since', '').strip() or None

    since = None
    if since_str:
        try:
            from datetime import datetime, timezone
            since = datetime.fromisoformat(since_str)
            if not since.tzinfo:
                since = since.replace(tzinfo=timezone.utc)
        except ValueError:
            return f"[LIST] Invalid 'since' format. Use ISO 8601 (e.g. '2026-01-01T00:00:00Z')."

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
