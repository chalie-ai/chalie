"""
List Service — Deterministic, id-addressed list management.

Stores named lists (shopping, to-do, chores, etc.) with per-item state.
Existing lists are addressed by ``id`` (8-char hex); ``name`` is used only
on ``create_list`` and ``rename_list``. Provides perfect, deterministic
recall — unlike probabilistic memory layers (gists, episodes, concepts).

All ``lists``/``list_items``/``lists_vec`` SQL lives on
:class:`~models.list.List` and :class:`~models.list_item.ListItem`; this
service only orchestrates the write-queue submission and
``Database.transaction()`` grouping around them, and shapes their rows into
the plain dicts the abilities/API boundary expects.
"""

from __future__ import annotations

import logging
from typing import cast

from models.list import List
from models.list_item import ListItem
from services.database import Database
from services.log_utils import safe
from services.write_queue_service import get_write_queue

logger = logging.getLogger(__name__)


def embed_list(list_id: str, name: str) -> None:
    """Generate an embedding for the list name and store it in lists_vec. Non-fatal."""
    try:
        from services.embedding_service import EmbeddingService
        from services.embedding_utils import pack_embedding
        embedding = EmbeddingService().generate_embedding(name)
        packed = pack_embedding(embedding)
        if packed is None:
            return
        with Database.transaction():
            List.set_embedding(list_id, packed)
    except Exception as e:
        logging.warning(f"[LISTS] Embedding failed (non-fatal): {e}")


class ListService:
    """Deterministic, id-addressed list management."""

    def __init__(self) -> None:
        self._write_queue = get_write_queue()

    # List operations

    def create_list(
        self,
        name: str,
        list_type: str = 'checklist',
    ) -> str:
        """Create a new list; returns its id."""
        existing = self._find_by_name(name)
        if existing:
            raise ValueError(f"A list named '{name}' already exists.")

        try:
            def _insert_list(_name: str = name, _type: str = list_type) -> str:
                with Database.transaction():
                    lst = List(name=_name, list_type=_type)
                    lst.save()
                    return cast(str, lst.id)

            list_id = cast(str, self._write_queue.submit_sync(_insert_list))

            embed_list(list_id, name)
            logger.info("[LISTS] Created list '%s' (id=%s)", safe(name), list_id)
            return list_id

        except Exception as e:
            if 'unique' in str(e).lower() or 'duplicate' in str(e).lower():
                raise ValueError(f"A list named '{name}' already exists.") from e
            logger.error(f"[LISTS] Failed to create list: {e}")
            raise

    def delete_list(self, list_id: str) -> bool:
        """Soft-delete a list; returns True on success."""
        list_row = self._get_list_row(list_id)
        if not list_row:
            return False

        try:
            def _soft_delete(_id: str = list_id) -> int:
                with Database.transaction():
                    return List.soft_delete(_id)

            updated = cast(int, self._write_queue.submit_sync(_soft_delete)) > 0
            if updated:
                logger.info("[LISTS] Deleted list '%s' (id=%s)", safe(list_row.name), list_id)
            return True

        except Exception as e:
            logger.error(f"[LISTS] delete_list failed: {e}")
            return False

    def clear_list(self, list_id: str) -> int:
        """Soft-delete all items in a list; returns count, or -1 if not found."""
        list_row = self._get_list_row(list_id)
        if not list_row:
            return -1

        try:
            def _clear_items(_id: str = list_id) -> int:
                with Database.transaction():
                    return ListItem.clear_active(_id)

            count = cast(int, self._write_queue.submit_sync(_clear_items))
            if count > 0:
                self._touch_list(list_id)
                logger.info(f"[LISTS] Cleared {count} items from list '{list_row.name}'")
            return count

        except Exception as e:
            logger.error(f"[LISTS] clear_list failed: {e}")
            return -1

    def rename_list(self, list_id: str, new_name: str) -> bool:
        """Rename a list; returns True on success."""
        list_row = self._get_list_row(list_id)
        if not list_row:
            return False

        existing = self._find_by_name(new_name)
        if existing and existing.id != list_id:
            logger.warning(f"[LISTS] Cannot rename to '{new_name}' — name already in use")
            return False

        old_name = list_row.name
        try:
            def _rename(_id: str = list_id, _new_name: str = new_name) -> int:
                with Database.transaction():
                    return List.update_fields(_id, {"name": _new_name})

            updated = cast(int, self._write_queue.submit_sync(_rename)) > 0
            if updated:
                embed_list(list_id, new_name)
                logger.info(f"[LISTS] Renamed list '{old_name}' -> '{new_name}'")
            return updated

        except Exception as e:
            logger.error(f"[LISTS] rename_list failed: {e}")
            return False

    def update_list(
        self,
        list_id: str,
        *,
        name: str | None = None,
        list_type: str | None = None,
    ) -> dict[str, object] | None:
        """Partial update of a list by id. Returns the updated list (with items),
        ``None`` if the list is missing, or raises ``ValueError`` on name collision."""
        list_row = self._get_list_row(list_id)
        if not list_row:
            return None

        if name is not None and name != list_row.name:
            existing = self._find_by_name(name)
            if existing and existing.id != list_id:
                raise ValueError(f"A list named '{name}' already exists.")

        fields: dict[str, object] = {}
        if name is not None:
            fields["name"] = name
        if list_type is not None:
            fields["list_type"] = list_type
        if not fields:
            return self.get_list(list_id)

        try:
            def _update(_id: str = list_id, _fields: dict[str, object] = fields) -> None:
                with Database.transaction():
                    List.update_fields(_id, _fields)

            self._write_queue.submit_sync(_update)
            if name is not None and name != list_row.name:
                embed_list(list_id, name)
            logger.info("[LISTS] Updated list '%s' (id=%s)", safe(name or list_row.name), list_id)
            return self.get_list(list_id)
        except Exception:
            logger.exception("[LISTS] update_list failed")
            return None

    def get_list(self, list_id: str) -> dict[str, object] | None:
        """Get a list with its active items and derived counts; None if not found."""
        lst = self._get_list_row(list_id)
        if lst is None:
            return None

        try:
            items = self._active_items(cast(str, lst.id))
            return {
                'id': lst.id,
                'name': lst.name,
                'list_type': lst.list_type,
                'created_at': lst.created_at,
                'updated_at': lst.updated_at,
                'items': items,
                'item_count': len(items),
                'checked_count': sum(1 for it in items if it['checked']),
            }
        except Exception as e:
            logger.error(f"[LISTS] get_list failed: {e}")
            return None

    def get_all_lists(self) -> list[dict[str, object]]:
        """Get all active lists with summary counts (item_count, checked_count)."""
        try:
            rows = List.with_counts()
            return [
                {
                    'id': row[0],
                    'name': row[1],
                    'list_type': row[2],
                    'created_at': row[3],
                    'updated_at': row[4],
                    'item_count': row[5] or 0,
                    'checked_count': row[6] or 0,
                }
                for row in rows
            ]

        except Exception as e:
            logger.error(f"[LISTS] get_all_lists failed: {e}")
            return []

    # Item operations (batch)

    def add_items(
        self,
        list_id: str,
        items: list[str],
        dedupe: bool = True,
    ) -> int:
        """Add items to a list; returns count added."""
        list_row = self._get_list_row(list_id)
        if not list_row:
            logger.warning("[LISTS] List '%s' not found", safe(list_id))
            return 0

        if not items:
            return 0

        try:
            def _add_items_block(_id: str = list_id, _items: list[str] = items, _dedupe: bool = dedupe) -> int:
                with Database.transaction():
                    max_pos = ListItem.max_position(_id)
                    existing_normalized: set[str] = (
                        ListItem.active_normalized_contents(_id) if _dedupe else set()
                    )

                    added = 0
                    for item_content in _items:
                        if not item_content or not item_content.strip():
                            continue

                        normalized = item_content.strip().lower()

                        if _dedupe and normalized in existing_normalized:
                            continue

                        removed_id = ListItem.find_removed_by_content(_id, normalized)
                        if removed_id is not None:
                            max_pos += 1
                            ListItem.revive(removed_id, max_pos)
                        else:
                            max_pos += 1
                            ListItem(list_id=_id, content=item_content.strip(), position=max_pos).save()

                        existing_normalized.add(normalized)
                        added += 1

                    return added

            added = cast(int, self._write_queue.submit_sync(_add_items_block))

            if added > 0:
                self._touch_list(list_id)
                logger.info("[LISTS] Added %d items to list '%s'", added, safe(list_row.name))

            return added

        except Exception as e:
            logger.error(f"[LISTS] add_items failed: {e}")
            return 0

    def remove_items(self, list_id: str, items: list[str]) -> int:
        """Soft-remove items by content (case-insensitive). Returns count removed."""
        list_row = self._get_list_row(list_id)
        if not list_row or not items:
            return 0

        try:
            def _remove_items_block(_id: str = list_id, _items: list[str] = items) -> int:
                with Database.transaction():
                    removed = 0
                    for item_content in _items:
                        normalized = item_content.strip().lower()
                        removed += ListItem.remove_by_content(_id, normalized)
                    return removed

            removed = cast(int, self._write_queue.submit_sync(_remove_items_block))
            if removed > 0:
                self._touch_list(list_id)
                logger.info(f"[LISTS] Removed {removed} items from list '{list_row.name}'")
            return removed

        except Exception as e:
            logger.error(f"[LISTS] remove_items failed: {e}")
            return 0

    def check_items(self, list_id: str, items: list[str]) -> int:
        """Check off items by content (case-insensitive). Returns count checked."""
        return self._set_checked(list_id, items, checked=True)

    def uncheck_items(self, list_id: str, items: list[str]) -> int:
        """Uncheck items by content (case-insensitive). Returns count unchecked."""
        return self._set_checked(list_id, items, checked=False)

    def _set_checked(self, list_id: str, items: list[str], checked: bool) -> int:
        list_row = self._get_list_row(list_id)
        if not list_row or not items:
            return 0

        try:
            def _set_checked_block(_id: str = list_id, _items: list[str] = items, _checked: bool = checked) -> int:
                with Database.transaction():
                    count = 0
                    for item_content in _items:
                        normalized = item_content.strip().lower()
                        count += ListItem.set_checked_by_content(_id, normalized, _checked)
                    return count

            count = cast(int, self._write_queue.submit_sync(_set_checked_block))
            if count > 0:
                self._touch_list(list_id)
            return count

        except Exception as e:
            logger.error(f"[LISTS] _set_checked failed: {e}")
            return 0

    # Item operations (id-addressed) — REST CRUD surface

    def get_items(self, list_id: str) -> list[dict[str, object]] | None:
        """Active items for a list ordered by position; None if the list is missing."""
        if self._get_list_row(list_id) is None:
            return None
        try:
            return [{**it, 'checked': bool(it['checked'])} for it in self._active_items(list_id)]
        except Exception:
            logger.exception("[LISTS] get_items failed")
            return None

    def add_item(self, list_id: str, content: str) -> dict[str, object] | None:
        """Insert one item at the end of the list; return the new item row, or None if the list is missing."""
        list_row = self._get_list_row(list_id)
        if not list_row:
            return None
        try:
            def _insert(_id: str = list_id, _content: str = content) -> str:
                with Database.transaction():
                    position = ListItem.max_position(_id) + 1
                    item = ListItem(list_id=_id, content=_content, position=position)
                    item.save()
                    return cast(str, item.id)

            item_id = cast(str, self._write_queue.submit_sync(_insert))
            self._touch_list(list_id)
            logger.info("[LISTS] Added item to list '%s' (id=%s)", safe(list_row.name), item_id)
            return self._get_item_row(list_id, item_id)
        except Exception:
            logger.exception("[LISTS] add_item failed")
            return None

    def update_item(
        self,
        list_id: str,
        item_id: str,
        *,
        content: str | None = None,
        checked: bool | None = None,
        position: int | None = None,
    ) -> dict[str, object] | None:
        """Partial update of an item by id; return the updated row, or None if not found."""
        if self._get_item_row(list_id, item_id) is None:
            return None

        fields: dict[str, object] = {}
        if content is not None:
            fields["content"] = content
        if checked is not None:
            fields["checked"] = 1 if checked else 0
        if position is not None:
            fields["position"] = position
        if not fields:
            return self._get_item_row(list_id, item_id)

        try:
            def _update(
                _item_id: str = item_id,
                _list_id: str = list_id,
                _fields: dict[str, object] = fields,
            ) -> None:
                with Database.transaction():
                    ListItem.update_fields(_item_id, _list_id, _fields)

            self._write_queue.submit_sync(_update)
            self._touch_list(list_id)
            return self._get_item_row(list_id, item_id)
        except Exception:
            logger.exception("[LISTS] update_item failed")
            return None

    def delete_item(self, list_id: str, item_id: str) -> bool:
        """Soft-remove one item by id; True on success, False if not found."""
        if self._get_item_row(list_id, item_id) is None:
            return False
        try:
            def _soft_remove(_item_id: str = item_id, _list_id: str = list_id) -> None:
                with Database.transaction():
                    ListItem.soft_remove(_item_id, _list_id)

            self._write_queue.submit_sync(_soft_remove)
            self._touch_list(list_id)
            logger.info("[LISTS] Removed item %s from list %s", item_id, list_id)
            return True
        except Exception:
            logger.exception("[LISTS] delete_item failed")
            return False

    # Internal helpers

    def _active_items(self, list_id: str) -> list[dict[str, object]]:
        """Active items for a list ordered by position (raw DB column values)."""
        items = ListItem.active(list_id).get()
        return [
            {
                'id': item.id, 'content': item.content, 'checked': item.checked,
                'position': item.position, 'added_at': item.added_at, 'updated_at': item.updated_at,
            }
            for item in items
        ]

    def _get_item_row(self, list_id: str, item_id: str) -> dict[str, object] | None:
        """Fetch one active item by id; None if missing."""
        item = ListItem.by_id_in_list(item_id, list_id)
        if item is None:
            return None
        return {
            'id': item.id, 'content': item.content, 'checked': bool(item.checked),
            'position': item.position, 'added_at': item.added_at, 'updated_at': item.updated_at,
        }

    def _get_list_row(self, list_id: str) -> List | None:
        """Resolve an active list by exact ID. Returns the model instance, or None."""
        if not list_id:
            return None
        try:
            return List.by_id(list_id)
        except Exception as e:
            logger.error(f"[LISTS] _get_list_row failed: {e}")
            return None

    def _find_by_name(self, name: str) -> List | None:
        """Resolve an active list by case-insensitive name. Used only for create/rename collision checks."""
        if not name:
            return None
        try:
            return List.find_by_name(name)
        except Exception as e:
            logger.error(f"[LISTS] _find_by_name failed: {e}")
            return None

    def _touch_list(self, list_id: str) -> None:
        def _touch(_id: str = list_id) -> None:
            with Database.transaction():
                List.touch(_id)

        try:
            self._write_queue.submit(_touch)
        except Exception as e:
            logger.warning(f"[LISTS] _touch_list failed (non-fatal): {e}")
