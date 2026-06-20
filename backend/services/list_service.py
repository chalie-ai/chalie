"""
List Service — Deterministic, id-addressed list management.

Stores named lists (shopping, to-do, chores, etc.) with per-item state.
Existing lists are addressed by ``id`` (8-char hex); ``name`` is used only
on ``create_list`` and ``rename_list``. Provides perfect, deterministic
recall — unlike probabilistic memory layers (gists, episodes, concepts).
"""

import logging
import secrets
from typing import Optional, List, Dict, cast

from services.database_service import DatabaseService
from services.log_utils import safe
from services.write_queue_service import get_write_queue

logger = logging.getLogger(__name__)


def embed_list(list_id: str, name: str, db: Optional[DatabaseService] = None) -> None:
    """Generate an embedding for the list name and store it in lists_vec. Non-fatal."""
    try:
        from services.embedding_service import EmbeddingService
        from services.embedding_utils import pack_embedding
        if db is None:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()
        embedding = EmbeddingService().generate_embedding(name)
        packed = pack_embedding(embedding)
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT rowid FROM lists WHERE id = ?", (list_id,))
            row = cursor.fetchone()
            if row:
                rowid = row[0]
                cursor.execute(
                    "INSERT OR REPLACE INTO lists_vec(rowid, embedding) VALUES (?, ?)",
                    (rowid, packed)
                )
                conn.commit()
    except Exception as e:
        logging.warning(f"[LISTS] Embedding failed (non-fatal): {e}")


class ListService:
    """Deterministic, id-addressed list management."""

    def __init__(self, db_service: DatabaseService) -> None:
        self.db = db_service
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

        list_id = secrets.token_hex(4)

        try:
            def _insert_list(_id: str = list_id, _name: str = name, _type: str = list_type, _db: DatabaseService = self.db) -> None:
                with _db.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO lists (id, name, list_type, created_at, updated_at)
                        VALUES (?, ?, ?, datetime('now'), datetime('now'))
                    """, (_id, _name, _type))
                    cursor.close()

            self._write_queue.submit_sync(_insert_list)

            embed_list(list_id, name, db=self.db)
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
            def _soft_delete(_id: str = list_id, _db: DatabaseService = self.db) -> int:
                with _db.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE lists SET deleted_at = datetime('now'), updated_at = datetime('now')
                        WHERE id = ? AND deleted_at IS NULL
                    """, (_id,))
                    rc = cursor.rowcount
                    cursor.close()
                    return rc

            updated = cast(int, self._write_queue.submit_sync(_soft_delete)) > 0
            if updated:
                logger.info("[LISTS] Deleted list '%s' (id=%s)", safe(list_row['name']), list_id)
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
            def _clear_items(_id: str = list_id, _db: DatabaseService = self.db) -> int:
                with _db.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE list_items SET removed_at = datetime('now'), updated_at = datetime('now')
                        WHERE list_id = ? AND removed_at IS NULL
                    """, (_id,))
                    rc = cursor.rowcount
                    cursor.close()
                    return rc

            count = cast(int, self._write_queue.submit_sync(_clear_items))
            if count > 0:
                self._touch_list(list_id)
                logger.info(f"[LISTS] Cleared {count} items from list '{list_row['name']}'")
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
        if existing and existing['id'] != list_id:
            logger.warning(f"[LISTS] Cannot rename to '{new_name}' — name already in use")
            return False

        old_name = list_row['name']
        try:
            def _rename(_id: str = list_id, _new_name: str = new_name, _db: DatabaseService = self.db) -> int:
                with _db.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE lists SET name = ?, updated_at = datetime('now')
                        WHERE id = ? AND deleted_at IS NULL
                    """, (_new_name, _id))
                    rc = cursor.rowcount
                    cursor.close()
                    return rc

            updated = cast(int, self._write_queue.submit_sync(_rename)) > 0
            if updated:
                embed_list(list_id, new_name, db=self.db)
                logger.info(f"[LISTS] Renamed list '{old_name}' -> '{new_name}'")
            return updated

        except Exception as e:
            logger.error(f"[LISTS] rename_list failed: {e}")
            return False

    def get_list(self, list_id: str) -> Optional[Dict[str, object]]:
        """Get a list with its active items; None if not found."""
        list_row = self._get_list_row(list_id)
        if not list_row:
            return None

        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, content, checked, position, added_at, updated_at
                    FROM list_items
                    WHERE list_id = ? AND removed_at IS NULL
                    ORDER BY position ASC, added_at ASC
                """, (list_row['id'],))
                rows = cursor.fetchall()
                cursor.close()

            items = [
                {
                    'id': row[0],
                    'content': row[1],
                    'checked': row[2],
                    'position': row[3],
                    'added_at': row[4],
                    'updated_at': row[5],
                }
                for row in rows
            ]

            return {**list_row, 'items': items}

        except Exception as e:
            logger.error(f"[LISTS] get_list failed: {e}")
            return None

    def get_all_lists(self) -> List[Dict[str, object]]:
        """Get all active lists with summary counts (item_count, checked_count)."""
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT
                        l.id,
                        l.name,
                        l.list_type,
                        l.updated_at,
                        SUM(CASE WHEN li.removed_at IS NULL AND li.id IS NOT NULL THEN 1 ELSE 0 END) AS item_count,
                        SUM(CASE WHEN li.removed_at IS NULL AND li.checked THEN 1 ELSE 0 END)        AS checked_count
                    FROM lists l
                    LEFT JOIN list_items li ON li.list_id = l.id
                    WHERE l.deleted_at IS NULL
                    GROUP BY l.id, l.name, l.list_type, l.updated_at
                    ORDER BY l.updated_at DESC
                """)
                rows = cursor.fetchall()
                cursor.close()

            return [
                {
                    'id': row[0],
                    'name': row[1],
                    'list_type': row[2],
                    'updated_at': row[3],
                    'item_count': row[4] or 0,
                    'checked_count': row[5] or 0,
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
        items: List[str],
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
            def _add_items_block(_id: str = list_id, _items: List[str] = items, _dedupe: bool = dedupe, _db: DatabaseService = self.db) -> int:
                with _db.connection() as conn:
                    cursor = conn.cursor()

                    cursor.execute("""
                        SELECT COALESCE(MAX(position), -1)
                        FROM list_items
                        WHERE list_id = ? AND removed_at IS NULL
                    """, (_id,))
                    max_pos = cursor.fetchone()[0]

                    existing_normalized = set()
                    if _dedupe:
                        cursor.execute("""
                            SELECT LOWER(TRIM(content))
                            FROM list_items
                            WHERE list_id = ? AND removed_at IS NULL
                        """, (_id,))
                        existing_normalized = {row[0] for row in cursor.fetchall()}

                    added = 0
                    for item_content in _items:
                        if not item_content or not item_content.strip():
                            continue

                        normalized = item_content.strip().lower()

                        if _dedupe and normalized in existing_normalized:
                            continue

                        cursor.execute("""
                            SELECT id FROM list_items
                            WHERE list_id = ?
                              AND LOWER(TRIM(content)) = ?
                              AND removed_at IS NOT NULL
                            ORDER BY removed_at DESC
                            LIMIT 1
                        """, (_id, normalized))
                        removed_row = cursor.fetchone()

                        if removed_row:
                            max_pos += 1
                            cursor.execute("""
                                UPDATE list_items
                                SET removed_at = NULL, checked = 0,
                                    position = ?, updated_at = datetime('now')
                                WHERE id = ?
                            """, (max_pos, removed_row[0]))
                        else:
                            max_pos += 1
                            item_id = secrets.token_hex(4)
                            cursor.execute("""
                                INSERT INTO list_items (id, list_id, content, position, added_at, updated_at)
                                VALUES (?, ?, ?, ?, datetime('now'), datetime('now'))
                            """, (item_id, _id, item_content.strip(), max_pos))

                        existing_normalized.add(normalized)
                        added += 1

                    cursor.close()
                    return added

            added = cast(int, self._write_queue.submit_sync(_add_items_block))

            if added > 0:
                self._touch_list(list_id)
                logger.info("[LISTS] Added %d items to list '%s'", added, safe(list_row['name']))

            return added

        except Exception as e:
            logger.error(f"[LISTS] add_items failed: {e}")
            return 0

    def remove_items(self, list_id: str, items: List[str]) -> int:
        """Soft-remove items by content (case-insensitive). Returns count removed."""
        list_row = self._get_list_row(list_id)
        if not list_row or not items:
            return 0

        try:
            def _remove_items_block(_id: str = list_id, _items: List[str] = items, _db: DatabaseService = self.db) -> int:
                with _db.connection() as conn:
                    cursor = conn.cursor()
                    removed = 0
                    for item_content in _items:
                        normalized = item_content.strip().lower()
                        cursor.execute("""
                            UPDATE list_items
                            SET removed_at = datetime('now'), updated_at = datetime('now')
                            WHERE list_id = ?
                              AND LOWER(TRIM(content)) = ?
                              AND removed_at IS NULL
                        """, (_id, normalized))
                        removed += cursor.rowcount
                    cursor.close()
                    return removed

            removed = cast(int, self._write_queue.submit_sync(_remove_items_block))
            if removed > 0:
                self._touch_list(list_id)
                logger.info(f"[LISTS] Removed {removed} items from list '{list_row['name']}'")
            return removed

        except Exception as e:
            logger.error(f"[LISTS] remove_items failed: {e}")
            return 0

    def check_items(self, list_id: str, items: List[str]) -> int:
        """Check off items by content (case-insensitive). Returns count checked."""
        return self._set_checked(list_id, items, checked=True)

    def uncheck_items(self, list_id: str, items: List[str]) -> int:
        """Uncheck items by content (case-insensitive). Returns count unchecked."""
        return self._set_checked(list_id, items, checked=False)

    def _set_checked(self, list_id: str, items: List[str], checked: bool) -> int:
        list_row = self._get_list_row(list_id)
        if not list_row or not items:
            return 0

        try:
            def _set_checked_block(_id: str = list_id, _items: List[str] = items, _checked: bool = checked, _db: DatabaseService = self.db) -> int:
                with _db.connection() as conn:
                    cursor = conn.cursor()
                    count = 0
                    for item_content in _items:
                        normalized = item_content.strip().lower()
                        cursor.execute("""
                            UPDATE list_items
                            SET checked = ?, updated_at = datetime('now')
                            WHERE list_id = ?
                              AND LOWER(TRIM(content)) = ?
                              AND removed_at IS NULL
                        """, (1 if _checked else 0, _id, normalized))
                        count += cursor.rowcount
                    cursor.close()
                    return count

            count = cast(int, self._write_queue.submit_sync(_set_checked_block))
            if count > 0:
                self._touch_list(list_id)
            return count

        except Exception as e:
            logger.error(f"[LISTS] _set_checked failed: {e}")
            return 0

    # Internal helpers

    def _get_list_row(self, list_id: str) -> Optional[Dict[str, object]]:
        """Resolve an active list by exact ID. Returns dict or None."""
        if not list_id:
            return None
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, name, list_type, updated_at
                    FROM lists
                    WHERE id = ? AND deleted_at IS NULL
                """, (list_id,))
                row = cursor.fetchone()
                cursor.close()

            if row:
                return {'id': row[0], 'name': row[1], 'list_type': row[2], 'updated_at': row[3]}
            return None

        except Exception as e:
            logger.error(f"[LISTS] _get_list_row failed: {e}")
            return None

    def _find_by_name(self, name: str) -> Optional[Dict[str, object]]:
        """Resolve an active list by case-insensitive name. Used only for create/rename collision checks."""
        if not name:
            return None
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, name, list_type, updated_at
                    FROM lists
                    WHERE LOWER(name) = LOWER(?) AND deleted_at IS NULL
                    LIMIT 1
                """, (name,))
                row = cursor.fetchone()
                cursor.close()

            if row:
                return {'id': row[0], 'name': row[1], 'list_type': row[2], 'updated_at': row[3]}
            return None

        except Exception as e:
            logger.error(f"[LISTS] _find_by_name failed: {e}")
            return None

    def _touch_list(self, list_id: str) -> None:
        def _touch(_id: str = list_id, _db: DatabaseService = self.db) -> None:
            with _db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE lists SET updated_at = datetime('now') WHERE id = ?
                """, (_id,))
                cursor.close()

        try:
            self._write_queue.submit(_touch)
        except Exception as e:
            logger.warning(f"[LISTS] _touch_list failed (non-fatal): {e}")
