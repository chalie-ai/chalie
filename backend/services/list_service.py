"""
List Service - Deterministic list management with full history tracking.

Stores named lists (shopping, to-do, chores, etc.) with per-item state
and an event log for temporal reasoning. Provides perfect, deterministic
recall — unlike probabilistic memory layers (gists, episodes, concepts).
"""

import logging
import secrets
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from services.log_utils import safe
from services.write_queue_service import get_write_queue

logger = logging.getLogger(__name__)


def embed_list(list_id: str, name: str, db=None) -> None:
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
    """Manages deterministic user lists with history tracking."""

    def __init__(self, db_service):
        """
        Initialize list service.

        Args:
            db_service: DatabaseService instance
        """
        self.db = db_service
        self._write_queue = get_write_queue()

    # List operations

    def create_list(
        self,
        name: str,
        list_type: str = 'checklist',
    ) -> str:
        """
        Create a new list.

        Args:
            name: List name (e.g. "Shopping List")
            list_type: List type (default 'checklist')

        Returns:
            list_id (8-char hex string)

        Raises:
            ValueError: If a list with that name already exists
        """
        list_id = secrets.token_hex(4)

        try:
            def _insert_list(_id=list_id, _name=name, _type=list_type, _db=self.db):
                """Insert the new list row on the write-queue thread.

                Args:
                    _id: Pre-generated 8-char hex list ID.
                    _name: List name.
                    _type: List type (e.g. ``'checklist'``).
                    _db: DatabaseService instance.
                """
                with _db.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO lists (id, name, list_type, created_at, updated_at)
                        VALUES (?, ?, ?, datetime('now'), datetime('now'))
                    """, (_id, _name, _type))
                    cursor.close()

            self._write_queue.submit_sync(_insert_list)

            self._log_event(list_id, 'list_created', details={'name': name})
            embed_list(list_id, name, db=self.db)
            logger.info("[LISTS] Created list '%s' (id=%s)", safe(name), list_id)
            return list_id

        except Exception as e:
            if 'unique' in str(e).lower() or 'duplicate' in str(e).lower():
                raise ValueError(f"A list named '{name}' already exists.") from e
            logger.error(f"[LISTS] Failed to create list: {e}")
            raise

    def delete_list(self, name_or_id: str) -> bool:
        """
        Soft-delete a list.

        Args:
            name_or_id: List name or ID

        Returns:
            True if deleted, False if not found
        """
        list_row = self._resolve_list(name_or_id)
        if not list_row:
            return False

        list_id = list_row['id']
        try:
            def _soft_delete(_id=list_id, _db=self.db):
                """Soft-delete the list by setting deleted_at; returns rowcount.

                Args:
                    _id: List ID to soft-delete.
                    _db: DatabaseService instance.

                Returns:
                    Number of rows updated (1 if deleted, 0 if already gone).
                """
                with _db.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE lists SET deleted_at = datetime('now'), updated_at = datetime('now')
                        WHERE id = ? AND deleted_at IS NULL
                    """, (_id,))
                    rc = cursor.rowcount
                    cursor.close()
                    return rc

            updated = self._write_queue.submit_sync(_soft_delete) > 0

            if updated:
                self._log_event(list_id, 'list_deleted', details={'name': list_row['name']})
                logger.info(f"[LISTS] Deleted list '{list_row['name']}' (id={list_id})")
            else:
                # Already deleted — idempotent: goal (resource gone) is already achieved
                logger.debug(f"[LISTS] List '{list_row['name']}' (id={list_id}) already deleted, idempotent no-op")
            return True

        except Exception as e:
            logger.error(f"[LISTS] delete_list failed: {e}")
            return False

    def clear_list(self, name_or_id: str) -> int:
        """
        Soft-delete all items in a list.

        Args:
            name_or_id: List name or ID

        Returns:
            Count of items removed, or -1 if list not found
        """
        list_row = self._resolve_list(name_or_id)
        if not list_row:
            return -1

        list_id = list_row['id']
        try:
            def _clear_items(_id=list_id, _db=self.db):
                """Soft-delete all active items in the list; returns removed count.

                Args:
                    _id: List ID whose items should be cleared.
                    _db: DatabaseService instance.

                Returns:
                    Number of items soft-deleted.
                """
                with _db.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE list_items SET removed_at = datetime('now'), updated_at = datetime('now')
                        WHERE list_id = ? AND removed_at IS NULL
                    """, (_id,))
                    rc = cursor.rowcount
                    cursor.close()
                    return rc

            count = self._write_queue.submit_sync(_clear_items)

            if count > 0:
                self._touch_list(list_id)
                self._log_event(list_id, 'list_cleared', details={'count': count})
                logger.info(f"[LISTS] Cleared {count} items from list '{list_row['name']}'")
            return count

        except Exception as e:
            logger.error(f"[LISTS] clear_list failed: {e}")
            return -1

    def rename_list(self, name_or_id: str, new_name: str) -> bool:
        """
        Rename a list.

        Args:
            name_or_id: List name or ID
            new_name: New name for the list

        Returns:
            True if renamed, False if not found or name collision
        """
        list_row = self._resolve_list(name_or_id)
        if not list_row:
            return False

        # Check for name collision
        existing = self._resolve_list(new_name)
        if existing and existing['id'] != list_row['id']:
            logger.warning(f"[LISTS] Cannot rename to '{new_name}' — name already in use")
            return False

        list_id = list_row['id']
        old_name = list_row['name']
        try:
            def _rename(_id=list_id, _new_name=new_name, _db=self.db):
                """Rename the list; returns rowcount.

                Args:
                    _id: List ID to rename.
                    _new_name: New display name.
                    _db: DatabaseService instance.

                Returns:
                    Number of rows updated (1 on success, 0 if not found/deleted).
                """
                with _db.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE lists SET name = ?, updated_at = datetime('now')
                        WHERE id = ? AND deleted_at IS NULL
                    """, (_new_name, _id))
                    rc = cursor.rowcount
                    cursor.close()
                    return rc

            updated = self._write_queue.submit_sync(_rename) > 0

            if updated:
                self._log_event(list_id, 'list_renamed', details={'old_name': old_name, 'new_name': new_name})
                embed_list(list_id, new_name, db=self.db)
                logger.info(f"[LISTS] Renamed list '{old_name}' -> '{new_name}'")
            return updated

        except Exception as e:
            logger.error(f"[LISTS] rename_list failed: {e}")
            return False

    def get_list(self, name_or_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a list with its active items.

        Args:
            name_or_id: List name or ID

        Returns:
            Dict with list data and items array, or None if not found
        """
        list_row = self._resolve_list(name_or_id)
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

    def get_all_lists(self) -> List[Dict[str, Any]]:
        """
        Get all active lists with summary counts.

        Returns:
            List of summary dicts (name, item_count, checked_count, updated_at)
        """
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
        name_or_id: str,
        items: List[str],
        dedupe: bool = True,
        auto_create: bool = True,
    ) -> int:
        """
        Add items to a list.

        Args:
            name_or_id: List name or ID
            items: List of item content strings
            dedupe: Skip items already on the list (case-insensitive, default True)
            auto_create: Create the list if it doesn't exist (default True)

        Returns:
            Count of items actually added
        """
        list_row = self._resolve_list(name_or_id)

        if not list_row:
            if not auto_create:
                logger.warning("[LISTS] List '%s' not found", safe(name_or_id))
                return 0
            list_id = self.create_list(name_or_id)
            list_row = {'id': list_id, 'name': name_or_id}

        list_id = list_row['id']
        if not items:
            return 0

        try:
            def _add_items_block(_id=list_id, _items=items, _dedupe=dedupe, _db=self.db):
                """Read position/dedup state then insert or restore items atomically.

                Runs on the write-queue thread so all reads and writes share one
                serialised SQLite connection, preventing position conflicts.

                Args:
                    _id: List ID to add items to.
                    _items: Ordered list of item content strings.
                    _dedupe: When ``True``, skip items already present (case-insensitive).
                    _db: DatabaseService instance.

                Returns:
                    Count of items actually added.
                """
                with _db.connection() as conn:
                    cursor = conn.cursor()

                    # Get current max position
                    cursor.execute("""
                        SELECT COALESCE(MAX(position), -1)
                        FROM list_items
                        WHERE list_id = ? AND removed_at IS NULL
                    """, (_id,))
                    max_pos = cursor.fetchone()[0]

                    # Get existing active items for dedup
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

                        # Dedupe check
                        if _dedupe and normalized in existing_normalized:
                            continue

                        # Check if this item was previously removed (restore instead of insert)
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
                            # Restore the soft-deleted row
                            max_pos += 1
                            cursor.execute("""
                                UPDATE list_items
                                SET removed_at = NULL, checked = 0,
                                    position = ?, updated_at = datetime('now')
                                WHERE id = ?
                            """, (max_pos, removed_row[0]))
                        else:
                            # Insert new row
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

            added = self._write_queue.submit_sync(_add_items_block)

            if added > 0:
                self._touch_list(list_id)
                for item_content in items[:added]:
                    self._log_event(
                        list_id, 'item_added',
                        item_content=item_content.strip(),
                        details={'normalized_content': item_content.strip().lower()},
                    )
                logger.info("[LISTS] Added %d items to list '%s'", added, safe(list_row['name']))

            return added

        except Exception as e:
            logger.error(f"[LISTS] add_items failed: {e}")
            return 0

    def remove_items(self, name_or_id: str, items: List[str]) -> int:
        """
        Soft-remove items from a list (case-insensitive match).

        Args:
            name_or_id: List name or ID
            items: List of item content strings to remove

        Returns:
            Count of items removed
        """
        list_row = self._resolve_list(name_or_id)
        if not list_row or not items:
            return 0

        list_id = list_row['id']
        removed = 0

        try:
            def _remove_items_block(
                _id=list_id, _items=items, _row=list_row,
                _log=self._log_event, _db=self.db,
            ):
                """Soft-remove matching items inside a single serialised connection.

                Args:
                    _id: List ID.
                    _items: Item content strings to remove (case-insensitive).
                    _row: Resolved list dict (used for logging).
                    _log: Bound reference to ``self._log_event`` for per-item events.
                    _db: DatabaseService instance.

                Returns:
                    Total number of rows soft-deleted.
                """
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
                        if cursor.rowcount > 0:
                            removed += cursor.rowcount
                            _log(
                                _id, 'item_removed',
                                item_content=item_content.strip(),
                                details={'normalized_content': normalized},
                            )
                    cursor.close()
                    return removed

            removed = self._write_queue.submit_sync(_remove_items_block)

            if removed > 0:
                self._touch_list(list_id)
                logger.info(f"[LISTS] Removed {removed} items from list '{list_row['name']}'")
            return removed

        except Exception as e:
            logger.error(f"[LISTS] remove_items failed: {e}")
            return 0

    def check_items(self, name_or_id: str, items: List[str]) -> int:
        """
        Check off items in a list (case-insensitive match).

        Args:
            name_or_id: List name or ID
            items: List of item content strings to check

        Returns:
            Count of items checked
        """
        return self._set_checked(name_or_id, items, checked=True)

    def uncheck_items(self, name_or_id: str, items: List[str]) -> int:
        """
        Uncheck items in a list (case-insensitive match).

        Args:
            name_or_id: List name or ID
            items: List of item content strings to uncheck

        Returns:
            Count of items unchecked
        """
        return self._set_checked(name_or_id, items, checked=False)

    def _set_checked(self, name_or_id: str, items: List[str], checked: bool) -> int:
        """Set the checked state of matching items in a serialised write-queue operation.

        Args:
            name_or_id: List name or 8-char hex ID.
            items: Item content strings to update (case-insensitive).
            checked: ``True`` to check, ``False`` to uncheck.

        Returns:
            Count of rows updated, or 0 on error or empty input.
        """
        list_row = self._resolve_list(name_or_id)
        if not list_row or not items:
            return 0

        list_id = list_row['id']
        event_type = 'item_checked' if checked else 'item_unchecked'

        try:
            def _set_checked_block(
                _id=list_id, _items=items, _checked=checked,
                _evt=event_type, _log=self._log_event, _db=self.db,
            ):
                """Toggle checked state inside a single serialised connection.

                Args:
                    _id: List ID.
                    _items: Item content strings to toggle.
                    _checked: Target checked state.
                    _evt: Event type string for the audit log.
                    _log: Bound reference to ``self._log_event``.
                    _db: DatabaseService instance.

                Returns:
                    Total number of rows updated.
                """
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
                        if cursor.rowcount > 0:
                            count += cursor.rowcount
                            _log(
                                _id, _evt,
                                item_content=item_content.strip(),
                                details={'normalized_content': normalized},
                            )
                    cursor.close()
                    return count

            count = self._write_queue.submit_sync(_set_checked_block)

            if count > 0:
                self._touch_list(list_id)
            return count

        except Exception as e:
            logger.error(f"[LISTS] _set_checked failed: {e}")
            return 0

    # History & context

    def get_history(
        self,
        name_or_id: Optional[str],
        since: Optional[datetime] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Return change log events for a list.

        Args:
            name_or_id: List name or ID (None returns events for all lists)
            since: Optional datetime filter
            limit: Max events to return

        Returns:
            List of event dicts
        """
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()

                if name_or_id:
                    list_row = self._resolve_list(name_or_id)
                    if not list_row:
                        return []
                    list_id = list_row['id']

                    if since:
                        cursor.execute("""
                            SELECT id, list_id, event_type, item_content, details, created_at
                            FROM list_events
                            WHERE list_id = ? AND created_at >= ?
                            ORDER BY created_at DESC
                            LIMIT ?
                        """, (list_id, since, limit))
                    else:
                        cursor.execute("""
                            SELECT id, list_id, event_type, item_content, details, created_at
                            FROM list_events
                            WHERE list_id = ?
                            ORDER BY created_at DESC
                            LIMIT ?
                        """, (list_id, limit))
                else:
                    if since:
                        cursor.execute("""
                            SELECT le.id, le.list_id, le.event_type, le.item_content,
                                   le.details, le.created_at
                            FROM list_events le
                            WHERE le.created_at >= ?
                            ORDER BY le.created_at DESC
                            LIMIT ?
                        """, (since, limit))
                    else:
                        cursor.execute("""
                            SELECT id, list_id, event_type, item_content, details, created_at
                            FROM list_events
                            ORDER BY created_at DESC
                            LIMIT ?
                        """, (limit,))

                rows = cursor.fetchall()
                cursor.close()

            return [
                {
                    'id': row[0],
                    'list_id': row[1],
                    'event_type': row[2],
                    'item_content': row[3],
                    'details': row[4],
                    'created_at': row[5],
                }
                for row in rows
            ]

        except Exception as e:
            logger.error(f"[LISTS] get_history failed: {e}")
            return []

    def get_lists_for_prompt(self) -> str:
        """
        Format active lists summary for LLM prompt injection.

        Returns compact representation with recency cues so the LLM
        knows what lists exist without needing to load all items.

        Returns:
            Formatted string or empty string if no lists
        """
        lists = self.get_all_lists()
        if not lists:
            return ""

        now = datetime.now(timezone.utc)
        lines = ["## Active Lists"]

        for lst in lists:
            item_count = lst['item_count']
            checked_count = lst['checked_count']
            updated_at = lst['updated_at']

            # Recency cue
            if updated_at:
                try:
                    if not updated_at.tzinfo:
                        updated_at = updated_at.replace(tzinfo=timezone.utc)
                    delta = now - updated_at
                    total_seconds = int(delta.total_seconds())
                    if total_seconds < 3600:
                        recency = f"{total_seconds // 60}m ago"
                    elif total_seconds < 86400:
                        recency = f"{total_seconds // 3600}h ago"
                    else:
                        recency = f"{delta.days} days ago"
                except Exception as e:
                    logger.debug(f"[LIST] recency calculation failed: {e}")
                    recency = "unknown"
            else:
                recency = "unknown"

            # Count summary
            if checked_count > 0:
                count_str = f"{item_count} items, {checked_count} checked"
            else:
                count_str = f"{item_count} items"

            lines.append(f"- {lst['name']} ({count_str}) — updated {recency}")

        return "\n".join(lines)

    # Internal helpers

    def _resolve_list(self, name_or_id: str) -> Optional[Dict[str, Any]]:
        """
        Resolve a list by exact ID first, then case-insensitive name.

        Args:
            name_or_id: List name or 8-char hex ID

        Returns:
            List dict or None
        """
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()

                # Try exact ID match first
                cursor.execute("""
                    SELECT id, name, list_type, updated_at
                    FROM lists
                    WHERE id = ? AND deleted_at IS NULL
                """, (name_or_id,))
                row = cursor.fetchone()

                if not row:
                    # Try case-insensitive name match
                    cursor.execute("""
                        SELECT id, name, list_type, updated_at
                        FROM lists
                        WHERE LOWER(name) = LOWER(?) AND deleted_at IS NULL
                        LIMIT 1
                    """, (name_or_id,))
                    row = cursor.fetchone()

                cursor.close()

            if row:
                return {'id': row[0], 'name': row[1], 'list_type': row[2], 'updated_at': row[3]}
            return None

        except Exception as e:
            logger.error(f"[LISTS] _resolve_list failed: {e}")
            return None

    def get_most_recent_list(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recently updated active list.

        Returns:
            List dict or None if no lists
        """
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, name, list_type, updated_at
                    FROM lists
                    WHERE deleted_at IS NULL
                    ORDER BY updated_at DESC
                    LIMIT 1
                """)
                row = cursor.fetchone()
                cursor.close()

            if row:
                return {'id': row[0], 'name': row[1], 'list_type': row[2], 'updated_at': row[3]}
            return None

        except Exception as e:
            logger.error(f"[LISTS] get_most_recent_list failed: {e}")
            return None

    def _log_event(
        self,
        list_id: str,
        event_type: str,
        item_content: Optional[str] = None,
        details: Optional[Dict] = None,
    ) -> None:
        """
        Write an event to list_events.

        Args:
            list_id: List identifier
            event_type: Event type string
            item_content: Optional item content
            details: Optional details dict
        """
        import json
        event_id = secrets.token_hex(4)
        details_json = json.dumps(details or {})

        def _insert_event(
            _eid=event_id, _lid=list_id, _et=event_type,
            _ic=item_content, _dj=details_json, _db=self.db,
        ):
            """Insert a list event record (fire-and-forget).

            Args:
                _eid: Pre-generated 8-char hex event ID.
                _lid: List ID this event belongs to.
                _et: Event type string (e.g. ``'item_added'``).
                _ic: Optional item content string.
                _dj: JSON-serialised details dict.
                _db: DatabaseService instance.
            """
            with _db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO list_events (id, list_id, event_type, item_content, details, created_at)
                    VALUES (?, ?, ?, ?, ?, datetime('now'))
                """, (_eid, _lid, _et, _ic, _dj))
                cursor.close()

        try:
            self._write_queue.submit(_insert_event)
        except Exception as e:
            logger.warning(f"[LISTS] _log_event failed (non-fatal): {e}")

    def _touch_list(self, list_id: str) -> None:
        """
        Update lists.updated_at to now.

        Args:
            list_id: List identifier
        """
        def _touch(_id=list_id, _db=self.db):
            """Update lists.updated_at to now (fire-and-forget).

            Args:
                _id: List ID to touch.
                _db: DatabaseService instance.
            """
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
