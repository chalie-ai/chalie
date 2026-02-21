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

logger = logging.getLogger(__name__)


class ListService:
    """Manages deterministic user lists with history tracking."""

    def __init__(self, db_service):
        """
        Initialize list service.

        Args:
            db_service: DatabaseService instance
        """
        self.db = db_service

    # ─────────────────────────────────────────────
    # List operations
    # ─────────────────────────────────────────────

    def create_list(
        self,
        name: str,
        list_type: str = 'checklist',
        user_id: str = 'primary',
    ) -> str:
        """
        Create a new list.

        Args:
            name: List name (e.g. "Shopping List")
            list_type: List type (default 'checklist')
            user_id: User identifier

        Returns:
            list_id (8-char hex string)

        Raises:
            ValueError: If a list with that name already exists for the user
        """
        list_id = secrets.token_hex(4)

        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO lists (id, user_id, name, list_type, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, NOW(), NOW())
                """, (list_id, user_id, name, list_type))
                cursor.close()

            self._log_event(list_id, 'list_created', details={'name': name})
            logger.info(f"[LISTS] Created list '{name}' (id={list_id})")
            return list_id

        except Exception as e:
            if 'unique' in str(e).lower() or 'duplicate' in str(e).lower():
                raise ValueError(f"A list named '{name}' already exists.") from e
            logger.error(f"[LISTS] Failed to create list: {e}")
            raise

    def delete_list(
        self,
        name_or_id: str,
        user_id: str = 'primary',
    ) -> bool:
        """
        Soft-delete a list.

        Args:
            name_or_id: List name or ID
            user_id: User identifier

        Returns:
            True if deleted, False if not found
        """
        list_row = self._resolve_list(name_or_id, user_id)
        if not list_row:
            return False

        list_id = list_row['id']
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE lists SET deleted_at = NOW(), updated_at = NOW()
                    WHERE id = %s AND user_id = %s AND deleted_at IS NULL
                """, (list_id, user_id))
                updated = cursor.rowcount > 0
                cursor.close()

            if updated:
                self._log_event(list_id, 'list_deleted', details={'name': list_row['name']})
                logger.info(f"[LISTS] Deleted list '{list_row['name']}' (id={list_id})")
            return updated

        except Exception as e:
            logger.error(f"[LISTS] delete_list failed: {e}")
            return False

    def clear_list(
        self,
        name_or_id: str,
        user_id: str = 'primary',
    ) -> int:
        """
        Soft-delete all items in a list.

        Args:
            name_or_id: List name or ID
            user_id: User identifier

        Returns:
            Count of items removed, or -1 if list not found
        """
        list_row = self._resolve_list(name_or_id, user_id)
        if not list_row:
            return -1

        list_id = list_row['id']
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE list_items SET removed_at = NOW(), updated_at = NOW()
                    WHERE list_id = %s AND removed_at IS NULL
                """, (list_id,))
                count = cursor.rowcount
                cursor.close()

            if count > 0:
                self._touch_list(list_id)
                self._log_event(list_id, 'list_cleared', details={'count': count})
                logger.info(f"[LISTS] Cleared {count} items from list '{list_row['name']}'")
            return count

        except Exception as e:
            logger.error(f"[LISTS] clear_list failed: {e}")
            return -1

    def rename_list(
        self,
        name_or_id: str,
        new_name: str,
        user_id: str = 'primary',
    ) -> bool:
        """
        Rename a list.

        Args:
            name_or_id: List name or ID
            new_name: New name for the list
            user_id: User identifier

        Returns:
            True if renamed, False if not found or name collision
        """
        list_row = self._resolve_list(name_or_id, user_id)
        if not list_row:
            return False

        # Check for name collision
        existing = self._resolve_list(new_name, user_id)
        if existing and existing['id'] != list_row['id']:
            logger.warning(f"[LISTS] Cannot rename to '{new_name}' — name already in use")
            return False

        list_id = list_row['id']
        old_name = list_row['name']
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE lists SET name = %s, updated_at = NOW()
                    WHERE id = %s AND user_id = %s AND deleted_at IS NULL
                """, (new_name, list_id, user_id))
                updated = cursor.rowcount > 0
                cursor.close()

            if updated:
                self._log_event(list_id, 'list_renamed', details={'old_name': old_name, 'new_name': new_name})
                logger.info(f"[LISTS] Renamed list '{old_name}' → '{new_name}'")
            return updated

        except Exception as e:
            logger.error(f"[LISTS] rename_list failed: {e}")
            return False

    def get_list(
        self,
        name_or_id: str,
        user_id: str = 'primary',
    ) -> Optional[Dict[str, Any]]:
        """
        Get a list with its active items.

        Args:
            name_or_id: List name or ID
            user_id: User identifier

        Returns:
            Dict with list data and items array, or None if not found
        """
        list_row = self._resolve_list(name_or_id, user_id)
        if not list_row:
            return None

        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, content, checked, position, added_at, updated_at
                    FROM list_items
                    WHERE list_id = %s AND removed_at IS NULL
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

    def get_all_lists(
        self,
        user_id: str = 'primary',
    ) -> List[Dict[str, Any]]:
        """
        Get all active lists with summary counts.

        Args:
            user_id: User identifier

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
                        COUNT(li.id) FILTER (WHERE li.removed_at IS NULL)        AS item_count,
                        COUNT(li.id) FILTER (WHERE li.removed_at IS NULL AND li.checked) AS checked_count
                    FROM lists l
                    LEFT JOIN list_items li ON li.list_id = l.id
                    WHERE l.user_id = %s AND l.deleted_at IS NULL
                    GROUP BY l.id, l.name, l.list_type, l.updated_at
                    ORDER BY l.updated_at DESC
                """, (user_id,))
                rows = cursor.fetchall()
                cursor.close()

            return [
                {
                    'id': row[0],
                    'name': row[1],
                    'list_type': row[2],
                    'updated_at': row[3],
                    'item_count': row[4],
                    'checked_count': row[5],
                }
                for row in rows
            ]

        except Exception as e:
            logger.error(f"[LISTS] get_all_lists failed: {e}")
            return []

    # ─────────────────────────────────────────────
    # Item operations (batch)
    # ─────────────────────────────────────────────

    def add_items(
        self,
        name_or_id: str,
        items: List[str],
        user_id: str = 'primary',
        dedupe: bool = True,
        auto_create: bool = True,
    ) -> int:
        """
        Add items to a list.

        Args:
            name_or_id: List name or ID
            items: List of item content strings
            user_id: User identifier
            dedupe: Skip items already on the list (case-insensitive, default True)
            auto_create: Create the list if it doesn't exist (default True)

        Returns:
            Count of items actually added
        """
        list_row = self._resolve_list(name_or_id, user_id)

        if not list_row:
            if not auto_create:
                logger.warning(f"[LISTS] List '{name_or_id}' not found")
                return 0
            list_id = self.create_list(name_or_id, user_id=user_id)
            list_row = {'id': list_id, 'name': name_or_id}

        list_id = list_row['id']
        if not items:
            return 0

        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()

                # Get current max position
                cursor.execute("""
                    SELECT COALESCE(MAX(position), -1)
                    FROM list_items
                    WHERE list_id = %s AND removed_at IS NULL
                """, (list_id,))
                max_pos = cursor.fetchone()[0]

                # Get existing active items for dedup
                existing_normalized = set()
                if dedupe:
                    cursor.execute("""
                        SELECT LOWER(TRIM(content))
                        FROM list_items
                        WHERE list_id = %s AND removed_at IS NULL
                    """, (list_id,))
                    existing_normalized = {row[0] for row in cursor.fetchall()}

                added = 0
                for item_content in items:
                    if not item_content or not item_content.strip():
                        continue

                    normalized = item_content.strip().lower()

                    # Dedupe check
                    if dedupe and normalized in existing_normalized:
                        continue

                    # Check if this item was previously removed (restore instead of insert)
                    cursor.execute("""
                        SELECT id FROM list_items
                        WHERE list_id = %s
                          AND LOWER(TRIM(content)) = %s
                          AND removed_at IS NOT NULL
                        ORDER BY removed_at DESC
                        LIMIT 1
                    """, (list_id, normalized))
                    removed_row = cursor.fetchone()

                    if removed_row:
                        # Restore the soft-deleted row
                        max_pos += 1
                        cursor.execute("""
                            UPDATE list_items
                            SET removed_at = NULL, checked = FALSE,
                                position = %s, updated_at = NOW()
                            WHERE id = %s
                        """, (max_pos, removed_row[0]))
                    else:
                        # Insert new row
                        max_pos += 1
                        item_id = secrets.token_hex(4)
                        cursor.execute("""
                            INSERT INTO list_items (id, list_id, content, position, added_at, updated_at)
                            VALUES (%s, %s, %s, %s, NOW(), NOW())
                        """, (item_id, list_id, item_content.strip(), max_pos))

                    existing_normalized.add(normalized)
                    added += 1

                cursor.close()

            if added > 0:
                self._touch_list(list_id)
                for item_content in items[:added]:
                    self._log_event(
                        list_id, 'item_added',
                        item_content=item_content.strip(),
                        details={'normalized_content': item_content.strip().lower()},
                    )
                logger.info(f"[LISTS] Added {added} items to list '{list_row['name']}'")

            return added

        except Exception as e:
            logger.error(f"[LISTS] add_items failed: {e}")
            return 0

    def remove_items(
        self,
        name_or_id: str,
        items: List[str],
        user_id: str = 'primary',
    ) -> int:
        """
        Soft-remove items from a list (case-insensitive match).

        Args:
            name_or_id: List name or ID
            items: List of item content strings to remove
            user_id: User identifier

        Returns:
            Count of items removed
        """
        list_row = self._resolve_list(name_or_id, user_id)
        if not list_row or not items:
            return 0

        list_id = list_row['id']
        removed = 0

        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                for item_content in items:
                    normalized = item_content.strip().lower()
                    cursor.execute("""
                        UPDATE list_items
                        SET removed_at = NOW(), updated_at = NOW()
                        WHERE list_id = %s
                          AND LOWER(TRIM(content)) = %s
                          AND removed_at IS NULL
                    """, (list_id, normalized))
                    if cursor.rowcount > 0:
                        removed += cursor.rowcount
                        self._log_event(
                            list_id, 'item_removed',
                            item_content=item_content.strip(),
                            details={'normalized_content': normalized},
                        )
                cursor.close()

            if removed > 0:
                self._touch_list(list_id)
                logger.info(f"[LISTS] Removed {removed} items from list '{list_row['name']}'")
            return removed

        except Exception as e:
            logger.error(f"[LISTS] remove_items failed: {e}")
            return 0

    def check_items(
        self,
        name_or_id: str,
        items: List[str],
        user_id: str = 'primary',
    ) -> int:
        """
        Check off items in a list (case-insensitive match).

        Args:
            name_or_id: List name or ID
            items: List of item content strings to check
            user_id: User identifier

        Returns:
            Count of items checked
        """
        return self._set_checked(name_or_id, items, user_id, checked=True)

    def uncheck_items(
        self,
        name_or_id: str,
        items: List[str],
        user_id: str = 'primary',
    ) -> int:
        """
        Uncheck items in a list (case-insensitive match).

        Args:
            name_or_id: List name or ID
            items: List of item content strings to uncheck
            user_id: User identifier

        Returns:
            Count of items unchecked
        """
        return self._set_checked(name_or_id, items, user_id, checked=False)

    def _set_checked(
        self,
        name_or_id: str,
        items: List[str],
        user_id: str,
        checked: bool,
    ) -> int:
        list_row = self._resolve_list(name_or_id, user_id)
        if not list_row or not items:
            return 0

        list_id = list_row['id']
        event_type = 'item_checked' if checked else 'item_unchecked'
        count = 0

        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                for item_content in items:
                    normalized = item_content.strip().lower()
                    cursor.execute("""
                        UPDATE list_items
                        SET checked = %s, updated_at = NOW()
                        WHERE list_id = %s
                          AND LOWER(TRIM(content)) = %s
                          AND removed_at IS NULL
                    """, (checked, list_id, normalized))
                    if cursor.rowcount > 0:
                        count += cursor.rowcount
                        self._log_event(
                            list_id, event_type,
                            item_content=item_content.strip(),
                            details={'normalized_content': normalized},
                        )
                cursor.close()

            if count > 0:
                self._touch_list(list_id)
            return count

        except Exception as e:
            logger.error(f"[LISTS] _set_checked failed: {e}")
            return 0

    # ─────────────────────────────────────────────
    # History & context
    # ─────────────────────────────────────────────

    def get_history(
        self,
        name_or_id: Optional[str],
        since: Optional[datetime] = None,
        limit: int = 50,
        user_id: str = 'primary',
    ) -> List[Dict[str, Any]]:
        """
        Return change log events for a list.

        Args:
            name_or_id: List name or ID (None returns events for all lists)
            since: Optional datetime filter
            limit: Max events to return
            user_id: User identifier

        Returns:
            List of event dicts
        """
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()

                if name_or_id:
                    list_row = self._resolve_list(name_or_id, user_id)
                    if not list_row:
                        return []
                    list_id = list_row['id']

                    if since:
                        cursor.execute("""
                            SELECT id, list_id, event_type, item_content, details, created_at
                            FROM list_events
                            WHERE list_id = %s AND created_at >= %s
                            ORDER BY created_at DESC
                            LIMIT %s
                        """, (list_id, since, limit))
                    else:
                        cursor.execute("""
                            SELECT id, list_id, event_type, item_content, details, created_at
                            FROM list_events
                            WHERE list_id = %s
                            ORDER BY created_at DESC
                            LIMIT %s
                        """, (list_id, limit))
                else:
                    # All lists for this user
                    if since:
                        cursor.execute("""
                            SELECT le.id, le.list_id, le.event_type, le.item_content,
                                   le.details, le.created_at
                            FROM list_events le
                            JOIN lists l ON l.id = le.list_id
                            WHERE l.user_id = %s AND le.created_at >= %s
                            ORDER BY le.created_at DESC
                            LIMIT %s
                        """, (user_id, since, limit))
                    else:
                        cursor.execute("""
                            SELECT le.id, le.list_id, le.event_type, le.item_content,
                                   le.details, le.created_at
                            FROM list_events le
                            JOIN lists l ON l.id = le.list_id
                            WHERE l.user_id = %s
                            ORDER BY le.created_at DESC
                            LIMIT %s
                        """, (user_id, limit))

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

    def get_lists_for_prompt(
        self,
        user_id: str = 'primary',
    ) -> str:
        """
        Format active lists summary for LLM prompt injection.

        Returns compact representation with recency cues so the LLM
        knows what lists exist without needing to load all items.

        Args:
            user_id: User identifier

        Returns:
            Formatted string or empty string if no lists
        """
        lists = self.get_all_lists(user_id)
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
                except Exception:
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

    # ─────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────

    def _resolve_list(
        self,
        name_or_id: str,
        user_id: str = 'primary',
    ) -> Optional[Dict[str, Any]]:
        """
        Resolve a list by exact ID first, then case-insensitive name.

        Args:
            name_or_id: List name or 8-char hex ID
            user_id: User identifier

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
                    WHERE id = %s AND user_id = %s AND deleted_at IS NULL
                """, (name_or_id, user_id))
                row = cursor.fetchone()

                if not row:
                    # Try case-insensitive name match
                    cursor.execute("""
                        SELECT id, name, list_type, updated_at
                        FROM lists
                        WHERE user_id = %s AND LOWER(name) = LOWER(%s) AND deleted_at IS NULL
                        LIMIT 1
                    """, (user_id, name_or_id))
                    row = cursor.fetchone()

                cursor.close()

            if row:
                return {'id': row[0], 'name': row[1], 'list_type': row[2], 'updated_at': row[3]}
            return None

        except Exception as e:
            logger.error(f"[LISTS] _resolve_list failed: {e}")
            return None

    def get_most_recent_list(
        self,
        user_id: str = 'primary',
    ) -> Optional[Dict[str, Any]]:
        """
        Get the most recently updated active list.

        Args:
            user_id: User identifier

        Returns:
            List dict or None if no lists
        """
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, name, list_type, updated_at
                    FROM lists
                    WHERE user_id = %s AND deleted_at IS NULL
                    ORDER BY updated_at DESC
                    LIMIT 1
                """, (user_id,))
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

        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO list_events (id, list_id, event_type, item_content, details, created_at)
                    VALUES (%s, %s, %s, %s, %s, NOW())
                """, (event_id, list_id, event_type, item_content, details_json))
                cursor.close()
        except Exception as e:
            logger.warning(f"[LISTS] _log_event failed (non-fatal): {e}")

    def _touch_list(self, list_id: str) -> None:
        """
        Update lists.updated_at to now.

        Args:
            list_id: List identifier
        """
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE lists SET updated_at = NOW() WHERE id = %s
                """, (list_id,))
                cursor.close()
        except Exception as e:
            logger.warning(f"[LISTS] _touch_list failed (non-fatal): {e}")
