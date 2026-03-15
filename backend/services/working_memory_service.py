"""
Working Memory Service - Rolling buffer of raw conversation turns.

MemoryStore-based FIFO buffer of the last N raw turns for immediate context.
Unlike gists (compressed summaries), this stores verbatim user/assistant turns.
"""

import json
import time
import logging
from typing import List, Dict
from services.memory_client import MemoryClientService


class WorkingMemoryService:
    """Manages a rolling buffer of raw conversation turns in MemoryStore."""

    def __init__(self, max_turns: int = 12):
        """
        Initialize working memory service.

        Args:
            max_turns: Maximum number of turns to keep (default 4).
                       Stored as 2x entries (role + content per turn).
        """
        self.store = MemoryClientService.create_connection()
        self.max_turns = max_turns
        self.max_entries = max_turns * 2  # Each turn has role + content

    def _get_memory_key(self, identifier: str) -> str:
        """Generate MemoryStore key for working memory list.

        Args:
            identifier: thread_id (preferred) or topic name (legacy)
        """
        return f"working_memory:{identifier}"

    def append_turn(self, identifier: str, role: str, content: str) -> int:
        """
        Append a turn to working memory.

        Args:
            identifier: Thread ID or topic name (for backward compat)
            role: Role identifier ("user" or "assistant")
            content: Raw message content

        Returns:
            Current number of entries in the buffer
        """
        memory_key = self._get_memory_key(identifier)

        turn_data = json.dumps({
            'role': role,
            'content': content,
            'timestamp': time.time()
        })

        # RPUSH + LTRIM for FIFO eviction + safety TTL
        pipe = self.store.pipeline()
        pipe.rpush(memory_key, turn_data)
        pipe.ltrim(memory_key, -self.max_entries, -1)
        pipe.expire(memory_key, 86400)  # 24-hour safety TTL, refreshed each append
        results = pipe.execute()

        current_length = results[0]  # rpush returns new length
        logging.debug(
            f"[WORKING MEMORY] Appended {role} turn for '{identifier}' "
            f"(buffer size: {min(current_length, self.max_entries)})"
        )

        return min(current_length, self.max_entries)

    def get_recent_turns(self, identifier: str, n: int = None) -> List[Dict]:
        """
        Get the most recent N turns from working memory.

        Args:
            identifier: Thread ID or topic name
            n: Number of turns to retrieve (default: all available)

        Returns:
            List of turn dicts with 'role', 'content', 'timestamp'
        """
        memory_key = self._get_memory_key(identifier)

        if n is not None:
            entries = n * 2
            raw_entries = self.store.lrange(memory_key, -entries, -1)
        else:
            raw_entries = self.store.lrange(memory_key, 0, -1)

        turns = []
        for entry in raw_entries:
            try:
                turns.append(json.loads(entry))
            except (json.JSONDecodeError, TypeError):
                continue

        return turns

    def get_formatted_context(self, identifier: str, n: int = None) -> str:
        """
        Get working memory formatted for prompt injection.

        Args:
            identifier: Thread ID or topic name
            n: Number of turns (default: all)

        Returns:
            Formatted working memory string or empty string
        """
        turns = self.get_recent_turns(identifier, n)

        if not turns:
            return ""

        lines = ["## Recent Conversation"]
        for turn in turns:
            role = turn.get('role', 'unknown').capitalize()
            content = turn.get('content', '')
            lines.append(f"{role}: {content}")

        return "\n".join(lines)

    def clear(self, identifier: str):
        """
        Clear working memory.

        Args:
            identifier: Thread ID or topic name
        """
        memory_key = self._get_memory_key(identifier)
        self.store.delete(memory_key)

    def get_buffer_size(self, identifier: str) -> int:
        """
        Get current number of entries in the buffer.

        Args:
            identifier: Thread ID or topic name

        Returns:
            Number of entries
        """
        memory_key = self._get_memory_key(identifier)
        return self.store.llen(memory_key)

    def hydrate_from_db(self, identifier: str) -> int:
        """
        Repopulate working memory from interaction_log (SQLite).

        Called on container restart when MemoryStore is empty but
        conversation history exists in the database. Loads the most
        recent user_input and system_response events.

        Args:
            identifier: Thread ID or topic name

        Returns:
            Number of turns loaded (0 if already populated or no history)
        """
        # Skip if already populated
        if self.get_buffer_size(identifier) > 0:
            return 0

        try:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()

            with db.connection() as conn:
                cursor = conn.cursor()
                # Get the most recent user/system exchanges (newest first, then reverse)
                cursor.execute("""
                    SELECT event_type, payload, created_at
                    FROM interaction_log
                    WHERE event_type IN ('user_input', 'system_response')
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (self.max_entries,))
                rows = cursor.fetchall()

            if not rows:
                return 0

            # Reverse to chronological order
            rows.reverse()

            loaded = 0
            for row in rows:
                event_type, payload_raw, created_at = row
                try:
                    payload = json.loads(payload_raw) if isinstance(payload_raw, str) else (payload_raw or {})
                except (json.JSONDecodeError, TypeError):
                    continue

                if event_type == 'user_input':
                    content = payload.get('text', payload.get('message', ''))
                    if content:
                        self.append_turn(identifier, 'user', content)
                        loaded += 1
                elif event_type == 'system_response':
                    content = payload.get('response', payload.get('text', ''))
                    if content:
                        self.append_turn(identifier, 'assistant', content)
                        loaded += 1

            if loaded:
                logging.info(
                    f"[WORKING MEMORY] Hydrated {loaded} turns from database "
                    f"for '{identifier}'"
                )
            return loaded

        except Exception as e:
            logging.warning(f"[WORKING MEMORY] Hydration failed (non-fatal): {e}")
            return 0
