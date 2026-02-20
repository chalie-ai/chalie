"""
Working Memory Service - Rolling buffer of raw conversation turns.

Redis-based FIFO buffer of the last N raw turns for immediate context.
Unlike gists (compressed summaries), this stores verbatim user/assistant turns.
"""

import json
import time
import logging
from typing import List, Dict
from services.redis_client import RedisClientService


class WorkingMemoryService:
    """Manages a rolling buffer of raw conversation turns in Redis."""

    def __init__(self, max_turns: int = 4):
        """
        Initialize working memory service.

        Args:
            max_turns: Maximum number of turns to keep (default 4).
                       Stored as 2x entries (role + content per turn).
        """
        self.redis = RedisClientService.create_connection()
        self.max_turns = max_turns
        self.max_entries = max_turns * 2  # Each turn has role + content

    def _get_memory_key(self, identifier: str) -> str:
        """Generate Redis key for working memory list.

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
        pipe = self.redis.pipeline()
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
            raw_entries = self.redis.lrange(memory_key, -entries, -1)
        else:
            raw_entries = self.redis.lrange(memory_key, 0, -1)

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
        self.redis.delete(memory_key)

    def get_buffer_size(self, identifier: str) -> int:
        """
        Get current number of entries in the buffer.

        Args:
            identifier: Thread ID or topic name

        Returns:
            Number of entries
        """
        memory_key = self._get_memory_key(identifier)
        return self.redis.llen(memory_key)
