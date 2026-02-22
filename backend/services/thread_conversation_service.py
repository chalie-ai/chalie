"""
Thread Conversation Service - Redis-backed conversation storage per thread.

Replaces TopicConversationService's file-based storage with Redis lists.
Each thread has a conversation list (thread_conv:{thread_id}) and an
exchange count index (thread_conv_index:{thread_id}).
"""

import json
import uuid
import logging
import time
from datetime import datetime
from typing import Optional, List

from services.redis_client import RedisClientService


class ThreadConversationService:
    """Manages conversation exchanges stored in Redis, scoped by thread."""

    TTL_SECONDS = 86400  # 24 hours
    MAX_EXCHANGES = 50

    def __init__(self):
        self.redis = RedisClientService.create_connection()

    def _conv_key(self, thread_id: str) -> str:
        return f"thread_conv:{thread_id}"

    def _index_key(self, thread_id: str) -> str:
        return f"thread_conv_index:{thread_id}"

    def _refresh_ttl(self, thread_id: str):
        """Refresh TTL on all conversation keys."""
        pipe = self.redis.pipeline()
        pipe.expire(self._conv_key(thread_id), self.TTL_SECONDS)
        pipe.expire(self._index_key(thread_id), self.TTL_SECONDS)
        pipe.execute()

    def add_exchange(self, thread_id: str, topic: str, prompt_data: dict) -> str:
        """
        Add a new exchange with prompt data. Response is added later.

        Args:
            thread_id: Thread identifier
            topic: Current topic name
            prompt_data: Dict with message, classification_time, etc.

        Returns:
            exchange_id: UUID for this exchange
        """
        exchange_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')

        exchange = {
            "id": exchange_id,
            "topic": topic,
            "prompt": {
                "id": exchange_id,
                "message": prompt_data.get("message", ""),
                "time": timestamp,
                "classification_time": prompt_data.get("classification_time", 0),
            },
            "response": None,
            "steps": [],
            "memory_chunk": {},
        }

        conv_key = self._conv_key(thread_id)
        pipe = self.redis.pipeline()
        pipe.rpush(conv_key, json.dumps(exchange))
        pipe.incr(self._index_key(thread_id))
        pipe.expire(conv_key, self.TTL_SECONDS)
        pipe.expire(self._index_key(thread_id), self.TTL_SECONDS)
        pipe.execute()

        # Trim to max exchanges
        self.redis.ltrim(conv_key, -self.MAX_EXCHANGES, -1)

        logging.debug(f"[THREAD_CONV] Added exchange {exchange_id[:8]} to thread {thread_id}")
        return exchange_id

    def add_response(self, thread_id: str, response_message: str, generation_time: float) -> None:
        """Add a response to the most recent exchange."""
        conv_key = self._conv_key(thread_id)
        raw = self.redis.lindex(conv_key, -1)
        if not raw:
            logging.warning(f"[THREAD_CONV] No exchange found in thread {thread_id}")
            return

        exchange = json.loads(raw)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
        exchange["response"] = {
            "message": response_message,
            "time": timestamp,
            "generation_time": generation_time,
        }

        self.redis.lset(conv_key, -1, json.dumps(exchange))
        self._refresh_ttl(thread_id)

    def add_response_error(self, thread_id: str, error_message: str) -> None:
        """Record an error for the most recent exchange."""
        conv_key = self._conv_key(thread_id)
        raw = self.redis.lindex(conv_key, -1)
        if not raw:
            return

        exchange = json.loads(raw)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
        exchange["response"] = {
            "error": error_message,
            "time": timestamp,
        }

        self.redis.lset(conv_key, -1, json.dumps(exchange))
        self._refresh_ttl(thread_id)

    def add_steps_to_exchange(self, thread_id: str, next_actions: list) -> None:
        """Add steps to the most recent exchange."""
        conv_key = self._conv_key(thread_id)
        raw = self.redis.lindex(conv_key, -1)
        if not raw:
            return

        exchange = json.loads(raw)
        steps = []
        for action in next_actions:
            step = {
                "type": action.get("type", "task"),
                "description": action.get("description", ""),
                "status": "pending",
            }
            if "when" in action:
                step["when"] = action["when"]
            if "query" in action:
                step["query"] = action["query"]
            steps.append(step)

        exchange["steps"] = steps
        self.redis.lset(conv_key, -1, json.dumps(exchange))
        self._refresh_ttl(thread_id)

    def add_memory_chunk(self, thread_id: str, exchange_id: str, memory_chunk: dict) -> bool:
        """Add or merge a memory chunk to a specific exchange by ID.

        With per-message encoding, this may be called twice per exchange:
        once for the user message (Phase A) and once for the assistant response
        (Phase D). Gists are merged so both survive in conversation history.

        Returns:
            True if the exchange was found and updated, False if not found.
        """
        conv_key = self._conv_key(thread_id)
        all_raw = self.redis.lrange(conv_key, 0, -1)

        for i, raw in enumerate(all_raw):
            exchange = json.loads(raw)
            if exchange.get("id") == exchange_id or exchange.get("prompt", {}).get("id") == exchange_id:
                existing = exchange.get("memory_chunk") or {}
                if existing.get("gists"):
                    # Merge gists from both encodes (user message + assistant response)
                    merged_gists = existing["gists"] + memory_chunk.get("gists", [])
                    memory_chunk = {**memory_chunk, "gists": merged_gists}
                exchange["memory_chunk"] = memory_chunk
                self.redis.lset(conv_key, i, json.dumps(exchange))
                self._refresh_ttl(thread_id)
                return True

        logging.warning(f"[THREAD_CONV] Exchange {exchange_id[:8]} not found in thread {thread_id}")
        return False

    def get_conversation_history(self, thread_id: str) -> list:
        """Get all exchanges for a thread."""
        conv_key = self._conv_key(thread_id)
        all_raw = self.redis.lrange(conv_key, 0, -1)

        exchanges = []
        for raw in all_raw:
            try:
                exchanges.append(json.loads(raw))
            except (json.JSONDecodeError, TypeError):
                continue
        return exchanges

    def get_latest_exchange_id(self, thread_id: str) -> str:
        """Get the ID of the most recent exchange."""
        conv_key = self._conv_key(thread_id)
        raw = self.redis.lindex(conv_key, -1)
        if not raw:
            return "unknown"

        exchange = json.loads(raw)
        return exchange.get("id", exchange.get("prompt", {}).get("id", "unknown"))

    def get_active_steps(self, thread_id: str) -> list:
        """
        Get all active (non-completed) steps across all exchanges.
        Replaces WorldStateService.get_world_state() for step tracking.
        """
        exchanges = self.get_conversation_history(thread_id)
        active_steps = []
        for exchange in exchanges:
            for step in exchange.get("steps", []):
                status = step.get("status", "pending")
                if status in ("scheduled", "in progress", "pending"):
                    active_steps.append(step)
        return active_steps

    def get_exchange_count(self, thread_id: str) -> int:
        """Get exchange count (O(1) via index key)."""
        count = self.redis.get(self._index_key(thread_id))
        return int(count) if count else 0

    def remove_exchanges(self, thread_id: str, exchange_ids: list) -> None:
        """Remove specific exchanges by ID (for post-episodic cleanup)."""
        conv_key = self._conv_key(thread_id)
        all_raw = self.redis.lrange(conv_key, 0, -1)

        ids_to_remove = set(exchange_ids)
        kept = []
        for raw in all_raw:
            try:
                exchange = json.loads(raw)
                eid = exchange.get("id") or exchange.get("prompt", {}).get("id")
                if eid not in ids_to_remove:
                    kept.append(raw)
            except (json.JSONDecodeError, TypeError):
                kept.append(raw)

        removed_count = len(all_raw) - len(kept)
        if removed_count > 0:
            pipe = self.redis.pipeline()
            pipe.delete(conv_key)
            for item in kept:
                pipe.rpush(conv_key, item)
            pipe.expire(conv_key, self.TTL_SECONDS)
            pipe.execute()
            logging.info(f"[THREAD_CONV] Removed {removed_count} exchanges from thread {thread_id}")
