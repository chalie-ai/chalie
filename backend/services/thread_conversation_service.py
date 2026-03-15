"""
Thread Conversation Service - MemoryStore-backed conversation storage per thread.

Write-through to SQLite so chat history survives server restarts.
MemoryStore is the hot read/write path; SQLite is the durable fallback
consulted only when MemoryStore returns empty (restart, TTL expiry).
"""

import json
import uuid
import logging
import time
from typing import Optional, List

from services.memory_client import MemoryClientService
from services.time_utils import utc_now, parse_utc

logger = logging.getLogger(__name__)

# Auto-truncation: keep at most this many exchanges globally in SQLite.
MAX_SQLITE_EXCHANGES = 10_000
# Run the purge check every N inserts (amortised cost).
_PURGE_EVERY = 100


class ThreadConversationService:
    """Manages conversation exchanges stored in MemoryStore, scoped by thread."""

    TTL_SECONDS = 86400  # 24 hours
    MAX_EXCHANGES = 120

    def __init__(self):
        self.store = MemoryClientService.create_connection()
        self._insert_counter = 0

    # ── internal helpers ──────────────────────────────────────────

    def _conv_key(self, thread_id: str) -> str:
        return f"thread_conv:{thread_id}"

    def _index_key(self, thread_id: str) -> str:
        return f"thread_conv_index:{thread_id}"

    def _refresh_ttl(self, thread_id: str):
        """Refresh TTL on all conversation keys."""
        pipe = self.store.pipeline()
        pipe.expire(self._conv_key(thread_id), self.TTL_SECONDS)
        pipe.expire(self._index_key(thread_id), self.TTL_SECONDS)
        pipe.execute()

    @property
    def _db(self):
        if not hasattr(self, '_db_service') or self._db_service is None:
            from services.database_service import get_shared_db_service
            self._db_service = get_shared_db_service()
        return self._db_service

    def _persist_exchange(self, exchange_id: str, thread_id: str, topic: str, prompt_message: str, prompt_time: str):
        """Write-through: INSERT exchange row into SQLite."""
        try:
            self._db.execute(
                """INSERT OR IGNORE INTO thread_exchanges
                   (id, thread_id, topic, prompt_message, prompt_time)
                   VALUES (?, ?, ?, ?, ?)""",
                (exchange_id, thread_id, topic, prompt_message, prompt_time)
            )
            self._insert_counter += 1
            if self._insert_counter % _PURGE_EVERY == 0:
                self._maybe_purge()
        except Exception as e:
            logger.debug(f"[THREAD_CONV] SQLite persist failed: {e}")

    def _persist_response(self, thread_id: str, exchange_id: str, response_message: str = None,
                          response_error: str = None, generation_time: float = None):
        """Write-through: UPDATE the latest exchange with its response."""
        try:
            self._db.execute(
                """UPDATE thread_exchanges
                   SET response_message = ?, response_time = ?,
                       response_error = ?, generation_time_ms = ?
                   WHERE id = ?""",
                (response_message, utc_now().isoformat(), response_error,
                 generation_time, exchange_id)
            )
        except Exception as e:
            logger.debug(f"[THREAD_CONV] SQLite response persist failed: {e}")

    def _persist_json_field(self, exchange_id: str, field: str, value):
        """Write-through: UPDATE a JSON column by exchange_id."""
        try:
            self._db.execute(
                f"UPDATE thread_exchanges SET {field} = ? WHERE id = ?",
                (json.dumps(value), exchange_id)
            )
        except Exception as e:
            logger.debug(f"[THREAD_CONV] SQLite {field} persist failed: {e}")

    def _maybe_purge(self):
        """Delete oldest exchanges when table exceeds MAX_SQLITE_EXCHANGES."""
        try:
            rows = self._db.fetch_all(
                "SELECT COUNT(*) AS cnt FROM thread_exchanges"
            )
            count = rows[0]["cnt"] if rows else 0
            if count > MAX_SQLITE_EXCHANGES:
                excess = count - MAX_SQLITE_EXCHANGES
                self._db.execute(
                    """DELETE FROM thread_exchanges WHERE id IN (
                        SELECT id FROM thread_exchanges
                        ORDER BY created_at ASC LIMIT ?
                    )""",
                    (excess,)
                )
                logger.info(f"[THREAD_CONV] Purged {excess} old exchanges (total was {count})")
        except Exception as e:
            logger.debug(f"[THREAD_CONV] Purge check failed: {e}")

    def _load_from_sqlite(self, thread_id: str) -> list:
        """Load exchanges from SQLite and repopulate MemoryStore."""
        try:
            rows = self._db.fetch_all(
                """SELECT id, topic, prompt_message, prompt_time,
                          response_message, response_time, response_error,
                          generation_time_ms, steps, memory_chunk
                   FROM thread_exchanges
                   WHERE thread_id = ?
                   ORDER BY created_at ASC
                   LIMIT ?""",
                (thread_id, self.MAX_EXCHANGES)
            )
        except Exception as e:
            logger.debug(f"[THREAD_CONV] SQLite load failed: {e}")
            return []

        if not rows:
            return []

        exchanges = []
        conv_key = self._conv_key(thread_id)
        pipe = self.store.pipeline()

        for row in rows:
            exchange = {
                "id": row["id"],
                "topic": row["topic"],
                "prompt": {
                    "id": row["id"],
                    "message": row["prompt_message"],
                    "time": row["prompt_time"],
                },
                "response": None,
                "steps": json.loads(row["steps"] or "[]"),
                "memory_chunk": json.loads(row["memory_chunk"] or "{}"),
            }
            if row["response_message"]:
                exchange["response"] = {
                    "message": row["response_message"],
                    "time": row["response_time"],
                    "generation_time": row["generation_time_ms"] or 0,
                }
            elif row["response_error"]:
                exchange["response"] = {
                    "error": row["response_error"],
                    "time": row["response_time"],
                }

            exchanges.append(exchange)
            pipe.rpush(conv_key, json.dumps(exchange))

        if exchanges:
            pipe.expire(conv_key, self.TTL_SECONDS)
            pipe.set(self._index_key(thread_id), str(len(exchanges)))
            pipe.expire(self._index_key(thread_id), self.TTL_SECONDS)
            pipe.execute()

        logger.debug(f"[THREAD_CONV] Loaded {len(exchanges)} exchanges from SQLite for thread {thread_id}")
        return exchanges

    # ── public API ────────────────────────────────────────────────

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
        timestamp = utc_now().strftime('%Y-%m-%d %H:%M')

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
        pipe = self.store.pipeline()
        pipe.rpush(conv_key, json.dumps(exchange))
        pipe.incr(self._index_key(thread_id))
        pipe.expire(conv_key, self.TTL_SECONDS)
        pipe.expire(self._index_key(thread_id), self.TTL_SECONDS)
        pipe.execute()

        # Trim to max exchanges
        self.store.ltrim(conv_key, -self.MAX_EXCHANGES, -1)

        # Write-through to SQLite
        self._persist_exchange(
            exchange_id, thread_id, topic,
            prompt_data.get("message", ""), utc_now().isoformat()
        )

        logging.debug(f"[THREAD_CONV] Added exchange {exchange_id[:8]} to thread {thread_id}")
        return exchange_id

    def add_response(self, thread_id: str, response_message: str, generation_time: float) -> None:
        """Add a response to the most recent exchange."""
        conv_key = self._conv_key(thread_id)
        raw = self.store.lindex(conv_key, -1)
        if not raw:
            logging.warning(f"[THREAD_CONV] No exchange found in thread {thread_id}")
            return

        exchange = json.loads(raw)
        timestamp = utc_now().strftime('%Y-%m-%d %H:%M')
        exchange["response"] = {
            "message": response_message,
            "time": timestamp,
            "generation_time": generation_time,
        }

        self.store.lset(conv_key, -1, json.dumps(exchange))
        self._refresh_ttl(thread_id)

        # Write-through to SQLite
        exchange_id = exchange.get("id") or exchange.get("prompt", {}).get("id")
        if exchange_id:
            self._persist_response(thread_id, exchange_id,
                                   response_message=response_message,
                                   generation_time=generation_time)

    def add_response_error(self, thread_id: str, error_message: str) -> None:
        """Record an error for the most recent exchange."""
        conv_key = self._conv_key(thread_id)
        raw = self.store.lindex(conv_key, -1)
        if not raw:
            return

        exchange = json.loads(raw)
        timestamp = utc_now().strftime('%Y-%m-%d %H:%M')
        exchange["response"] = {
            "error": error_message,
            "time": timestamp,
        }

        self.store.lset(conv_key, -1, json.dumps(exchange))
        self._refresh_ttl(thread_id)

        # Write-through to SQLite
        exchange_id = exchange.get("id") or exchange.get("prompt", {}).get("id")
        if exchange_id:
            self._persist_response(thread_id, exchange_id,
                                   response_error=error_message)

    def add_steps_to_exchange(self, thread_id: str, next_actions: list) -> None:
        """Add steps to the most recent exchange."""
        conv_key = self._conv_key(thread_id)
        raw = self.store.lindex(conv_key, -1)
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
        self.store.lset(conv_key, -1, json.dumps(exchange))
        self._refresh_ttl(thread_id)

        # Write-through to SQLite
        exchange_id = exchange.get("id") or exchange.get("prompt", {}).get("id")
        if exchange_id:
            self._persist_json_field(exchange_id, "steps", steps)

    def add_memory_chunk(self, thread_id: str, exchange_id: str, memory_chunk: dict) -> bool:
        """Add or merge a memory chunk to a specific exchange by ID.

        With per-message encoding, this may be called twice per exchange:
        once for the user message (Phase A) and once for the assistant response
        (Phase D). Gists are merged so both survive in conversation history.

        Returns:
            True if the exchange was found and updated, False if not found.
        """
        conv_key = self._conv_key(thread_id)
        all_raw = self.store.lrange(conv_key, 0, -1)

        for i, raw in enumerate(all_raw):
            exchange = json.loads(raw)
            if exchange.get("id") == exchange_id or exchange.get("prompt", {}).get("id") == exchange_id:
                existing = exchange.get("memory_chunk") or {}
                if existing.get("gists"):
                    # Merge gists from both encodes (user message + assistant response)
                    merged_gists = existing["gists"] + memory_chunk.get("gists", [])
                    memory_chunk = {**memory_chunk, "gists": merged_gists}
                exchange["memory_chunk"] = memory_chunk
                self.store.lset(conv_key, i, json.dumps(exchange))
                self._refresh_ttl(thread_id)

                # Write-through to SQLite
                self._persist_json_field(exchange_id, "memory_chunk", memory_chunk)
                return True

        logging.warning(f"[THREAD_CONV] Exchange {exchange_id[:8]} not found in thread {thread_id}")
        return False

    def get_conversation_history(self, thread_id: str) -> list:
        """Get all exchanges for a thread."""
        conv_key = self._conv_key(thread_id)
        all_raw = self.store.lrange(conv_key, 0, -1)

        if not all_raw:
            # MemoryStore empty — try SQLite fallback
            return self._load_from_sqlite(thread_id)

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
        raw = self.store.lindex(conv_key, -1)
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
        count = self.store.get(self._index_key(thread_id))
        return int(count) if count else 0

    def get_paginated_history(self, thread_id: str, limit: int = 12, offset: int = 0) -> dict:
        """Get a paginated slice of conversation history for a thread.

        Args:
            thread_id: Thread identifier.
            limit: Number of exchanges to return.
            offset: Number of exchanges to skip from the END (0 = most recent).

        Returns:
            Dict with keys: exchanges (chronological slice), total, has_more.
        """
        conv_key = self._conv_key(thread_id)
        total = self.store.llen(conv_key)

        if total == 0:
            # MemoryStore empty — try SQLite fallback
            loaded = self._load_from_sqlite(thread_id)
            if not loaded:
                return {"exchanges": [], "total": 0, "has_more": False}
            total = self.store.llen(conv_key)

        # Compute lrange indices using negative indexing from end.
        # offset=0, limit=12  → lrange(key, -12, -1)
        # offset=12, limit=12 → lrange(key, -24, -13)
        end_idx = -(offset + 1) if offset > 0 else -1
        start_idx = -(offset + limit)

        raw = self.store.lrange(conv_key, start_idx, end_idx)
        exchanges = [json.loads(item) if isinstance(item, str) else item for item in raw]
        has_more = (offset + limit) < total
        return {"exchanges": exchanges, "total": total, "has_more": has_more}

    def get_most_recent_expired_thread_id(self) -> Optional[str]:
        """Return the thread_id of the most recently expired thread from SQLite."""
        try:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT thread_id FROM threads
                    WHERE state = 'expired'
                    ORDER BY expired_at DESC
                    LIMIT 1
                """)
                row = cursor.fetchone()
                cursor.close()
                return row[0] if row else None
        except Exception:
            return None

    def remove_exchanges(self, thread_id: str, exchange_ids: list) -> None:
        """Remove specific exchanges by ID (for post-episodic cleanup)."""
        conv_key = self._conv_key(thread_id)
        all_raw = self.store.lrange(conv_key, 0, -1)

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
            pipe = self.store.pipeline()
            pipe.delete(conv_key)
            for item in kept:
                pipe.rpush(conv_key, item)
            pipe.expire(conv_key, self.TTL_SECONDS)
            pipe.execute()
            logging.info(f"[THREAD_CONV] Removed {removed_count} exchanges from thread {thread_id}")
