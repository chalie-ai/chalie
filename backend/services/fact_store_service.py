"""
Fact Store Service - Key-value structured short-term memory.

Redis-based storage of atomic facts extracted from exchanges.
Follows GistStorageService pattern (Redis sorted sets, TTL, confidence filtering).
"""

import json
import time
import logging
from typing import List, Dict, Optional, Any
from services.redis_client import RedisClientService


class FactStoreService:
    """Manages storage and retrieval of atomic facts in Redis."""

    def __init__(self, ttl_minutes: int = 1440, max_facts_per_topic: int = 50):
        """
        Initialize fact store service.

        Args:
            ttl_minutes: TTL for facts in minutes (default 1440 = 24 hours)
            max_facts_per_topic: Maximum facts per topic (default 50)
        """
        self.redis = RedisClientService.create_connection()
        self.ttl_seconds = ttl_minutes * 60
        self.max_facts_per_topic = max_facts_per_topic

    def _get_fact_key(self, topic: str, key: str) -> str:
        """Generate Redis key for a fact."""
        return f"fact:{topic}:{key}"

    def _get_fact_index_key(self, topic: str) -> str:
        """Generate Redis key for the sorted set index of facts."""
        return f"fact_index:{topic}"

    def store_fact(
        self,
        topic: str,
        key: str,
        value: Any,
        confidence: float = 0.5,
        source: str = None
    ) -> bool:
        """
        Store or update a fact with conflict resolution.

        Args:
            topic: Topic name
            key: Fact key (e.g., "name", "preferred_language")
            value: Fact value
            confidence: Confidence score (0.0-1.0)
            source: Source of the fact (e.g., exchange_id)

        Returns:
            True if stored, False if existing fact had higher confidence
        """
        fact_redis_key = self._get_fact_key(topic, key)
        current_time = time.time()

        # Check for existing fact (conflict resolution)
        existing_json = self.redis.get(fact_redis_key)
        if existing_json:
            existing = json.loads(existing_json)
            if not self._resolve_conflict(existing, confidence, current_time):
                logging.debug(
                    f"[FACT STORE] Kept existing fact '{key}' "
                    f"(existing confidence: {existing.get('confidence')}, new: {confidence})"
                )
                return False

        # Store fact data
        fact_data = {
            'key': key,
            'value': value,
            'confidence': confidence,
            'source': source,
            'created_at': current_time,
            'updated_at': current_time
        }

        self.redis.setex(
            fact_redis_key,
            self.ttl_seconds,
            json.dumps(fact_data)
        )

        # Add to sorted set index (score = timestamp for ordering)
        index_key = self._get_fact_index_key(topic)
        self.redis.zadd(index_key, {key: current_time})
        self.redis.expire(index_key, self.ttl_seconds)

        # Trim if over max
        fact_count = self.redis.zcard(index_key)
        if fact_count > self.max_facts_per_topic:
            # Remove oldest facts
            excess = fact_count - self.max_facts_per_topic
            oldest_keys = self.redis.zrange(index_key, 0, excess - 1)
            for old_key in oldest_keys:
                self.redis.delete(self._get_fact_key(topic, old_key))
                self.redis.zrem(index_key, old_key)

        logging.info(f"[FACT STORE] Stored fact '{key}' = '{value}' for topic '{topic}' (confidence: {confidence})")
        return True

    def _resolve_conflict(self, existing: dict, new_confidence: float, new_time: float) -> bool:
        """
        Resolve conflict between existing and new fact.
        Highest confidence wins; recency breaks ties.

        Args:
            existing: Existing fact dict
            new_confidence: Confidence of new fact
            new_time: Timestamp of new fact

        Returns:
            True if new fact should replace existing
        """
        existing_confidence = existing.get('confidence', 0.0)

        if new_confidence > existing_confidence:
            return True
        elif new_confidence == existing_confidence:
            # Recency tiebreak
            return new_time > existing.get('updated_at', 0)
        return False

    def get_fact(self, topic: str, key: str) -> Optional[Dict]:
        """
        Retrieve a single fact by key.

        Args:
            topic: Topic name
            key: Fact key

        Returns:
            Fact dict or None
        """
        fact_redis_key = self._get_fact_key(topic, key)
        fact_json = self.redis.get(fact_redis_key)

        if fact_json:
            return json.loads(fact_json)
        return None

    def get_all_facts(self, topic: str) -> List[Dict]:
        """
        Retrieve all facts for a topic.

        Args:
            topic: Topic name

        Returns:
            List of fact dicts ordered by most recent first
        """
        index_key = self._get_fact_index_key(topic)
        fact_keys = self.redis.zrevrange(index_key, 0, -1)

        if not fact_keys:
            return []

        facts = []
        for key in fact_keys:
            fact_redis_key = self._get_fact_key(topic, key)
            fact_json = self.redis.get(fact_redis_key)

            if fact_json:
                facts.append(json.loads(fact_json))
            else:
                # Clean up stale entry
                self.redis.zrem(index_key, key)

        return facts

    def get_facts_formatted(self, topic: str) -> str:
        """
        Get all facts formatted for prompt injection.

        Args:
            topic: Topic name

        Returns:
            Formatted facts string or empty string
        """
        facts = self.get_all_facts(topic)
        if not facts:
            return ""

        lines = ["## Known Facts"]
        for fact in facts:
            key = fact.get('key', 'unknown')
            value = fact.get('value', '')
            confidence = fact.get('confidence', 0.0)
            lines.append(f"- {key}: {value} (confidence: {confidence:.1f})")

        return "\n".join(lines)

    def clear_facts(self, topic: str):
        """
        Clear all facts for a topic.

        Args:
            topic: Topic name
        """
        index_key = self._get_fact_index_key(topic)
        fact_keys = self.redis.zrange(index_key, 0, -1)

        for key in fact_keys:
            self.redis.delete(self._get_fact_key(topic, key))

        self.redis.delete(index_key)
