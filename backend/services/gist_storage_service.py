import json
import logging
import time
import uuid
from collections import defaultdict
from typing import List, Dict, Optional, Tuple
from services.redis_client import RedisClientService


class GistStorageService:
    """
    Manages storage and retrieval of gists in Redis with TTL and confidence filtering.
    """

    def __init__(self, attention_span_minutes: int = 30, min_confidence: int = 7, max_gists: int = 8,
                 similarity_threshold: float = 0.7, max_per_type: int = 2):
        """
        Initialize gist storage service.

        Args:
            attention_span_minutes: TTL for gists in minutes (default 30)
            min_confidence: Minimum confidence threshold for storing gists (default 7)
            max_gists: Maximum number of gists to retrieve (default 8)
            similarity_threshold: Jaccard similarity threshold for deduplication (default 0.7)
            max_per_type: Maximum gists per type to retain (default 2)
        """
        self.redis = RedisClientService.create_connection()
        self.attention_span_seconds = attention_span_minutes * 60
        self.min_confidence = min_confidence
        self.max_gists = max_gists
        self.similarity_threshold = similarity_threshold
        self.max_per_type = max_per_type

    def _get_gist_key(self, topic: str, gist_id: str) -> str:
        """Generate Redis key for a gist."""
        return f"gist:{topic}:{gist_id}"

    def _get_gist_index_key(self, topic: str) -> str:
        """Generate Redis key for the sorted set index of gists."""
        return f"gist_index:{topic}"

    def _get_last_message_key(self, topic: str) -> str:
        """Generate Redis key for last message fallback."""
        return f"last_message:{topic}"

    def store_gists(self, topic: str, gists: List[Dict], prompt: str, response: str) -> int:
        """
        Store gists in Redis with TTL, confidence filtering, and deduplication.

        Args:
            topic: Topic name
            gists: List of gist dicts with 'content', 'type', 'confidence'
            prompt: Original prompt (for last_message fallback)
            response: Response message (for last_message fallback)

        Returns:
            int: Number of gists stored
        """
        stored_count = 0
        current_time = time.time()

        # Check if we have any gists at all
        has_existing_gists = self.redis.exists(self._get_gist_index_key(topic))

        # Load existing gists once for dedup comparisons
        existing_gists = self._get_all_gists_with_ids(topic)

        for gist in gists:
            try:
                confidence = int(gist.get('confidence', 0))
            except (ValueError, TypeError):
                confidence = 0
            content = gist.get('content', '')

            # Only store gists >= min_confidence, unless Redis is empty
            if not (confidence >= self.min_confidence or not has_existing_gists):
                logging.debug(f"[gist_storage] Rejected gist (confidence {confidence} < {self.min_confidence}): {content[:60]}")
                continue

            # Check for duplicate against existing + already-stored-this-batch
            duplicate = self._find_duplicate(gist, existing_gists, self.similarity_threshold)

            if duplicate:
                old_id, old_data = duplicate
                old_confidence = old_data.get('confidence', 0)

                if confidence > old_confidence:
                    # Replace with higher-confidence version
                    self._replace_gist(topic, old_id, {
                        'content': content,
                        'type': gist.get('type', 'unknown'),
                        'confidence': confidence,
                        'created_at': current_time
                    })
                    # Update in-memory list for intra-batch dedup
                    for i, (eid, edata) in enumerate(existing_gists):
                        if eid == old_id:
                            existing_gists[i] = (old_id, {
                                'content': content,
                                'type': gist.get('type', 'unknown'),
                                'confidence': confidence,
                                'created_at': current_time
                            })
                            break
                    stored_count += 1
                    logging.info(f"[gist_storage] Replaced duplicate gist (confidence {old_confidence} → {confidence}): {content[:60]}")
                else:
                    logging.info(f"[gist_storage] Skipped duplicate gist (existing confidence {old_confidence} >= {confidence}): {content[:60]}")
                continue

            # No duplicate — store new gist
            gist_id = str(uuid.uuid4())
            gist_key = self._get_gist_key(topic, gist_id)

            gist_data = {
                'content': content,
                'type': gist.get('type', 'unknown'),
                'confidence': confidence,
                'created_at': current_time
            }

            self.redis.setex(
                gist_key,
                self.attention_span_seconds,
                json.dumps(gist_data)
            )

            index_key = self._get_gist_index_key(topic)
            self.redis.zadd(index_key, {gist_id: current_time})
            self.redis.expire(index_key, self.attention_span_seconds)

            # Track for intra-batch dedup
            existing_gists.append((gist_id, gist_data))
            stored_count += 1

        # Enforce per-type caps after all inserts
        if existing_gists:
            self._enforce_type_caps(topic, existing_gists, self.max_per_type)

        # Always update last message fallback
        last_message_data = {
            'prompt': prompt,
            'response': response,
            'timestamp': current_time
        }
        self.redis.setex(
            self._get_last_message_key(topic),
            self.attention_span_seconds,
            json.dumps(last_message_data)
        )

        return stored_count

    def _get_all_gists_with_ids(self, topic: str) -> List[Tuple[str, Dict]]:
        """Load all gists from Redis with their IDs for dedup comparison."""
        index_key = self._get_gist_index_key(topic)
        gist_ids = self.redis.zrange(index_key, 0, -1)

        results = []
        for gist_id in gist_ids:
            gist_key = self._get_gist_key(topic, gist_id)
            gist_json = self.redis.get(gist_key)

            if gist_json:
                results.append((gist_id, json.loads(gist_json)))
            else:
                # Clean up stale index entry (TTL expired on the gist key)
                self.redis.zrem(index_key, gist_id)

        return results

    @staticmethod
    def _calculate_jaccard_similarity(text_a: str, text_b: str) -> float:
        """Word-level Jaccard similarity between two strings."""
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)

    def _find_duplicate(self, new_gist: Dict, existing_gists: List[Tuple[str, Dict]],
                        threshold: float) -> Optional[Tuple[str, Dict]]:
        """Find the best matching duplicate above threshold. Returns (id, data) or None."""
        new_content = new_gist.get('content', '')
        best_match = None
        best_similarity = 0.0

        for gist_id, gist_data in existing_gists:
            similarity = self._calculate_jaccard_similarity(new_content, gist_data.get('content', ''))
            if similarity >= threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match = (gist_id, gist_data)

        return best_match

    def _replace_gist(self, topic: str, old_id: str, new_data: Dict):
        """Replace an existing gist's data in-place (same ID, refreshed TTL)."""
        gist_key = self._get_gist_key(topic, old_id)
        self.redis.setex(gist_key, self.attention_span_seconds, json.dumps(new_data))
        # Update score in index to current time
        index_key = self._get_gist_index_key(topic)
        self.redis.zadd(index_key, {old_id: new_data['created_at']})
        self.redis.expire(index_key, self.attention_span_seconds)

    def _enforce_type_caps(self, topic: str, all_gists: List[Tuple[str, Dict]], max_per_type: int):
        """Remove lowest-confidence excess gists per type."""
        by_type = defaultdict(list)
        for gist_id, gist_data in all_gists:
            gist_type = gist_data.get('type', 'unknown')
            by_type[gist_type].append((gist_id, gist_data))

        index_key = self._get_gist_index_key(topic)
        for gist_type, gists in by_type.items():
            if len(gists) <= max_per_type:
                continue

            # Sort by confidence descending, keep top max_per_type
            sorted_gists = sorted(gists, key=lambda g: g[1].get('confidence', 0), reverse=True)
            to_remove = sorted_gists[max_per_type:]

            for gist_id, gist_data in to_remove:
                gist_key = self._get_gist_key(topic, gist_id)
                self.redis.delete(gist_key)
                self.redis.zrem(index_key, gist_id)
                logging.info(f"[gist_storage] Type cap: removed '{gist_type}' gist (confidence {gist_data.get('confidence', 0)}): {gist_data.get('content', '')[:60]}")

    def get_latest_gists(self, topic: str) -> List[Dict]:
        """
        Retrieve the latest gists for a topic, up to max_gists.

        Args:
            topic: Topic name

        Returns:
            List of gist dicts with 'content', 'type', 'confidence'
        """
        index_key = self._get_gist_index_key(topic)

        # Get latest gist IDs from sorted set (highest scores = most recent)
        gist_ids = self.redis.zrevrange(index_key, 0, self.max_gists - 1)

        if not gist_ids:
            return []

        gists = []
        for gist_id in gist_ids:
            gist_key = self._get_gist_key(topic, gist_id)
            gist_json = self.redis.get(gist_key)

            if gist_json:
                gist_data = json.loads(gist_json)
                gists.append({
                    'content': gist_data['content'],
                    'type': gist_data['type'],
                    'confidence': gist_data['confidence']
                })
                # Refresh TTL on read (touch-on-read)
                self.redis.expire(gist_key, self.attention_span_seconds)
            else:
                # Clean up stale entry from index
                self.redis.zrem(index_key, gist_id)

        # Refresh index TTL on read
        if gists:
            self.redis.expire(index_key, self.attention_span_seconds)

        return gists

    def get_last_message(self, topic: str) -> Optional[Dict]:
        """
        Get the last message as fallback when no gists are available.

        Args:
            topic: Topic name

        Returns:
            Dict with 'prompt' and 'response' or None
        """
        last_message_key = self._get_last_message_key(topic)
        last_message_json = self.redis.get(last_message_key)

        if last_message_json:
            # Refresh TTL on read (touch-on-read)
            self.redis.expire(last_message_key, self.attention_span_seconds)
            return json.loads(last_message_json)

        return None

    def has_gists(self, topic: str) -> bool:
        """
        Check if topic has any gists stored.

        Args:
            topic: Topic name

        Returns:
            bool: True if gists exist
        """
        index_key = self._get_gist_index_key(topic)
        return self.redis.exists(index_key) > 0

    def clear_gists(self, topic: str):
        """
        Clear all gists for a topic (useful for testing).

        Args:
            topic: Topic name
        """
        index_key = self._get_gist_index_key(topic)
        gist_ids = self.redis.zrange(index_key, 0, -1)

        # Delete all gist keys
        for gist_id in gist_ids:
            gist_key = self._get_gist_key(topic, gist_id)
            self.redis.delete(gist_key)

        # Delete index
        self.redis.delete(index_key)

        # Delete last message
        self.redis.delete(self._get_last_message_key(topic))

    # Cold-start booster gists — injected when a topic has zero gists
    COLD_START_GISTS = [
        {
            'content': 'I am a learning system beginning a new conversation. I have no prior context with this user and will build understanding through interaction.',
            'type': 'cold_start',
            'confidence': 5
        },
        {
            'content': 'I can form memories from our exchanges, track facts, recognize topics, and build understanding over time. My responses improve as I learn.',
            'type': 'cold_start',
            'confidence': 5
        }
    ]
    COLD_START_TTL = 3600  # 1 hour

    def store_cold_start_gists(self, topic: str) -> bool:
        """
        Inject temporary identity/capability gists for a topic with no existing gists.

        Guards on has_gists() — does nothing if the topic already has gists.
        Cold-start gists have type='cold_start' so they can be distinguished from
        real gists (e.g. excluded from warmth calculations).

        Args:
            topic: Topic name

        Returns:
            bool: True if gists were injected, False if topic already had gists
        """
        if self.has_gists(topic):
            return False

        current_time = time.time()
        index_key = self._get_gist_index_key(topic)

        for gist in self.COLD_START_GISTS:
            gist_id = str(uuid.uuid4())
            gist_key = self._get_gist_key(topic, gist_id)

            gist_data = {
                'content': gist['content'],
                'type': gist['type'],
                'confidence': gist['confidence'],
                'created_at': current_time
            }

            self.redis.setex(gist_key, self.COLD_START_TTL, json.dumps(gist_data))
            self.redis.zadd(index_key, {gist_id: current_time})

        self.redis.expire(index_key, self.COLD_START_TTL)

        logging.info(f"[gist_storage] Cold-start gists injected for topic '{topic}'")
        return True
