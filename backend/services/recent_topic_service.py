"""
Recent Topic Cache Service

Maintains the most recently active topic in MemoryStore for recency bias in classification.
This simulates working memory / topic activation in the neurological model.

Supports per-channel scoping via constructor parameters.
"""

from services.memory_client import MemoryClientService


class RecentTopicService:
    """Manages the most recent topic cache in MemoryStore."""

    def __init__(self, ttl_minutes: int = 30, channel_id: str = None):
        """
        Initialize recent topic service.

        Args:
            ttl_minutes: Time-to-live for recent topic cache (default 30 minutes)
            channel_id: Optional channel ID for per-channel scoping
        """
        self.store = MemoryClientService.create_connection()
        self.ttl_seconds = ttl_minutes * 60
        self.key = f"recent_topic:{channel_id}" if channel_id else "recent_topic"

    def set_recent_topic(self, topic: str) -> None:
        """
        Store the most recent topic with TTL.

        Args:
            topic: Topic name to cache
        """
        self.store.setex(self.key, self.ttl_seconds, topic)

    def get_recent_topic(self) -> str:
        """
        Retrieve the most recent topic from cache.

        Returns:
            str: Most recent topic name, or empty string if cache is empty
        """
        topic = self.store.get(self.key)
        return topic or ""

    def clear_recent_topic(self) -> None:
        """Clear the recent topic cache."""
        self.store.delete(self.key)
