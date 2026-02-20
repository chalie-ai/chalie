"""
Recent Topic Cache Service

Maintains the most recently active topic in Redis for recency bias in classification.
This simulates working memory / topic activation in the neurological model.

Supports per-user-channel scoping via constructor parameters.
"""

from services.redis_client import RedisClientService


class RecentTopicService:
    """Manages the most recent topic cache in Redis."""

    def __init__(self, ttl_minutes: int = 30, user_id: str = None, channel_id: str = None):
        """
        Initialize recent topic service.

        Args:
            ttl_minutes: Time-to-live for recent topic cache (default 30 minutes)
            user_id: Optional user ID for per-user scoping
            channel_id: Optional channel ID for per-channel scoping
        """
        self.redis = RedisClientService.create_connection()
        self.ttl_seconds = ttl_minutes * 60

        if user_id and channel_id:
            self.key = f"recent_topic:{user_id}:{channel_id}"
        else:
            self.key = "recent_topic"

    def set_recent_topic(self, topic: str) -> None:
        """
        Store the most recent topic with TTL.

        Args:
            topic: Topic name to cache
        """
        self.redis.setex(self.key, self.ttl_seconds, topic)

    def get_recent_topic(self) -> str:
        """
        Retrieve the most recent topic from cache.

        Returns:
            str: Most recent topic name, or empty string if cache is empty
        """
        topic = self.redis.get(self.key)
        return topic or ""

    def clear_recent_topic(self) -> None:
        """Clear the recent topic cache."""
        self.redis.delete(self.key)
