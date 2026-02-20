import time
import logging
from typing import Dict, Tuple, Optional

from .redis_client import RedisClientService


logger = logging.getLogger(__name__)


class SemanticConsolidationTracker:
    """
    Tracker for intelligent semantic consolidation triggering.

    Implements two triggers:
    1. Episode count: Every 20 episodes
    2. High salience: 40% above rolling average (with 5-minute cooldown)
    """

    def __init__(self):
        """Initialize tracker with Redis connection."""
        self.redis = RedisClientService.create_connection()
        self.state_key = "semantic_consolidation_state"
        self.salience_history_key = "semantic_consolidation_salience_history"

        # Initialize state if not exists
        if not self.redis.exists(self.state_key):
            self._initialize_state()

    def _initialize_state(self) -> None:
        """Initialize consolidation state in Redis."""
        self.redis.hset(self.state_key, mapping={
            'episodes_since_last': 0,
            'last_consolidation_time': time.time()
        })
        logger.info("[CONSOLIDATION TRACKER] Initialized state")

    def get_state(self) -> Dict[str, float]:
        """
        Get current consolidation state.

        Returns:
            dict: {episodes_since_last, last_consolidation_time, average_salience}
        """
        state = self.redis.hgetall(self.state_key)

        episodes_since_last = int(state.get('episodes_since_last', 0))
        last_consolidation_time = float(state.get('last_consolidation_time', 0))
        average_salience = self._calculate_average_salience()

        return {
            'episodes_since_last': episodes_since_last,
            'last_consolidation_time': last_consolidation_time,
            'average_salience': average_salience
        }

    def increment_episode_count(self) -> None:
        """Increment episodes_since_last counter."""
        self.redis.hincrby(self.state_key, 'episodes_since_last', 1)

    def record_episode_salience(self, salience: float) -> None:
        """
        Record episode salience and trim history to 100 entries.

        Args:
            salience: Salience score of the episode
        """
        # Add to history
        self.redis.rpush(self.salience_history_key, salience)

        # Trim to 100 entries (FIFO)
        history_length = self.redis.llen(self.salience_history_key)
        if history_length > 100:
            trim_count = history_length - 100
            for _ in range(trim_count):
                self.redis.lpop(self.salience_history_key)

    def reset_episode_count(self) -> None:
        """Reset episode counter and update last consolidation timestamp."""
        self.redis.hset(self.state_key, mapping={
            'episodes_since_last': 0,
            'last_consolidation_time': time.time()
        })
        logger.info("[CONSOLIDATION TRACKER] Reset episode count")

    def _calculate_average_salience(self) -> float:
        """
        Calculate rolling average salience from history.

        Returns:
            float: Average salience score
        """
        history = self.redis.lrange(self.salience_history_key, 0, -1)

        if not history:
            return 5.0  # Default average if no history

        salience_values = [float(s) for s in history]
        return sum(salience_values) / len(salience_values)

    def should_trigger_consolidation(self, episode_salience: float) -> Tuple[bool, Optional[str]]:
        """
        Check if consolidation should be triggered.

        Args:
            episode_salience: Salience score of the current episode

        Returns:
            tuple: (should_trigger: bool, reason: str or None)
        """
        state = self.get_state()

        # Trigger 1: Episode count (every 20 episodes)
        if state['episodes_since_last'] >= 20:
            logger.info(
                f"[CONSOLIDATION TRACKER] Trigger: episode_count "
                f"(count={state['episodes_since_last']})"
            )
            return True, 'episode_count'

        # Trigger 2: High salience (40% above average + 5-minute cooldown)
        average_salience = state['average_salience']
        salience_threshold = average_salience * 1.4
        time_since_last = time.time() - state['last_consolidation_time']

        if episode_salience >= salience_threshold and time_since_last >= 300:
            logger.info(
                f"[CONSOLIDATION TRACKER] Trigger: high_salience "
                f"(salience={episode_salience:.2f}, threshold={salience_threshold:.2f}, "
                f"avg={average_salience:.2f})"
            )
            return True, 'high_salience'

        # No trigger
        logger.debug(
            f"[CONSOLIDATION TRACKER] No trigger "
            f"(count={state['episodes_since_last']}, "
            f"salience={episode_salience:.2f}, threshold={salience_threshold:.2f})"
        )
        return False, None
