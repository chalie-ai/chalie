"""
Spark State Service — Redis-backed phase state machine for first-contact rapport.

Tracks the relationship phase between Chalie and a new user, from first contact
through graduated (when normal systems take over). Uses effective exchanges
(weighted scoring) to prevent spammy messages from accelerating progression.

Redis key: spark_state:{user_id}
TTL: 30 days, refreshed on every write.

Phases: first_contact → surface → exploratory → connected → graduated
"""

import json
import logging
import time
from typing import Optional

from services.redis_client import RedisClientService

logger = logging.getLogger(__name__)

LOG_PREFIX = "[SPARK STATE]"

# Phase transition table: (from_phase, to_phase) → (threshold, hold_required)
_TRANSITIONS = {
    ('first_contact', 'surface'): {
        'requires_welcome': True,
        'min_exchanges': 1,
        'effective_threshold': None,
        'hold_required': 0,
    },
    ('surface', 'exploratory'): {
        'requires_welcome': False,
        'min_exchanges': None,
        'effective_threshold': 4.0,
        'hold_required': 2,
    },
    ('exploratory', 'connected'): {
        'requires_welcome': False,
        'min_exchanges': None,
        'effective_threshold': 12.0,
        'hold_required': 3,
        'min_traits': 3,
    },
    ('connected', 'graduated'): {
        'requires_welcome': False,
        'min_exchanges': None,
        'effective_threshold': 25.0,
        'hold_required': 5,
        'min_traits': 5,
    },
}

_PHASE_ORDER = ['first_contact', 'surface', 'exploratory', 'connected', 'graduated']

# Rate limit: max effective_exchanges growth per scoring event
_MIN_SCORE_INTERVAL = 30.0  # seconds between score accumulations
_MAX_TOPICS = 50  # max topics tracked


class SparkStateService:
    """Redis-backed phase state machine for Spark rapport system."""

    _REDIS_KEY_PREFIX = "spark_state"
    REDIS_TTL = 2592000  # 30 days

    def __init__(self, user_id: str = 'primary'):
        self._user_id = user_id
        self._redis_key = f"{self._REDIS_KEY_PREFIX}:{user_id}"

    def _default_state(self) -> dict:
        return {
            'version': 1,
            'phase': 'first_contact',
            'exchange_count': 0,
            'effective_exchanges': 0.0,
            'traits_learned': 0,
            'welcome_sent': False,
            'welcome_sent_at': None,
            'phase_entered_at': time.time(),
            'phase_hold_count': 0,
            'last_active_at': None,
            'last_scored_at': None,
            'first_suggestion_seeded': False,
            'topics_discussed': [],
        }

    def get_state(self) -> dict:
        """
        Get the current spark state from Redis.

        Falls back to graduated if state is missing but user has 5+ traits.
        Returns default first_contact state for new users.
        """
        try:
            r = RedisClientService.create_connection()
            raw = r.get(self._redis_key)
            if raw:
                return json.loads(raw)

            # Fallback: check if user already has established traits
            if self._has_established_traits(5):
                state = self._default_state()
                state['phase'] = 'graduated'
                state['welcome_sent'] = True
                self._save_state(state)
                logger.info(f"{LOG_PREFIX} Initialized as graduated (5+ existing traits)")
                return state

            return self._default_state()
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} get_state failed (non-fatal): {e}")
            return self._default_state()

    def _save_state(self, state: dict) -> bool:
        """Save state to Redis with TTL refresh."""
        try:
            r = RedisClientService.create_connection()
            r.setex(self._redis_key, self.REDIS_TTL, json.dumps(state))
            return True
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} _save_state failed (non-fatal): {e}")
            return False

    def get_phase(self) -> str:
        """Get current phase name."""
        return self.get_state().get('phase', 'first_contact')

    def is_graduated(self) -> bool:
        """Check if spark system has completed its job."""
        return self.get_phase() == 'graduated'

    def needs_welcome(self) -> bool:
        """Check if welcome message still needs to be sent."""
        state = self.get_state()
        return not state.get('welcome_sent', False)

    def mark_welcome_sent(self) -> bool:
        """Mark welcome as sent and transition to surface if first exchange exists."""
        try:
            state = self.get_state()
            state['welcome_sent'] = True
            state['welcome_sent_at'] = time.time()
            self._save_state(state)
            logger.info(f"{LOG_PREFIX} Welcome marked as sent")
            return True
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} mark_welcome_sent failed: {e}")
            return False

    def increment_exchange(
        self,
        message: str,
        response_gap_seconds: float = 5.0,
        source: str = 'text',
    ) -> dict:
        """
        Record a new exchange and potentially transition phases.

        Applies idle decay, scores the exchange, checks transitions.

        Args:
            message: The user's message text
            response_gap_seconds: Time since last message (for engagement scoring)
            source: Message source type ('text', 'voice', 'system', 'quick_reply')

        Returns:
            dict: Updated state
        """
        try:
            state = self.get_state()

            if state['phase'] == 'graduated':
                return state

            now = time.time()

            # Apply idle decay before adding new score
            state = self._apply_idle_decay(state, now)

            # Rate limiter: skip scoring if too recent
            last_scored = state.get('last_scored_at')
            if last_scored and (now - last_scored) < _MIN_SCORE_INTERVAL:
                # Still increment raw count
                state['exchange_count'] = state.get('exchange_count', 0) + 1
                state['last_active_at'] = now
                self._save_state(state)
                return state

            # Score the exchange
            score = self._score_exchange(message, response_gap_seconds, source)

            state['exchange_count'] = state.get('exchange_count', 0) + 1
            state['effective_exchanges'] = state.get('effective_exchanges', 0.0) + score
            state['last_active_at'] = now
            state['last_scored_at'] = now

            # Update traits count
            state['traits_learned'] = self._count_user_traits()

            # Check phase transition
            state = self._check_transition(state)

            self._save_state(state)
            return state
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} increment_exchange failed: {e}")
            return self.get_state()

    def record_topic(self, topic: str) -> bool:
        """Record a discussed topic."""
        try:
            state = self.get_state()
            topics = state.get('topics_discussed', [])
            if topic and topic not in topics:
                topics.append(topic)
                if len(topics) > _MAX_TOPICS:
                    topics = topics[-_MAX_TOPICS:]
                state['topics_discussed'] = topics
                self._save_state(state)
            return True
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} record_topic failed: {e}")
            return False

    def mark_suggestion_seeded(self) -> bool:
        """Mark that the first soft capability seed has been sent."""
        try:
            state = self.get_state()
            state['first_suggestion_seeded'] = True
            self._save_state(state)
            return True
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} mark_suggestion_seeded failed: {e}")
            return False

    # ── Scoring ───────────────────────────────────────────────────

    @staticmethod
    def _score_exchange(
        message: str,
        response_gap_seconds: float,
        source: str = 'text',
    ) -> float:
        """Score a single exchange for meaningful engagement."""
        if source in ('system', 'quick_reply'):
            return 0.1

        score = 0.0
        word_count = len(message.split())

        if word_count >= 3:
            score += 0.3
        if word_count >= 10:
            score += 0.3
        if word_count >= 25:
            score += 0.2
        if word_count > 120:
            score += 0.1

        # Engagement signal: user took time (>3s gap = intentional)
        if response_gap_seconds > 3.0:
            score += 0.2

        return min(score, 1.0)

    @staticmethod
    def _apply_idle_decay(state: dict, now: float) -> dict:
        """Apply time-based decay to effective_exchanges."""
        last_active = state.get('last_active_at')
        if not last_active:
            return state

        idle_days = (now - last_active) / 86400
        if idle_days > 0:
            decay = 0.95 ** idle_days
            state['effective_exchanges'] = state.get('effective_exchanges', 0.0) * decay

        return state

    # ── Phase transitions ─────────────────────────────────────────

    def _check_transition(self, state: dict) -> dict:
        """Check if the current phase should transition to the next."""
        current_phase = state.get('phase', 'first_contact')

        if current_phase == 'graduated':
            return state

        idx = _PHASE_ORDER.index(current_phase)
        if idx >= len(_PHASE_ORDER) - 1:
            return state

        next_phase = _PHASE_ORDER[idx + 1]
        transition_key = (current_phase, next_phase)
        rule = _TRANSITIONS.get(transition_key)

        if not rule:
            return state

        # Check all conditions
        qualifies = True

        if rule.get('requires_welcome') and not state.get('welcome_sent'):
            qualifies = False

        if rule.get('min_exchanges') is not None:
            if state.get('exchange_count', 0) < rule['min_exchanges']:
                qualifies = False

        if rule.get('effective_threshold') is not None:
            if state.get('effective_exchanges', 0.0) < rule['effective_threshold']:
                qualifies = False

        if rule.get('min_traits') is not None:
            if state.get('traits_learned', 0) < rule['min_traits']:
                qualifies = False

        # Hysteresis: hold count must meet requirement
        hold_required = rule.get('hold_required', 0)

        if qualifies:
            state['phase_hold_count'] = state.get('phase_hold_count', 0) + 1

            if state['phase_hold_count'] >= hold_required:
                old_phase = current_phase
                state['phase'] = next_phase
                state['phase_entered_at'] = time.time()
                state['phase_hold_count'] = 0

                logger.info(
                    f"{LOG_PREFIX} Phase transition: {old_phase} → {next_phase} "
                    f"(effective={state.get('effective_exchanges', 0):.1f}, "
                    f"traits={state.get('traits_learned', 0)})"
                )

                # Log phase change event
                self._log_phase_change(old_phase, next_phase, state)
        else:
            # Reset hold count if conditions are no longer met
            state['phase_hold_count'] = 0

        return state

    # ── Helpers ────────────────────────────────────────────────────

    def _has_established_traits(self, min_count: int) -> bool:
        """Check if user has enough traits in the database."""
        try:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT COUNT(*) FROM user_traits WHERE confidence >= 0.5"
                )
                count = cursor.fetchone()[0]
                cursor.close()
                return count >= min_count
        except Exception:
            return False

    def _count_user_traits(self) -> int:
        """Count user traits with reasonable confidence."""
        try:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT COUNT(*) FROM user_traits WHERE confidence >= 0.5"
                )
                count = cursor.fetchone()[0]
                cursor.close()
                return count
        except Exception:
            return 0

    def _log_phase_change(self, from_phase: str, to_phase: str, state: dict) -> None:
        """Log phase change to interaction_log."""
        try:
            from services.database_service import get_shared_db_service
            from services.interaction_log_service import InteractionLogService

            db = get_shared_db_service()
            log_service = InteractionLogService(db)
            log_service.log_event(
                event_type='spark_phase_change',
                payload={
                    'from_phase': from_phase,
                    'to_phase': to_phase,
                    'effective_exchanges': state.get('effective_exchanges', 0),
                    'traits_learned': state.get('traits_learned', 0),
                },
                topic='spark',
                source='spark_state',
            )
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Phase change logging failed: {e}")
