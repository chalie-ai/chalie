"""Unit tests for NurtureAction."""
import json
import time
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass

# Mark all tests as unit (no external dependencies)
pytestmark = pytest.mark.unit


@dataclass
class FakeThought:
    """Minimal ThoughtContext for testing."""
    thought_type: str = 'reflection'
    thought_content: str = 'Small patterns in daily routines are worth noticing.'
    activation_energy: float = 0.6
    seed_concept: str = 'routine'
    seed_topic: str = 'daily-routines'
    thought_embedding: list = None
    drift_gist_id: str = None
    drift_gist_ttl: int = 1800
    extra: dict = None

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}


@pytest.fixture
def mock_redis():
    """FakeRedis that stores values in a dict."""
    store = {}

    class FakeRedis:
        def get(self, key):
            return store.get(key)

        def set(self, key, value):
            store[key] = str(value)

        def setex(self, key, ttl, value):
            store[key] = str(value)

        def setnx(self, key, value):
            if key not in store:
                store[key] = str(value)
                return True
            return False

        def delete(self, key):
            store.pop(key, None)

        def expire(self, key, ttl):
            pass

        def incr(self, key):
            val = int(store.get(key, 0)) + 1
            store[key] = str(val)
            return val

    fake = FakeRedis()
    fake._store = store
    return fake


@pytest.fixture
def nurture(mock_redis):
    """Create NurtureAction with mocked Redis."""
    with patch(
        'services.autonomous_actions.nurture_action.RedisClientService'
    ) as mock_cls:
        mock_cls.create_connection.return_value = mock_redis
        from services.autonomous_actions.nurture_action import NurtureAction
        action = NurtureAction(config={'user_id': 'test'})
        action.redis = mock_redis
        yield action


@pytest.fixture
def thought():
    return FakeThought()


# ── Phase Gate ────────────────────────────────────────────────────────

class TestPhaseGate:
    def test_passes_surface(self, nurture):
        with patch(
            'services.spark_state_service.SparkStateService'
        ) as mock_cls:
            mock_cls.return_value.get_phase.return_value = 'surface'
            passes, phase = nurture._phase_gate()
            assert passes is True
            assert phase == 'surface'

    def test_passes_exploratory(self, nurture):
        with patch(
            'services.spark_state_service.SparkStateService'
        ) as mock_cls:
            mock_cls.return_value.get_phase.return_value = 'exploratory'
            passes, phase = nurture._phase_gate()
            assert passes is True
            assert phase == 'exploratory'

    def test_fails_first_contact(self, nurture):
        with patch(
            'services.spark_state_service.SparkStateService'
        ) as mock_cls:
            mock_cls.return_value.get_phase.return_value = 'first_contact'
            passes, _ = nurture._phase_gate()
            assert passes is False

    def test_fails_connected(self, nurture):
        with patch(
            'services.spark_state_service.SparkStateService'
        ) as mock_cls:
            mock_cls.return_value.get_phase.return_value = 'connected'
            passes, _ = nurture._phase_gate()
            assert passes is False

    def test_fails_graduated(self, nurture):
        with patch(
            'services.spark_state_service.SparkStateService'
        ) as mock_cls:
            mock_cls.return_value.get_phase.return_value = 'graduated'
            passes, _ = nurture._phase_gate()
            assert passes is False


# ── Timing Gate ───────────────────────────────────────────────────────

class TestTimingGate:
    def _set_interaction(self, redis, seconds_ago):
        redis.set('proactive:test:last_interaction_ts', str(time.time() - seconds_ago))

    def test_blocks_during_quiet_hours(self, nurture, mock_redis):
        self._set_interaction(mock_redis, 25000)
        with patch(
            'services.autonomous_actions.nurture_action.datetime'
        ) as mock_dt:
            mock_dt.now.return_value.hour = 2  # 2 AM = quiet
            passes, details = nurture._timing_gate('surface')
            assert passes is False
            assert details['rejected'] == 'quiet_hours'

    def test_blocks_no_interaction_history(self, nurture):
        passes, details = nurture._timing_gate('surface')
        assert passes is False
        assert details['rejected'] == 'no_interaction_history'

    def test_blocks_when_idle_below_surface_threshold(self, nurture, mock_redis):
        self._set_interaction(mock_redis, 3600)  # 1h ago, need 6h
        with patch(
            'services.autonomous_actions.nurture_action.datetime'
        ) as mock_dt:
            mock_dt.now.return_value.hour = 14  # 2 PM = not quiet
            passes, details = nurture._timing_gate('surface')
            assert passes is False
            assert details['rejected'] == 'too_soon'

    def test_surface_uses_6h_min_idle(self, nurture, mock_redis):
        self._set_interaction(mock_redis, 22000)  # ~6.1h ago
        with patch(
            'services.autonomous_actions.nurture_action.datetime'
        ) as mock_dt:
            mock_dt.now.return_value.hour = 14
            passes, _ = nurture._timing_gate('surface')
            assert passes is True

    def test_exploratory_uses_2h_min_idle(self, nurture, mock_redis):
        self._set_interaction(mock_redis, 7500)  # ~2.1h ago
        with patch(
            'services.autonomous_actions.nurture_action.datetime'
        ) as mock_dt:
            mock_dt.now.return_value.hour = 14
            passes, _ = nurture._timing_gate('exploratory')
            assert passes is True

    def test_blocks_within_daily_cooldown(self, nurture, mock_redis):
        self._set_interaction(mock_redis, 25000)  # 7h ago
        mock_redis.set('spark_nurture:test:last_sent_ts', str(time.time() - 3600))
        with patch(
            'services.autonomous_actions.nurture_action.datetime'
        ) as mock_dt:
            mock_dt.now.return_value.hour = 14
            passes, details = nurture._timing_gate('surface')
            assert passes is False
            assert details['rejected'] == 'daily_cooldown'

    def test_backoff_multiplies_daily_cooldown(self, nurture, mock_redis):
        self._set_interaction(mock_redis, 25000)
        # Sent 25h ago — normally would pass 24h cooldown
        mock_redis.set(
            'spark_nurture:test:last_sent_ts', str(time.time() - 90000)
        )
        # But backoff is 2x → effective cooldown is 48h
        mock_redis.set('spark_nurture:test:backoff_multiplier', '2')
        with patch(
            'services.autonomous_actions.nurture_action.datetime'
        ) as mock_dt:
            mock_dt.now.return_value.hour = 14
            passes, details = nurture._timing_gate('surface')
            assert passes is False
            assert details['rejected'] == 'daily_cooldown'
            assert details['backoff'] == 2


# ── Backoff Gate ──────────────────────────────────────────────────────

class TestBackoffGate:
    def test_passes_with_zero_unanswered(self, nurture):
        passes, details = nurture._backoff_gate()
        assert passes is True

    def test_passes_with_two_unanswered(self, nurture, mock_redis):
        mock_redis.set('spark_nurture:test:unanswered_count', '2')
        passes, details = nurture._backoff_gate()
        assert passes is True
        assert details['unanswered'] == 2

    def test_fails_at_three_unanswered(self, nurture, mock_redis):
        mock_redis.set('spark_nurture:test:unanswered_count', '3')
        passes, details = nurture._backoff_gate()
        assert passes is False
        assert details['rejected'] == 'max_unanswered'

    def test_pauses_at_max_unanswered(self, nurture, mock_redis):
        mock_redis.set('spark_nurture:test:unanswered_count', '3')
        nurture._backoff_gate()
        assert mock_redis.get('spark_nurture:test:paused') == '1'

    def test_fails_when_paused(self, nurture, mock_redis):
        mock_redis.set('spark_nurture:test:paused', '1')
        passes, details = nurture._backoff_gate()
        assert passes is False
        assert details['rejected'] == 'paused'


# ── Content Gate ──────────────────────────────────────────────────────

class TestContentGate:
    def test_passes_with_episodes(self, nurture):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (5,)
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        mock_db = MagicMock()
        mock_db.connection.return_value = mock_conn

        with patch(
            'services.database_service.get_shared_db_service',
            return_value=mock_db,
        ):
            passes, details = nurture._content_gate()
            assert passes is True
            assert details['episodes'] == 5

    def test_fails_with_no_episodes(self, nurture):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (0,)
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        mock_db = MagicMock()
        mock_db.connection.return_value = mock_conn

        with patch(
            'services.database_service.get_shared_db_service',
            return_value=mock_db,
        ):
            passes, details = nurture._content_gate()
            assert passes is False


# ── Scoring ───────────────────────────────────────────────────────────

class TestScoring:
    def test_returns_base_score_when_all_gates_pass(self, nurture, thought):
        nurture._phase_gate = MagicMock(return_value=(True, 'surface'))
        nurture._timing_gate = MagicMock(return_value=(True, {}))
        nurture._backoff_gate = MagicMock(return_value=(True, {}))
        nurture._content_gate = MagicMock(return_value=(True, {'episodes': 3}))

        score, eligible = nurture.should_execute(thought)
        assert eligible is True
        assert score == 0.35

    def test_ineligible_when_phase_gate_fails(self, nurture, thought):
        nurture._phase_gate = MagicMock(return_value=(False, 'connected'))

        score, eligible = nurture.should_execute(thought)
        assert eligible is False
        assert score == 0.0

    def test_ineligible_when_timing_gate_fails(self, nurture, thought):
        nurture._phase_gate = MagicMock(return_value=(True, 'surface'))
        nurture._timing_gate = MagicMock(
            return_value=(False, {'rejected': 'too_soon'})
        )

        score, eligible = nurture.should_execute(thought)
        assert eligible is False


# ── Execute ───────────────────────────────────────────────────────────

class TestExecute:
    def test_updates_tracking_on_success(self, nurture, mock_redis, thought):
        nurture._pending_phase = 'surface'
        nurture._gather_context = MagicMock(return_value={
            'phase': 'surface',
            'thought_content': 'test',
            'thought_type': 'reflection',
            'seed_topic': 'test',
            'topics_discussed': [],
            'recent_episode_gist': 'test gist',
            'days_since_welcome': 1.0,
        })
        nurture._generate_nurture_message = MagicMock(
            return_value='Some thoughts settle more clearly when the day slows down.'
        )
        nurture._deliver_nurture = MagicMock()
        nurture._log_nurture_event = MagicMock()

        result = nurture.execute(thought)

        assert result.success is True
        assert result.details['phase'] == 'surface'
        nurture._deliver_nurture.assert_called_once()

        # Tracking should be updated
        assert int(mock_redis.get('spark_nurture:test:unanswered_count')) == 1
        assert int(mock_redis.get('spark_nurture:test:total_sent')) == 1
        assert mock_redis.get('spark_nurture:test:last_sent_ts') is not None

    def test_doubles_backoff_on_send(self, nurture, mock_redis, thought):
        mock_redis.set('spark_nurture:test:backoff_multiplier', '1')
        nurture._pending_phase = 'exploratory'
        nurture._gather_context = MagicMock(return_value={
            'phase': 'exploratory', 'thought_content': 'test',
            'thought_type': 'reflection', 'seed_topic': 'test',
            'topics_discussed': [], 'recent_episode_gist': '',
            'days_since_welcome': 2.0,
        })
        nurture._generate_nurture_message = MagicMock(return_value='Test msg.')
        nurture._deliver_nurture = MagicMock()
        nurture._log_nurture_event = MagicMock()

        nurture.execute(thought)
        assert int(mock_redis.get('spark_nurture:test:backoff_multiplier')) == 2

    def test_caps_backoff_at_max(self, nurture, mock_redis, thought):
        mock_redis.set('spark_nurture:test:backoff_multiplier', '4')
        nurture._pending_phase = 'surface'
        nurture._gather_context = MagicMock(return_value={
            'phase': 'surface', 'thought_content': 'test',
            'thought_type': 'reflection', 'seed_topic': 'test',
            'topics_discussed': [], 'recent_episode_gist': '',
            'days_since_welcome': 3.0,
        })
        nurture._generate_nurture_message = MagicMock(return_value='Test msg.')
        nurture._deliver_nurture = MagicMock()
        nurture._log_nurture_event = MagicMock()

        result = nurture.execute(thought)
        # 4 * 2 = 8 but capped at 4
        assert int(mock_redis.get('spark_nurture:test:backoff_multiplier')) == 4
        assert result.details['backoff'] == 4

    def test_tracks_llm_failure(self, nurture, mock_redis, thought):
        nurture._pending_phase = 'surface'
        nurture._gather_context = MagicMock(return_value={
            'phase': 'surface', 'thought_content': 'test',
            'thought_type': 'reflection', 'seed_topic': 'test',
            'topics_discussed': [], 'recent_episode_gist': '',
            'days_since_welcome': 1.0,
        })
        nurture._generate_nurture_message = MagicMock(return_value=None)

        result = nurture.execute(thought)
        assert result.success is False
        assert result.details['reason'] == 'generation_failed'
        assert int(mock_redis.get('spark_nurture:test:llm_fail_count')) == 1


# ── User Activity Reset ──────────────────────────────────────────────

class TestUserActivityReset:
    def test_resets_unanswered_count(self, mock_redis):
        mock_redis.set('spark_nurture:default:unanswered_count', '3')
        with patch(
            'services.autonomous_actions.nurture_action.RedisClientService'
        ) as mock_cls:
            mock_cls.create_connection.return_value = mock_redis
            from services.autonomous_actions.nurture_action import NurtureAction
            NurtureAction.record_user_activity()
        assert mock_redis.get('spark_nurture:default:unanswered_count') == '0'

    def test_resets_backoff_multiplier(self, mock_redis):
        mock_redis.set('spark_nurture:default:backoff_multiplier', '4')
        with patch(
            'services.autonomous_actions.nurture_action.RedisClientService'
        ) as mock_cls:
            mock_cls.create_connection.return_value = mock_redis
            from services.autonomous_actions.nurture_action import NurtureAction
            NurtureAction.record_user_activity()
        assert mock_redis.get('spark_nurture:default:backoff_multiplier') == '1'

    def test_clears_pause(self, mock_redis):
        mock_redis.set('spark_nurture:default:paused', '1')
        with patch(
            'services.autonomous_actions.nurture_action.RedisClientService'
        ) as mock_cls:
            mock_cls.create_connection.return_value = mock_redis
            from services.autonomous_actions.nurture_action import NurtureAction
            NurtureAction.record_user_activity()
        assert mock_redis.get('spark_nurture:default:paused') is None
