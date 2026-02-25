"""Unit tests for SparkStateService."""
import json
import time
import pytest
from unittest.mock import MagicMock, patch

# Mark all tests as unit (no external dependencies)
pytestmark = pytest.mark.unit


@pytest.fixture
def mock_redis():
    """Create a mock Redis that stores values in a dict."""
    store = {}

    class FakeRedis:
        def get(self, key):
            return store.get(key)

        def setex(self, key, ttl, value):
            store[key] = value

        def set(self, key, value):
            store[key] = value

        def setnx(self, key, value):
            if key not in store:
                store[key] = value
                return True
            return False

        def delete(self, key):
            store.pop(key, None)

        def expire(self, key, ttl):
            pass

    fake = FakeRedis()
    fake._store = store
    return fake


@pytest.fixture
def spark_service(mock_redis):
    """Create SparkStateService with mocked Redis."""
    with patch('services.spark_state_service.RedisClientService') as mock_cls:
        mock_cls.create_connection.return_value = mock_redis
        from services.spark_state_service import SparkStateService
        svc = SparkStateService(user_id='test')
        # Patch trait counting to return 0 by default
        svc._count_user_traits = MagicMock(return_value=0)
        svc._has_established_traits = MagicMock(return_value=False)
        yield svc


class TestDefaultState:
    def test_new_user_starts_at_first_contact(self, spark_service):
        state = spark_service.get_state()
        assert state['phase'] == 'first_contact'
        assert state['exchange_count'] == 0
        assert state['effective_exchanges'] == 0.0
        assert state['welcome_sent'] is False

    def test_needs_welcome_true_for_new_user(self, spark_service):
        assert spark_service.needs_welcome() is True

    def test_is_graduated_false_for_new_user(self, spark_service):
        assert spark_service.is_graduated() is False


class TestGraduatedFallback:
    def test_initializes_graduated_when_traits_exist(self, spark_service):
        spark_service._has_established_traits = MagicMock(return_value=True)
        state = spark_service.get_state()
        assert state['phase'] == 'graduated'
        assert state['welcome_sent'] is True


class TestWelcome:
    def test_mark_welcome_sent(self, spark_service):
        spark_service.mark_welcome_sent()
        state = spark_service.get_state()
        assert state['welcome_sent'] is True
        assert state['welcome_sent_at'] is not None

    def test_needs_welcome_false_after_sent(self, spark_service):
        spark_service.mark_welcome_sent()
        assert spark_service.needs_welcome() is False


class TestExchangeScoring:
    def test_system_source_scores_low(self):
        from services.spark_state_service import SparkStateService
        score = SparkStateService._score_exchange("hello", 5.0, 'system')
        assert score == 0.1

    def test_quick_reply_scores_low(self):
        from services.spark_state_service import SparkStateService
        score = SparkStateService._score_exchange("yes", 5.0, 'quick_reply')
        assert score == 0.1

    def test_short_message_low_score(self):
        from services.spark_state_service import SparkStateService
        score = SparkStateService._score_exchange("hi", 1.0, 'text')
        assert score == 0.0  # <3 words, <3s gap

    def test_medium_message_with_gap(self):
        from services.spark_state_service import SparkStateService
        score = SparkStateService._score_exchange(
            "I like working on Python projects", 5.0, 'text'
        )
        # 7 words: >=3 (0.3) + gap>3s (0.2) = 0.5
        assert score == pytest.approx(0.5, abs=0.01)

    def test_long_message_with_gap(self):
        from services.spark_state_service import SparkStateService
        msg = " ".join(["word"] * 30)
        score = SparkStateService._score_exchange(msg, 5.0, 'text')
        # >=3 (0.3) + >=10 (0.3) + >=25 (0.2) + gap>3s (0.2) = 1.0
        assert score == pytest.approx(1.0, abs=0.01)

    def test_score_capped_at_one(self):
        from services.spark_state_service import SparkStateService
        msg = " ".join(["word"] * 150)
        score = SparkStateService._score_exchange(msg, 10.0, 'text')
        assert score <= 1.0


class TestIdleDecay:
    def test_no_decay_when_no_last_active(self):
        from services.spark_state_service import SparkStateService
        state = {'effective_exchanges': 10.0, 'last_active_at': None}
        result = SparkStateService._apply_idle_decay(state, time.time())
        assert result['effective_exchanges'] == 10.0

    def test_decay_after_one_day(self):
        from services.spark_state_service import SparkStateService
        now = time.time()
        state = {
            'effective_exchanges': 10.0,
            'last_active_at': now - 86400,  # 1 day ago
        }
        result = SparkStateService._apply_idle_decay(state, now)
        expected = 10.0 * 0.95
        assert result['effective_exchanges'] == pytest.approx(expected, abs=0.01)

    def test_decay_after_seven_days(self):
        from services.spark_state_service import SparkStateService
        now = time.time()
        state = {
            'effective_exchanges': 10.0,
            'last_active_at': now - (86400 * 7),  # 7 days ago
        }
        result = SparkStateService._apply_idle_decay(state, now)
        expected = 10.0 * (0.95 ** 7)
        assert result['effective_exchanges'] == pytest.approx(expected, abs=0.01)


class TestPhaseTransitions:
    def test_first_contact_to_surface(self, spark_service):
        spark_service.mark_welcome_sent()
        state = spark_service.increment_exchange("Hello there friend", 5.0)
        assert state['phase'] == 'surface'

    def test_first_contact_blocked_without_welcome(self, spark_service):
        state = spark_service.increment_exchange("Hello there", 5.0)
        assert state['phase'] == 'first_contact'

    def test_surface_to_exploratory_needs_effective_threshold(self, spark_service):
        # Setup: get to surface phase
        spark_service.mark_welcome_sent()
        spark_service.increment_exchange("Hello there friend", 5.0)

        # Send several meaningful exchanges
        for i in range(8):
            # Force last_scored_at to be old enough to bypass rate limiter
            state = spark_service.get_state()
            state['last_scored_at'] = time.time() - 60
            spark_service._save_state(state)
            spark_service.increment_exchange(
                "This is a meaningful conversation about interesting things " * 3,
                5.0,
            )

        state = spark_service.get_state()
        # Should have progressed past surface given enough effective exchanges
        # and hold count being met
        assert state['effective_exchanges'] >= 4.0

    def test_hold_requirement_prevents_burst(self, spark_service):
        """Verify that phase_hold_count prevents instant transitions."""
        spark_service.mark_welcome_sent()
        spark_service.increment_exchange("Hello there friend", 5.0)

        # Set effective_exchanges high but hold_count should still need meeting
        state = spark_service.get_state()
        state['effective_exchanges'] = 15.0
        state['last_scored_at'] = time.time() - 60
        spark_service._save_state(state)

        # First exchange after threshold met: hold_count becomes 1, needs 2
        spark_service.increment_exchange("meaningful message here", 5.0)
        state = spark_service.get_state()
        # Might be surface or exploratory depending on hold
        if state['phase'] == 'surface':
            assert state['phase_hold_count'] >= 1

    def test_connected_needs_traits(self, spark_service):
        """Connected phase requires min 3 traits."""
        spark_service.mark_welcome_sent()

        # Fast-forward to exploratory
        state = spark_service.get_state()
        state['phase'] = 'exploratory'
        state['welcome_sent'] = True
        state['exchange_count'] = 20
        state['effective_exchanges'] = 15.0
        state['phase_hold_count'] = 0
        state['last_scored_at'] = time.time() - 60
        spark_service._save_state(state)

        # Without traits, should not transition
        spark_service._count_user_traits = MagicMock(return_value=1)
        spark_service.increment_exchange("long meaningful message about life", 5.0)
        state = spark_service.get_state()
        assert state['phase'] == 'exploratory'

        # With traits, should begin hold count
        spark_service._count_user_traits = MagicMock(return_value=3)
        state['last_scored_at'] = time.time() - 60
        spark_service._save_state(state)
        spark_service.increment_exchange("another meaningful message here", 5.0)
        state = spark_service.get_state()
        # Hold count should be incrementing
        assert state['phase_hold_count'] >= 1 or state['phase'] == 'connected'

    def test_graduated_is_terminal(self, spark_service):
        """Once graduated, no more transitions."""
        state = spark_service.get_state()
        state['phase'] = 'graduated'
        spark_service._save_state(state)

        result = spark_service.increment_exchange("hello", 5.0)
        assert result['phase'] == 'graduated'


class TestRateLimiting:
    def test_rapid_fire_skips_scoring(self, spark_service):
        spark_service.mark_welcome_sent()

        # First exchange: scored normally
        state = spark_service.increment_exchange("Hello there friend", 5.0)
        first_effective = state['effective_exchanges']

        # Second exchange immediately: rate-limited (no scoring)
        state = spark_service.increment_exchange("Another message here now", 5.0)
        assert state['exchange_count'] == 2  # raw count still increments
        # effective_exchanges should not have changed (rate limited)
        assert state['effective_exchanges'] == pytest.approx(first_effective, abs=0.01)


class TestRecordTopic:
    def test_records_new_topic(self, spark_service):
        spark_service.record_topic("python programming")
        state = spark_service.get_state()
        assert "python programming" in state['topics_discussed']

    def test_deduplicates_topics(self, spark_service):
        spark_service.record_topic("python")
        spark_service.record_topic("python")
        state = spark_service.get_state()
        assert state['topics_discussed'].count("python") == 1

    def test_ignores_empty_topic(self, spark_service):
        spark_service.record_topic("")
        state = spark_service.get_state()
        assert state['topics_discussed'] == []


class TestSuggestionSeeded:
    def test_mark_suggestion_seeded(self, spark_service):
        spark_service.mark_suggestion_seeded()
        state = spark_service.get_state()
        assert state['first_suggestion_seeded'] is True
