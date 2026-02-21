"""Tests for SessionService — episode triggers, counters, topic switching."""

import time
import pytest
from unittest.mock import patch
from services.session_service import SessionService


pytestmark = pytest.mark.unit


class TestSessionService:

    def test_episode_trigger_at_3_topic_exchanges(self):
        """3 exchanges on same topic → should_generate_episode returns True."""
        svc = SessionService(inactivity_timeout=600)
        now = time.time()
        svc.track_classification("topic-a", False, now)

        for i in range(3):
            svc.add_exchange({'prompt': f'q{i}', 'response': f'a{i}'})

        should, reason = svc.should_generate_episode()
        assert should is True
        assert 'topic_exchange_threshold' in reason

    def test_episode_trigger_at_5_global_exchanges(self):
        """5 global exchanges across topics → trigger on global threshold."""
        svc = SessionService(inactivity_timeout=600)
        now = time.time()
        svc.track_classification("topic-a", False, now)

        # 2 exchanges on topic-a
        for i in range(2):
            svc.add_exchange({'prompt': f'q{i}', 'response': f'a{i}'})

        # Switch topic, add 2 more (topic count=2, global=4, no trigger yet)
        svc.mark_topic_switch("topic-b")
        for i in range(2):
            svc.add_exchange({'prompt': f'q{i}', 'response': f'a{i}'})

        should_pre, _ = svc.should_generate_episode()
        assert should_pre is False  # topic=2, global=4 — no trigger

        # Switch again, add 1 more (topic count=1, global=5 → trigger)
        svc.mark_topic_switch("topic-c")
        svc.add_exchange({'prompt': 'q5', 'response': 'a5'})

        should, reason = svc.should_generate_episode()
        assert should is True
        assert 'global_exchange_threshold' in reason

    def test_inactivity_trigger(self):
        """600s idle → trigger."""
        svc = SessionService(inactivity_timeout=600)
        now = time.time()
        svc.track_classification("topic-a", False, now)
        svc.add_exchange({'prompt': 'q', 'response': 'a'})

        # Simulate inactivity by backdating last_activity_time
        svc.last_activity_time = time.time() - 700

        should, reason = svc.should_generate_episode()
        assert should is True
        assert reason == 'inactivity'

    def test_no_trigger_below_thresholds(self):
        """2 exchanges, recent activity → no trigger."""
        svc = SessionService(inactivity_timeout=600)
        now = time.time()
        svc.track_classification("topic-a", False, now)

        for i in range(2):
            svc.add_exchange({'prompt': f'q{i}', 'response': f'a{i}'})

        should, reason = svc.should_generate_episode()
        assert should is False

    def test_counter_reset_on_topic_switch(self):
        """Topic switch resets topic counter but not global."""
        svc = SessionService(inactivity_timeout=600)
        now = time.time()
        svc.track_classification("topic-a", False, now)

        svc.add_exchange({'prompt': 'q1', 'response': 'a1'})
        svc.add_exchange({'prompt': 'q2', 'response': 'a2'})
        assert svc.topic_exchange_count == 2
        assert svc.global_exchange_count == 2

        svc.mark_topic_switch("topic-b")
        assert svc.topic_exchange_count == 0
        assert svc.global_exchange_count == 2

    def test_session_data_structure(self):
        """get_session_data returns expected dict shape."""
        svc = SessionService(inactivity_timeout=600)
        now = time.time()
        svc.track_classification("topic-a", False, now)
        svc.add_exchange({'prompt': 'q', 'response': 'a'})

        data = svc.get_session_data()
        assert 'topic' in data
        assert 'exchanges' in data
        assert 'start_time' in data
        assert data['topic'] == 'topic-a'
        assert len(data['exchanges']) == 1


class TestIsReturningFromSilence:

    def test_returns_zero_before_any_activity(self):
        """No prior activity → is_returning_from_silence returns 0.0."""
        svc = SessionService()
        assert svc.is_returning_from_silence() == 0.0

    def test_returns_zero_for_short_gap(self):
        """100s gap (below 2700s threshold) → returns 0.0."""
        svc = SessionService()
        svc.last_activity_time = time.time() - 100
        assert svc.is_returning_from_silence() == 0.0

    def test_returns_gap_at_threshold(self):
        """2700s gap → returns approximately 2700.0 (a positive float)."""
        svc = SessionService()
        svc.last_activity_time = time.time() - 2700
        result = svc.is_returning_from_silence(threshold_seconds=2700)
        assert result > 0
        assert isinstance(result, float)

    def test_must_be_called_before_track_classification(self):
        """
        Verify that calling track_classification updates last_activity_time,
        so is_returning_from_silence must be called before it.
        """
        svc = SessionService()
        # Set up an old activity time
        old_time = time.time() - 5000
        svc.last_activity_time = old_time

        # Before track_classification: should detect silence
        gap = svc.is_returning_from_silence(threshold_seconds=2700)
        assert gap > 0

        # After track_classification: last_activity_time is updated to now
        svc.track_classification("topic-a", False, time.time())

        # Now gap is near 0 — no silence detected
        gap_after = svc.is_returning_from_silence(threshold_seconds=2700)
        assert gap_after == 0.0

    def test_thread_backed_reads_redis(self):
        """When thread_id is set, reads last_activity from Redis thread hash."""
        from unittest.mock import MagicMock, patch

        svc = SessionService()
        svc._thread_id = "test-thread"
        svc._redis = MagicMock()

        # Simulate 3000s gap stored in Redis
        svc._redis.hget.return_value = str(time.time() - 3000)

        result = svc.is_returning_from_silence(threshold_seconds=2700)
        assert result > 0
