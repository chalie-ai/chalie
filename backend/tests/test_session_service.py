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
