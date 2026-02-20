"""Tests for ThreadService — resolution algorithm, expiry transitions, concurrency."""

import time
import json
import pytest
from unittest.mock import patch, MagicMock

from services.thread_service import ThreadService, ThreadResolution


@pytest.fixture
def thread_service(mock_redis):
    """ThreadService with fake Redis."""
    with patch('services.thread_service.RedisClientService.create_connection', return_value=mock_redis):
        svc = ThreadService(soft_expiry_minutes=30, hard_expiry_minutes=240)
        # Patch out PostgreSQL persistence (non-critical)
        with patch.object(svc, '_persist_thread_created'), \
             patch.object(svc, '_persist_thread_expired'):
            yield svc


class TestResolveThread:
    def test_creates_new_thread_when_none_exists(self, thread_service):
        result = thread_service.resolve_thread("user1", "chan1", "telegram")

        assert result.is_new is True
        assert result.is_resumed is False
        assert result.thread_id.startswith("telegram:user1:chan1:")
        assert result.resume_gap_minutes == 0.0

    def test_seamless_continuation_within_soft_expiry(self, thread_service):
        # Create initial thread
        r1 = thread_service.resolve_thread("user1", "chan1", "telegram")
        assert r1.is_new is True

        # Immediate follow-up
        r2 = thread_service.resolve_thread("user1", "chan1", "telegram")
        assert r2.is_new is False
        assert r2.is_resumed is False
        assert r2.thread_id == r1.thread_id

    def test_soft_resume_after_soft_expiry(self, thread_service, mock_redis):
        r1 = thread_service.resolve_thread("user1", "chan1", "telegram")

        # Simulate 35 minutes of inactivity (past soft, before hard)
        mock_redis.hset(f"thread:{r1.thread_id}", "last_activity", str(time.time() - 35 * 60))

        r2 = thread_service.resolve_thread("user1", "chan1", "telegram")
        assert r2.is_new is False
        assert r2.is_resumed is True
        assert r2.thread_id == r1.thread_id
        assert r2.resume_gap_minutes > 30

    def test_hard_expiry_creates_new_thread(self, thread_service, mock_redis):
        r1 = thread_service.resolve_thread("user1", "chan1", "telegram")

        # Simulate 5 hours of inactivity (past hard expiry)
        mock_redis.hset(f"thread:{r1.thread_id}", "last_activity", str(time.time() - 5 * 3600))

        r2 = thread_service.resolve_thread("user1", "chan1", "telegram")
        assert r2.is_new is True
        assert r2.thread_id != r1.thread_id
        assert r2.previous_thread_id == r1.thread_id

    def test_different_users_get_different_threads(self, thread_service):
        r1 = thread_service.resolve_thread("user1", "chan1", "telegram")
        r2 = thread_service.resolve_thread("user2", "chan1", "telegram")

        assert r1.thread_id != r2.thread_id

    def test_different_channels_get_different_threads(self, thread_service):
        r1 = thread_service.resolve_thread("user1", "chan1", "telegram")
        r2 = thread_service.resolve_thread("user1", "chan2", "telegram")

        assert r1.thread_id != r2.thread_id

    def test_sequence_increments(self, thread_service, mock_redis):
        r1 = thread_service.resolve_thread("user1", "chan1", "telegram")

        # Force hard expiry
        mock_redis.hset(f"thread:{r1.thread_id}", "last_activity", str(time.time() - 5 * 3600))

        r2 = thread_service.resolve_thread("user1", "chan1", "telegram")
        # Second thread should have sequence 2
        assert r2.thread_id.endswith(":2")


class TestExpireThread:
    def test_expire_sets_state(self, thread_service, mock_redis):
        r = thread_service.resolve_thread("user1", "chan1", "telegram")
        thread_service.expire_thread(r.thread_id)

        state = mock_redis.hget(f"thread:{r.thread_id}", "state")
        assert state == "expired"

    def test_expire_clears_pointer(self, thread_service, mock_redis):
        r = thread_service.resolve_thread("user1", "chan1", "telegram")
        thread_service.expire_thread(r.thread_id)

        pointer = mock_redis.get("active_thread:user1:chan1")
        assert pointer is None

    def test_double_expire_is_safe(self, thread_service):
        r = thread_service.resolve_thread("user1", "chan1", "telegram")
        thread_service.expire_thread(r.thread_id)
        thread_service.expire_thread(r.thread_id)  # Should not raise


class TestUpdateTopic:
    def test_update_topic_sets_current(self, thread_service, mock_redis):
        r = thread_service.resolve_thread("user1", "chan1", "telegram")
        thread_service.update_topic(r.thread_id, "daily-routine")

        current = mock_redis.hget(f"thread:{r.thread_id}", "current_topic")
        assert current == "daily-routine"

    def test_update_topic_appends_history(self, thread_service, mock_redis):
        r = thread_service.resolve_thread("user1", "chan1", "telegram")
        thread_service.update_topic(r.thread_id, "topic-a")
        thread_service.update_topic(r.thread_id, "topic-b")

        history = json.loads(mock_redis.hget(f"thread:{r.thread_id}", "topic_history"))
        assert "topic-a" in history
        assert "topic-b" in history

    def test_duplicate_topic_not_added_twice(self, thread_service, mock_redis):
        r = thread_service.resolve_thread("user1", "chan1", "telegram")
        thread_service.update_topic(r.thread_id, "topic-a")
        thread_service.update_topic(r.thread_id, "topic-a")

        history = json.loads(mock_redis.hget(f"thread:{r.thread_id}", "topic_history"))
        assert history.count("topic-a") == 1


class TestVisualContinuityBridge:
    def test_recent_context_returned_on_hard_expiry(self, thread_service, mock_redis):
        r1 = thread_service.resolve_thread("user1", "chan1", "telegram")

        # Store some conversation data in the old thread
        exchange = {
            "id": "test-id",
            "prompt": {"message": "What's the weather?"},
            "response": {"message": "It's sunny today."},
        }
        mock_redis.rpush(f"thread_conv:{r1.thread_id}", json.dumps(exchange))

        # Force hard expiry
        mock_redis.hset(f"thread:{r1.thread_id}", "last_activity", str(time.time() - 5 * 3600))

        r2 = thread_service.resolve_thread("user1", "chan1", "telegram")
        assert r2.recent_visible_context is not None
        assert len(r2.recent_visible_context) == 1
        assert r2.recent_visible_context[0]["prompt"] == "What's the weather?"
        assert r2.recent_visible_context[0]["response"] == "It's sunny today."
