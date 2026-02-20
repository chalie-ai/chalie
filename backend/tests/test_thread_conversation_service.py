"""Tests for ThreadConversationService — CRUD operations, TTL behavior."""

import json
import pytest
from unittest.mock import patch

from services.thread_conversation_service import ThreadConversationService


@pytest.fixture
def conv_service(mock_redis):
    """ThreadConversationService with fake Redis."""
    with patch('services.thread_conversation_service.RedisClientService.create_connection', return_value=mock_redis):
        yield ThreadConversationService()


THREAD_ID = "telegram:user1:chan1:1"


class TestAddExchange:
    def test_adds_exchange_with_prompt(self, conv_service, mock_redis):
        eid = conv_service.add_exchange(THREAD_ID, "test-topic", {
            "message": "Hello there",
            "classification_time": 0.05,
        })

        assert eid  # non-empty UUID
        history = conv_service.get_conversation_history(THREAD_ID)
        assert len(history) == 1
        assert history[0]["prompt"]["message"] == "Hello there"
        assert history[0]["topic"] == "test-topic"

    def test_exchange_has_no_response_initially(self, conv_service):
        conv_service.add_exchange(THREAD_ID, "topic", {"message": "Hi"})
        history = conv_service.get_conversation_history(THREAD_ID)
        assert history[0]["response"] is None

    def test_exchange_count_increments(self, conv_service):
        conv_service.add_exchange(THREAD_ID, "topic", {"message": "1"})
        conv_service.add_exchange(THREAD_ID, "topic", {"message": "2"})
        assert conv_service.get_exchange_count(THREAD_ID) == 2


class TestAddResponse:
    def test_adds_response_to_latest_exchange(self, conv_service):
        conv_service.add_exchange(THREAD_ID, "topic", {"message": "Hi"})
        conv_service.add_response(THREAD_ID, "Hello back!", 1.5)

        history = conv_service.get_conversation_history(THREAD_ID)
        assert history[0]["response"]["message"] == "Hello back!"
        assert history[0]["response"]["generation_time"] == 1.5

    def test_response_error(self, conv_service):
        conv_service.add_exchange(THREAD_ID, "topic", {"message": "Hi"})
        conv_service.add_response_error(THREAD_ID, "LLM timeout")

        history = conv_service.get_conversation_history(THREAD_ID)
        assert "error" in history[0]["response"]
        assert history[0]["response"]["error"] == "LLM timeout"


class TestAddSteps:
    def test_adds_steps_to_latest_exchange(self, conv_service):
        conv_service.add_exchange(THREAD_ID, "topic", {"message": "Check weather"})
        conv_service.add_steps_to_exchange(THREAD_ID, [
            {"type": "recall", "description": "Look up weather data"},
        ])

        history = conv_service.get_conversation_history(THREAD_ID)
        assert len(history[0]["steps"]) == 1
        assert history[0]["steps"][0]["status"] == "pending"


class TestAddMemoryChunk:
    def test_adds_memory_chunk_by_exchange_id(self, conv_service):
        eid = conv_service.add_exchange(THREAD_ID, "topic", {"message": "Hi"})
        conv_service.add_memory_chunk(THREAD_ID, eid, {"gists": [{"content": "Greeting"}]})

        history = conv_service.get_conversation_history(THREAD_ID)
        assert history[0]["memory_chunk"]["gists"][0]["content"] == "Greeting"


class TestGetActiveSteps:
    def test_returns_only_active_steps(self, conv_service):
        conv_service.add_exchange(THREAD_ID, "topic", {"message": "Do stuff"})
        conv_service.add_steps_to_exchange(THREAD_ID, [
            {"type": "task", "description": "Step 1"},
            {"type": "task", "description": "Step 2"},
        ])

        # Mark step 1 as completed by updating the exchange directly
        history = conv_service.get_conversation_history(THREAD_ID)
        exchange = history[0]
        exchange["steps"][0]["status"] = "completed"
        conv_service.redis.lset(conv_service._conv_key(THREAD_ID), 0, json.dumps(exchange))

        active = conv_service.get_active_steps(THREAD_ID)
        assert len(active) == 1
        assert active[0]["description"] == "Step 2"


class TestGetLatestExchangeId:
    def test_returns_latest_id(self, conv_service):
        eid1 = conv_service.add_exchange(THREAD_ID, "topic", {"message": "First"})
        eid2 = conv_service.add_exchange(THREAD_ID, "topic", {"message": "Second"})

        assert conv_service.get_latest_exchange_id(THREAD_ID) == eid2

    def test_returns_unknown_for_empty(self, conv_service):
        assert conv_service.get_latest_exchange_id(THREAD_ID) == "unknown"


class TestRemoveExchanges:
    def test_removes_specific_exchanges(self, conv_service):
        eid1 = conv_service.add_exchange(THREAD_ID, "topic", {"message": "Keep"})
        eid2 = conv_service.add_exchange(THREAD_ID, "topic", {"message": "Remove"})

        conv_service.remove_exchanges(THREAD_ID, [eid2])

        history = conv_service.get_conversation_history(THREAD_ID)
        assert len(history) == 1
        assert history[0]["prompt"]["message"] == "Keep"
