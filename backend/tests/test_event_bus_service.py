"""Tests for EventBusService — handler registration, emit, process, synchronous dispatch."""

import json
import pytest
from unittest.mock import patch, MagicMock

from services.event_bus_service import EventBusService, ENCODE_EVENT, INTERACTION_LOGGED


pytestmark = pytest.mark.unit


@pytest.fixture
def event_bus(mock_redis):
    """EventBusService wired to fakeredis."""
    with patch('services.event_bus_service.RedisClientService') as mock_cls:
        mock_cls.create_connection.return_value = mock_redis
        bus = EventBusService()
    return bus


# ── Handler registration ─────────────────────────────────────────────

class TestSubscribe:

    def test_registers_single_handler(self, event_bus):
        handler = MagicMock()
        event_bus.subscribe(ENCODE_EVENT, handler)
        assert handler in event_bus._handlers[ENCODE_EVENT]

    def test_registers_multiple_handlers_same_event(self, event_bus):
        h1, h2 = MagicMock(), MagicMock()
        event_bus.subscribe(ENCODE_EVENT, h1)
        event_bus.subscribe(ENCODE_EVENT, h2)
        assert len(event_bus._handlers[ENCODE_EVENT]) == 2

    def test_registers_handlers_for_different_events(self, event_bus):
        h1, h2 = MagicMock(), MagicMock()
        event_bus.subscribe(ENCODE_EVENT, h1)
        event_bus.subscribe(INTERACTION_LOGGED, h2)
        assert ENCODE_EVENT in event_bus._handlers
        assert INTERACTION_LOGGED in event_bus._handlers


# ── emit ─────────────────────────────────────────────────────────────

class TestEmit:

    def test_pushes_to_redis_list(self, event_bus, mock_redis):
        result = event_bus.emit(ENCODE_EVENT, {'data': 'test'})
        assert result is True
        queue_key = f"event_bus:{ENCODE_EVENT}"
        assert mock_redis.llen(queue_key) == 1

    def test_event_is_json_serializable(self, event_bus, mock_redis):
        event_bus.emit(ENCODE_EVENT, {'key': 'value'})
        queue_key = f"event_bus:{ENCODE_EVENT}"
        raw = mock_redis.lpop(queue_key)
        parsed = json.loads(raw)
        assert parsed['payload'] == {'key': 'value'}

    def test_event_includes_timestamp(self, event_bus, mock_redis):
        event_bus.emit(ENCODE_EVENT, {})
        queue_key = f"event_bus:{ENCODE_EVENT}"
        raw = mock_redis.lpop(queue_key)
        parsed = json.loads(raw)
        assert 'timestamp' in parsed
        assert isinstance(parsed['timestamp'], float)


# ── process_events ───────────────────────────────────────────────────

class TestProcessEvents:

    def test_calls_handler_with_payload(self, event_bus, mock_redis):
        handler = MagicMock()
        event_bus.subscribe(ENCODE_EVENT, handler)
        event_bus.emit(ENCODE_EVENT, {'msg': 'hello'})

        count = event_bus.process_events(ENCODE_EVENT)

        assert count == 1
        handler.assert_called_once()
        args = handler.call_args[0]
        assert args[0] == ENCODE_EVENT
        assert args[1] == {'msg': 'hello'}

    def test_returns_count_of_processed(self, event_bus, mock_redis):
        handler = MagicMock()
        event_bus.subscribe(ENCODE_EVENT, handler)
        event_bus.emit(ENCODE_EVENT, {'n': 1})
        event_bus.emit(ENCODE_EVENT, {'n': 2})
        event_bus.emit(ENCODE_EVENT, {'n': 3})

        count = event_bus.process_events(ENCODE_EVENT, batch_size=10)
        assert count == 3

    def test_respects_batch_size(self, event_bus, mock_redis):
        handler = MagicMock()
        event_bus.subscribe(ENCODE_EVENT, handler)
        for i in range(5):
            event_bus.emit(ENCODE_EVENT, {'n': i})

        count = event_bus.process_events(ENCODE_EVENT, batch_size=2)
        assert count == 2
        # 3 remain
        queue_key = f"event_bus:{ENCODE_EVENT}"
        assert mock_redis.llen(queue_key) == 3

    def test_isolates_handler_errors(self, event_bus, mock_redis):
        """One handler error doesn't prevent other events from processing."""
        bad_handler = MagicMock(side_effect=ValueError("boom"))
        event_bus.subscribe(ENCODE_EVENT, bad_handler)
        event_bus.emit(ENCODE_EVENT, {'test': True})

        count = event_bus.process_events(ENCODE_EVENT)
        # Event is still consumed even though handler raised
        assert count == 1

    def test_returns_zero_when_no_handlers(self, event_bus, mock_redis):
        event_bus.emit(ENCODE_EVENT, {'orphan': True})
        count = event_bus.process_events(ENCODE_EVENT)
        assert count == 0


# ── emit_and_handle ──────────────────────────────────────────────────

class TestEmitAndHandle:

    def test_synchronous_dispatch_to_handlers(self, event_bus, mock_redis):
        handler = MagicMock()
        event_bus.subscribe(ENCODE_EVENT, handler)

        event_bus.emit_and_handle(ENCODE_EVENT, {'sync': True})

        handler.assert_called_once_with(ENCODE_EVENT, {'sync': True})

    def test_falls_back_to_redis_without_handlers(self, event_bus, mock_redis):
        """No handlers registered → falls back to emit() into Redis."""
        event_bus.emit_and_handle(INTERACTION_LOGGED, {'fallback': True})

        queue_key = f"event_bus:{INTERACTION_LOGGED}"
        assert mock_redis.llen(queue_key) == 1
