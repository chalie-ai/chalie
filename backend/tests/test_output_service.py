"""
Tests for backend/services/output_service.py

OutputService manages the output queue and delivery via MemoryStore pub/sub,
routing text responses through SSE channels or the drift stream.
"""

import json
import pytest
from unittest.mock import patch

from services.memory_store import MemoryStore
from services.output_service import OutputService


@pytest.mark.unit
class TestOutputService:

    @pytest.fixture
    def mock_store(self):
        """
        Real MemoryStore instance with a spy wrapping ``publish``.

        The spy records every ``(channel, message)`` pair in ``store._published``
        so tests can assert on which channels received payloads without relying
        on mock call-history APIs.  All other operations (``set``, ``get``,
        ``rpush``, ``brpop``, ``delete``, etc.) hit the real in-memory store so
        state assertions work correctly.

        Yields:
            MemoryStore: Configured store instance with ``_published`` attribute.
        """
        store = MemoryStore()
        published = []
        _real_publish = store.publish

        def spy_publish(ch, msg):
            """Record ``(channel, message)`` then delegate to the real publish."""
            published.append((ch, msg))
            return _real_publish(ch, msg)

        store.publish = spy_publish
        store._published = published
        return store

    @pytest.fixture
    def service(self, mock_store):
        """
        Create OutputService with a real MemoryStore and stubbed config.

        Args:
            mock_store: Real MemoryStore (with publish spy) injected by fixture.

        Returns:
            OutputService: Fully initialised service instance.
        """
        connections = {
            "memory": {
                "topics": {"output_queue": "output-queue"},
            }
        }
        with patch('services.memory_client.MemoryClientService.create_connection',
                    return_value=mock_store), \
             patch('services.config_service.ConfigService.connections',
                   return_value=connections):
            svc = OutputService()
        return svc

    # ------------------------------------------------------------------ #
    # enqueue_text — SSE channel routing
    # ------------------------------------------------------------------ #

    def test_enqueue_text_with_sse_uuid_publishes_to_sse_channel(self, service, mock_store):
        """When metadata contains a uuid, text is published to sse:{uuid}."""
        metadata = {"uuid": "abc-123", "source": "user"}
        service.enqueue_text("topic-1", "Hello", "UNIFIED", 0.9, 0.5, metadata)

        channels = [ch for ch, _ in mock_store._published]
        assert "sse:abc-123" in channels

    def test_enqueue_text_with_sse_uuid_does_not_publish_to_output_events(self, service, mock_store):
        """SSE-routed text must NOT also go to output:events (prevents duplicates)."""
        metadata = {"uuid": "abc-123", "source": "user"}
        service.enqueue_text("topic-1", "Hello", "UNIFIED", 0.9, 0.5, metadata)

        channels = [ch for ch, _ in mock_store._published]
        assert "output:events" not in channels

    def test_enqueue_text_without_sse_uuid_publishes_to_output_events(self, service, mock_store):
        """Background text with a non-response source is published to output:events."""
        metadata = {"source": "task"}

        with patch('api.push.send_push_to_all'):
            service.enqueue_text("topic-1", "Drift thought", "UNIFIED", 0.8, 0.3, metadata)

        channels = [ch for ch, _ in mock_store._published]
        assert "output:events" in channels

    def test_enqueue_text_without_sse_uuid_buffers_to_notifications(self, service, mock_store):
        """Background text with a non-response source is pushed to notifications:recent."""
        metadata = {"source": "task"}

        with patch('api.push.send_push_to_all'):
            service.enqueue_text("topic-1", "Drift", "UNIFIED", 0.8, 0.3, metadata)

        recent = mock_store.lrange("notifications:recent", 0, -1)
        assert len(recent) > 0

    def test_enqueue_text_stores_output_with_setex(self, service, mock_store):
        """Output is persisted in MemoryStore under output:{id} with setex."""
        metadata = {"uuid": "xyz-789"}
        output_id = service.enqueue_text("t", "msg", "UNIFIED", 0.9, 0.1, metadata)

        raw = mock_store.get(f"output:{output_id}")
        assert raw is not None
        parsed = json.loads(raw)
        assert parsed["type"] == "TEXT"
        assert parsed["topic"] == "t"

    def test_enqueue_text_sets_one_hour_ttl(self, service, mock_store):
        """The stored output key has a 3600-second (1 hour) TTL."""
        output_id = service.enqueue_text("t", "msg", "UNIFIED", 0.9, 0.1, {"uuid": "u"})

        ttl_val = mock_store.ttl(f"output:{output_id}")
        assert 3590 <= ttl_val <= 3600

    # ------------------------------------------------------------------ #
    # enqueue_proactive
    # ------------------------------------------------------------------ #

    def test_enqueue_proactive_publishes_to_output_events(self, service, mock_store):
        """enqueue_proactive publishes to output:events (no SSE channel)."""
        with patch('api.push.send_push_to_all'):
            service.enqueue_proactive("thread-42", "Progress update")

        channels = [ch for ch, _ in mock_store._published]
        assert "output:events" in channels

    def test_enqueue_proactive_default_source_maps_to_task_event(self, service, mock_store):
        """Default source='task' maps to SSE event type 'task'."""
        with patch('api.push.send_push_to_all'):
            service.enqueue_proactive("thread-42", "Done!")

        for ch, msg in mock_store._published:
            if ch == "output:events":
                payload = json.loads(msg)
                assert payload["type"] == "task"
                return
        pytest.fail("No publish to output:events found")

    def test_enqueue_proactive_returns_output_id(self, service, mock_store):
        """enqueue_proactive returns a UUID string."""
        with patch('api.push.send_push_to_all'):
            output_id = service.enqueue_proactive("thread-1", "hello")
        assert isinstance(output_id, str)
        assert len(output_id) > 0

    def test_enqueue_proactive_buffers_to_notifications(self, service, mock_store):
        """Proactive output is buffered to notifications:recent for catch-up."""
        with patch('api.push.send_push_to_all'):
            service.enqueue_proactive("thread-42", "Update")

        recent = mock_store.lrange("notifications:recent", 0, -1)
        assert len(recent) > 0

    # ------------------------------------------------------------------ #
    # dequeue
    # ------------------------------------------------------------------ #

    def test_dequeue_returns_output_from_queue(self, service, mock_store):
        """Successful dequeue returns the parsed output dict."""
        output_data = {
            "id": "out-1",
            "type": "ACT",
            "topic": "t",
            "metadata": {"actions": ["search"]},
        }
        mock_store.set("output:out-1", json.dumps(output_data))
        mock_store.rpush("output-queue", "out-1")

        result = service.dequeue(output_type="ACT", timeout=1)

        assert result is not None
        assert result["id"] == "out-1"
        assert result["type"] == "ACT"

    def test_dequeue_requeues_mismatched_type(self, service, mock_store):
        """When dequeued output type does not match, it is re-queued via lpush."""
        text_output = {"id": "out-2", "type": "TEXT", "topic": "t", "metadata": {}}
        act_output = {"id": "out-3", "type": "ACT", "topic": "t", "metadata": {}}

        # rpush appends to the right; brpop pops from the right.
        # Push out-3 first so out-2 sits at the right end and is popped first.
        mock_store.set("output:out-2", json.dumps(text_output))
        mock_store.set("output:out-3", json.dumps(act_output))
        mock_store.rpush("output-queue", "out-3")
        mock_store.rpush("output-queue", "out-2")

        result = service.dequeue(output_type="ACT", timeout=1)

        assert result["type"] == "ACT"
        # out-2 (TEXT) must have been re-queued back onto the list
        queued = mock_store.lrange("output-queue", 0, -1)
        assert "out-2" in queued

    def test_dequeue_returns_none_on_timeout(self, service, mock_store):
        """When brpop times out (returns None), dequeue returns None."""
        # Empty queue — brpop will exhaust the timeout and return None
        result = service.dequeue(output_type="ACT", timeout=1)
        assert result is None

    # ------------------------------------------------------------------ #
    # delete_output
    # ------------------------------------------------------------------ #

    def test_delete_output_calls_store_delete(self, service, mock_store):
        """delete_output removes the output key from the store."""
        mock_store.set("output:dead-uuid-42", json.dumps({"id": "dead-uuid-42"}))
        service.delete_output("dead-uuid-42")
        assert mock_store.get("output:dead-uuid-42") is None

    # ------------------------------------------------------------------ #
    # notifications:recent trimming
    # ------------------------------------------------------------------ #

    def test_notifications_list_trimmed_to_200(self, service, mock_store):
        """After rpush to notifications:recent, ltrim keeps only the last 200."""
        metadata = {"source": "task"}

        # Pre-populate with 250 items so the trim is verifiable by state
        for i in range(250):
            mock_store.rpush("notifications:recent", f"old-item-{i}")

        with patch('api.push.send_push_to_all'):
            service.enqueue_text("t", "msg", "UNIFIED", 0.8, 0.2, metadata)

        recent = mock_store.lrange("notifications:recent", 0, -1)
        assert len(recent) <= 200

    # ------------------------------------------------------------------ #
    # response-source gating — notifications:recent (cycle 181)
    # ------------------------------------------------------------------ #

    def test_response_source_not_buffered_in_notifications(self, service, mock_store):
        """source='subagent_return' (maps to event_type='response') must not rpush to notifications:recent."""
        metadata = {"source": "subagent_return"}

        with patch('api.push.send_push_to_all'):
            service.enqueue_text("topic-1", "Async findings", "UNIFIED", 0.9, 0.1, metadata)

        recent = mock_store.lrange("notifications:recent", 0, -1)
        assert len(recent) == 0

    def test_task_source_is_buffered_in_notifications(self, service, mock_store):
        """source='task' (maps to event_type='task') must rpush to notifications:recent."""
        metadata = {"source": "task"}

        with patch('api.push.send_push_to_all'):
            service.enqueue_text("topic-1", "Background update", "UNIFIED", 0.9, 0.1, metadata)

        recent = mock_store.lrange("notifications:recent", 0, -1)
        assert len(recent) > 0

    # ------------------------------------------------------------------ #
    # response-source gating — output:events live publish (cycle 182)
    # ------------------------------------------------------------------ #

    def test_response_source_not_published_to_output_events(self, service, mock_store):
        """source='subagent_return' (maps to event_type='response') must not publish to output:events."""
        metadata = {"source": "subagent_return"}

        with patch('api.push.send_push_to_all'):
            service.enqueue_text("topic-1", "Async findings", "UNIFIED", 0.9, 0.1, metadata)

        channels = [ch for ch, _ in mock_store._published]
        assert "output:events" not in channels

    def test_task_source_is_published_to_output_events(self, service, mock_store):
        """source='task' (maps to event_type='task') must publish to output:events."""
        metadata = {"source": "task"}

        with patch('api.push.send_push_to_all'):
            service.enqueue_text("topic-1", "Background update", "UNIFIED", 0.9, 0.1, metadata)

        channels = [ch for ch, _ in mock_store._published]
        assert "output:events" in channels

        # Verify the published payload contains the correct event type
        for ch, msg in mock_store._published:
            if ch == "output:events":
                payload = json.loads(msg)
                assert payload["type"] == "task"
                return
        pytest.fail("No publish to output:events found")
