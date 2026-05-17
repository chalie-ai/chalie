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
        with patch('services.memory_client.MemoryClientService.create_connection',
                    return_value=mock_store):
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
        """Background text (no SSE channel) is published to output:events."""
        metadata = {"source": "proactive"}
        service.enqueue_text("topic-1", "Drift thought", "UNIFIED", 0.8, 0.3, metadata)

        channels = [ch for ch, _ in mock_store._published]
        assert "output:events" in channels

    def test_enqueue_text_without_sse_uuid_buffers_to_notifications(self, service, mock_store):
        """Background text is pushed to notifications:recent for catch-up."""
        metadata = {"source": "proactive"}

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

    def test_enqueue_proactive_buffers_to_notifications(self, service, mock_store):
        """Proactive output is buffered to notifications:recent for catch-up."""
        with patch('api.push.send_push_to_all'):
            service.enqueue_proactive("thread-42", "Update")

        recent = mock_store.lrange("notifications:recent", 0, -1)
        assert len(recent) > 0

    # ------------------------------------------------------------------ #
    # notifications:recent trimming
    # ------------------------------------------------------------------ #

    def test_notifications_list_trimmed_to_200(self, service, mock_store):
        """After rpush to notifications:recent, ltrim keeps only the last 200."""
        metadata = {"source": "proactive"}

        # Pre-populate with 250 items so the trim is verifiable by state
        for i in range(250):
            mock_store.rpush("notifications:recent", f"old-item-{i}")

        with patch('api.push.send_push_to_all'):
            service.enqueue_text("t", "msg", "UNIFIED", 0.8, 0.2, metadata)

        recent = mock_store.lrange("notifications:recent", 0, -1)
        assert len(recent) <= 200
