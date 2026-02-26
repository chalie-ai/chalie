"""
Tests for backend/services/output_service.py

OutputService manages the output queue and delivery via Redis pub/sub,
routing text responses through SSE channels or the drift stream, and
cards always through the drift stream.
"""

import json
import pytest
from unittest.mock import patch, MagicMock, call

from services.output_service import OutputService


@pytest.mark.unit
class TestOutputService:

    @pytest.fixture
    def mock_redis(self):
        """Isolated mock Redis — no real connections."""
        return MagicMock()

    @pytest.fixture
    def service(self, mock_redis):
        """Create OutputService with mocked Redis and config."""
        connections = {
            "redis": {
                "host": "localhost",
                "port": 6379,
                "topics": {"output_queue": "output-queue"},
            }
        }
        with patch('services.redis_client.RedisClientService.create_connection',
                    return_value=mock_redis), \
             patch('services.config_service.ConfigService.connections',
                   return_value=connections):
            svc = OutputService()
        return svc

    # ------------------------------------------------------------------ #
    # enqueue_text — SSE channel routing
    # ------------------------------------------------------------------ #

    def test_enqueue_text_with_sse_uuid_publishes_to_sse_channel(self, service, mock_redis):
        """When metadata contains a uuid, text is published to sse:{uuid}."""
        metadata = {"uuid": "abc-123", "source": "user"}
        service.enqueue_text("topic-1", "Hello", "RESPOND", 0.9, 0.5, metadata)

        publish_calls = mock_redis.publish.call_args_list
        channels = [c[0][0] for c in publish_calls]
        assert "sse:abc-123" in channels

    def test_enqueue_text_with_sse_uuid_does_not_publish_to_output_events(self, service, mock_redis):
        """SSE-routed text must NOT also go to output:events (prevents duplicates)."""
        metadata = {"uuid": "abc-123", "source": "user"}
        service.enqueue_text("topic-1", "Hello", "RESPOND", 0.9, 0.5, metadata)

        publish_calls = mock_redis.publish.call_args_list
        channels = [c[0][0] for c in publish_calls]
        assert "output:events" not in channels

    def test_enqueue_text_without_sse_uuid_publishes_to_output_events(self, service, mock_redis):
        """Background text (no SSE channel) is published to output:events."""
        metadata = {"source": "proactive_drift"}
        service.enqueue_text("topic-1", "Drift thought", "RESPOND", 0.8, 0.3, metadata)

        publish_calls = mock_redis.publish.call_args_list
        channels = [c[0][0] for c in publish_calls]
        assert "output:events" in channels

    def test_enqueue_text_without_sse_uuid_buffers_to_notifications(self, service, mock_redis):
        """Background text is pushed to notifications:recent for catch-up."""
        metadata = {"source": "proactive_drift"}

        with patch('api.push.send_push_to_all'):
            service.enqueue_text("topic-1", "Drift", "RESPOND", 0.8, 0.3, metadata)

        rpush_calls = mock_redis.rpush.call_args_list
        keys = [c[0][0] for c in rpush_calls]
        assert "notifications:recent" in keys

    def test_enqueue_text_stores_output_with_setex(self, service, mock_redis):
        """Output is persisted in Redis under output:{id} with setex."""
        metadata = {"uuid": "xyz-789"}
        output_id = service.enqueue_text("t", "msg", "RESPOND", 0.9, 0.1, metadata)

        setex_calls = mock_redis.setex.call_args_list
        assert len(setex_calls) == 1
        key, ttl, data = setex_calls[0][0]
        assert key == f"output:{output_id}"
        assert ttl == 3600
        parsed = json.loads(data)
        assert parsed["type"] == "TEXT"
        assert parsed["topic"] == "t"

    def test_enqueue_text_sets_one_hour_ttl(self, service, mock_redis):
        """The stored output key has a 3600-second (1 hour) TTL."""
        output_id = service.enqueue_text("t", "msg", "RESPOND", 0.9, 0.1, {"uuid": "u"})

        _key, ttl, _data = mock_redis.setex.call_args[0]
        assert ttl == 3600

    # ------------------------------------------------------------------ #
    # enqueue_card — always drift stream
    # ------------------------------------------------------------------ #

    def test_enqueue_card_publishes_to_output_events(self, service, mock_redis):
        """Card output always goes to output:events (drift stream)."""
        card_data = {
            "html": "<div>card</div>",
            "css": "",
            "scope_id": "s1",
            "title": "Test",
            "tool_name": "weather",
        }
        service.enqueue_card("topic-2", card_data)

        publish_calls = mock_redis.publish.call_args_list
        channels = [c[0][0] for c in publish_calls]
        assert "output:events" in channels

        # Verify the published payload contains card type
        for c in publish_calls:
            if c[0][0] == "output:events":
                payload = json.loads(c[0][1])
                assert payload["type"] == "card"
                assert payload["html"] == "<div>card</div>"
                break

    def test_enqueue_card_buffers_to_notifications(self, service, mock_redis):
        """Card output is buffered to notifications:recent for reconnect catch-up."""
        card_data = {"html": "<p>hi</p>", "css": "", "scope_id": "s", "title": "T", "tool_name": "t"}
        service.enqueue_card("topic-2", card_data)

        rpush_calls = mock_redis.rpush.call_args_list
        keys = [c[0][0] for c in rpush_calls]
        assert "notifications:recent" in keys

    # ------------------------------------------------------------------ #
    # enqueue_close_signal
    # ------------------------------------------------------------------ #

    def test_enqueue_close_signal_publishes_close_to_sse_channel(self, service, mock_redis):
        """Close signal publishes {"type": "close"} to the correct SSE channel."""
        service.enqueue_close_signal("close-uuid-99")

        mock_redis.publish.assert_called_once()
        channel, payload = mock_redis.publish.call_args[0]
        assert channel == "sse:close-uuid-99"
        assert json.loads(payload) == {"type": "close"}

    def test_enqueue_close_signal_noop_for_empty_uuid(self, service, mock_redis):
        """Empty or falsy sse_uuid is a no-op (no publish)."""
        service.enqueue_close_signal("")
        service.enqueue_close_signal(None)
        mock_redis.publish.assert_not_called()

    # ------------------------------------------------------------------ #
    # dequeue
    # ------------------------------------------------------------------ #

    def test_dequeue_returns_output_from_queue(self, service, mock_redis):
        """Successful dequeue returns the parsed output dict."""
        output_data = {
            "id": "out-1",
            "type": "ACT",
            "topic": "t",
            "metadata": {"actions": ["search"]},
        }
        mock_redis.brpop.return_value = ("output-queue", "out-1")
        mock_redis.get.return_value = json.dumps(output_data)

        result = service.dequeue(output_type="ACT", timeout=5)

        assert result is not None
        assert result["id"] == "out-1"
        assert result["type"] == "ACT"
        mock_redis.brpop.assert_called_once_with("output-queue", 5)

    def test_dequeue_requeues_mismatched_type(self, service, mock_redis):
        """When dequeued output type does not match, it is re-queued via lpush."""
        text_output = json.dumps({"id": "out-2", "type": "TEXT", "topic": "t", "metadata": {}})
        act_output = json.dumps({"id": "out-3", "type": "ACT", "topic": "t", "metadata": {}})

        # First brpop returns TEXT (wrong type), second returns ACT (correct)
        mock_redis.brpop.side_effect = [
            ("output-queue", "out-2"),
            ("output-queue", "out-3"),
        ]
        mock_redis.get.side_effect = [text_output, act_output]

        result = service.dequeue(output_type="ACT", timeout=1)

        assert result["type"] == "ACT"
        # The TEXT output should have been re-queued
        mock_redis.lpush.assert_called_once_with("output-queue", "out-2")

    def test_dequeue_returns_none_on_timeout(self, service, mock_redis):
        """When brpop times out (returns None), dequeue returns None."""
        mock_redis.brpop.return_value = None

        result = service.dequeue(output_type="ACT", timeout=1)
        assert result is None

    # ------------------------------------------------------------------ #
    # delete_output
    # ------------------------------------------------------------------ #

    def test_delete_output_calls_redis_delete(self, service, mock_redis):
        """delete_output calls redis.delete with the correct key."""
        mock_redis.delete.return_value = 1
        service.delete_output("dead-uuid-42")
        mock_redis.delete.assert_called_once_with("output:dead-uuid-42")

    # ------------------------------------------------------------------ #
    # notifications:recent trimming
    # ------------------------------------------------------------------ #

    def test_notifications_list_trimmed_to_200(self, service, mock_redis):
        """After rpush to notifications:recent, ltrim keeps only the last 200."""
        metadata = {"source": "proactive_drift"}

        with patch('api.push.send_push_to_all'):
            service.enqueue_text("t", "msg", "RESPOND", 0.8, 0.2, metadata)

        ltrim_calls = mock_redis.ltrim.call_args_list
        assert len(ltrim_calls) >= 1
        # Verify the trim retains the last 200 entries
        for c in ltrim_calls:
            if c[0][0] == "notifications:recent":
                assert c[0][1] == -200
                assert c[0][2] == -1
                break
        else:
            pytest.fail("ltrim was not called on notifications:recent")
