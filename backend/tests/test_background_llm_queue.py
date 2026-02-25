"""
Unit tests for BackgroundLLMProxy and background_llm_worker helpers.

All Redis interactions use fakeredis — no real connections are made.
"""

import json
import time
import pytest
import fakeredis
from unittest.mock import patch, MagicMock

from services.background_llm_queue import (
    BackgroundLLMProxy,
    create_background_llm_proxy,
    QUEUE_KEY,
    RESULT_KEY_PREFIX,
    HEARTBEAT_KEY,
    MAX_QUEUE_DEPTH,
)
from workers.background_llm_worker import _get_sleep_interval, BUSY_SLEEP, NORMAL_SLEEP, IDLE_SLEEP


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_redis():
    r = fakeredis.FakeRedis(decode_responses=True)
    yield r
    r.flushall()


@pytest.fixture
def proxy(fake_redis):
    """BackgroundLLMProxy wired to a fakeredis instance."""
    with patch("services.background_llm_queue.RedisClientService.create_connection",
               return_value=fake_redis):
        yield BackgroundLLMProxy("test-agent")


# ---------------------------------------------------------------------------
# BackgroundLLMProxy
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBackgroundLLMProxy:

    def test_factory_returns_proxy(self, fake_redis):
        with patch("services.background_llm_queue.RedisClientService.create_connection",
                   return_value=fake_redis):
            proxy = create_background_llm_proxy("my-agent")
        assert isinstance(proxy, BackgroundLLMProxy)
        assert proxy.agent_name == "my-agent"

    def test_enqueue_job_pushed_to_queue(self, proxy, fake_redis):
        """send_message should RPUSH a job to bg_llm:queue before blocking."""
        # Simulate an immediate result so BRPOP doesn't block
        job_data = {"text": "hello", "model": "m", "provider": None,
                    "tokens_input": 1, "tokens_output": 1, "latency_ms": 100}

        # We'll push a fake result immediately after proxy enqueues
        def fake_brpop(key, timeout):
            items = fake_redis.lrange(QUEUE_KEY, 0, -1)
            assert len(items) == 1
            job = json.loads(items[0])
            result_key = f"{RESULT_KEY_PREFIX}{job['job_id']}"
            return (result_key, json.dumps(job_data))

        with patch.object(fake_redis, "brpop", side_effect=fake_brpop):
            response = proxy.send_message("sys", "user")

        assert response is not None
        assert response.text == "hello"
        assert response.model == "m"

    def test_returns_none_when_queue_full(self, proxy, fake_redis):
        """Proxy should return None and log warning when queue depth >= MAX_QUEUE_DEPTH."""
        # Fill the queue beyond the cap
        for _ in range(MAX_QUEUE_DEPTH):
            fake_redis.rpush(QUEUE_KEY, json.dumps({"job_id": "x"}))

        result = proxy.send_message("sys", "user")
        assert result is None

    def test_returns_none_on_brpop_timeout(self, proxy, fake_redis):
        """Proxy should return None when BRPOP times out."""
        with patch.object(fake_redis, "brpop", return_value=None):
            result = proxy.send_message("sys", "user")
        assert result is None

    def test_returns_none_on_error_result(self, proxy, fake_redis):
        """Proxy returns None when worker pushes an error payload."""
        error_payload = json.dumps({"error": "max_retries_exceeded"})

        def fake_brpop(key, timeout):
            return (key, error_payload)

        with patch.object(fake_redis, "brpop", side_effect=fake_brpop):
            result = proxy.send_message("sys", "user")

        assert result is None

    def test_stale_heartbeat_logs_critical(self, proxy, fake_redis, caplog):
        """Proxy should log CRITICAL when worker heartbeat is >30s old."""
        import logging
        stale_ts = str(time.time() - 60)  # 60 seconds ago
        fake_redis.set(HEARTBEAT_KEY, stale_ts)

        # Return None from brpop so test exits quickly
        with patch.object(fake_redis, "brpop", return_value=None):
            with caplog.at_level(logging.CRITICAL, logger="services.background_llm_queue"):
                proxy.send_message("sys", "user")

        assert any("stale" in r.message.lower() for r in caplog.records)

    def test_job_payload_includes_retry_count_zero(self, proxy, fake_redis):
        """First enqueue should always set retry_count=0."""
        captured = []

        def fake_brpop(key, timeout):
            items = fake_redis.lrange(QUEUE_KEY, 0, -1)
            if items:
                captured.append(json.loads(items[0]))
            return None  # timeout to exit quickly

        with patch.object(fake_redis, "brpop", side_effect=fake_brpop):
            proxy.send_message("sys", "user")

        assert len(captured) == 1
        assert captured[0]["retry_count"] == 0
        assert captured[0]["agent_name"] == "test-agent"


# ---------------------------------------------------------------------------
# Adaptive sleep helper
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGetSleepInterval:

    def test_busy_when_prompt_queue_has_items(self, fake_redis):
        fake_redis.rpush("prompt-queue", "item")
        sleep = _get_sleep_interval(fake_redis)
        assert BUSY_SLEEP * 0.89 <= sleep <= BUSY_SLEEP * 1.11

    def test_busy_when_recent_interaction(self, fake_redis):
        fake_redis.set("proactive:default:last_interaction_ts", str(time.time() - 60))
        sleep = _get_sleep_interval(fake_redis)
        assert BUSY_SLEEP * 0.89 <= sleep <= BUSY_SLEEP * 1.11

    def test_normal_when_moderately_recent_interaction(self, fake_redis):
        fake_redis.set("proactive:default:last_interaction_ts", str(time.time() - 180))
        sleep = _get_sleep_interval(fake_redis)
        assert NORMAL_SLEEP * 0.89 <= sleep <= NORMAL_SLEEP * 1.11

    def test_idle_when_no_interaction_key(self, fake_redis):
        sleep = _get_sleep_interval(fake_redis)
        assert IDLE_SLEEP * 0.89 <= sleep <= IDLE_SLEEP * 1.11

    def test_idle_when_old_interaction(self, fake_redis):
        fake_redis.set("proactive:default:last_interaction_ts", str(time.time() - 600))
        sleep = _get_sleep_interval(fake_redis)
        assert IDLE_SLEEP * 0.89 <= sleep <= IDLE_SLEEP * 1.11

    def test_jitter_applied(self, fake_redis):
        """Same state should produce slightly different sleep durations across calls."""
        fake_redis.set("proactive:default:last_interaction_ts", str(time.time() - 600))
        sleeps = {_get_sleep_interval(fake_redis) for _ in range(20)}
        # With ±10% jitter over 20 calls, we expect multiple unique values
        assert len(sleeps) > 1
