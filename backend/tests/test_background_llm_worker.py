"""Tests for background_llm_worker — adaptive sleep intervals and constants."""

import time
import pytest
from unittest.mock import MagicMock

from workers.background_llm_worker import (
    _get_sleep_interval,
    BUSY_SLEEP,
    NORMAL_SLEEP,
    IDLE_SLEEP,
    STALE_THRESHOLD,
    LLM_CALL_TIMEOUT,
    MAX_RETRIES,
    QUEUE_KEY,
    LAST_INTERACTION_KEY,
    PROMPT_QUEUE_KEY,
)


pytestmark = pytest.mark.unit


# ── Constants ────────────────────────────────────────────────────────

class TestConstants:

    def test_sleep_tier_values(self):
        assert BUSY_SLEEP == 10
        assert NORMAL_SLEEP == 5
        assert IDLE_SLEEP == 1

    def test_stale_threshold(self):
        assert STALE_THRESHOLD == 300

    def test_llm_call_timeout(self):
        assert LLM_CALL_TIMEOUT == 120

    def test_max_retries(self):
        assert MAX_RETRIES == 2


# ── _get_sleep_interval ──────────────────────────────────────────────

class TestGetSleepInterval:

    def test_busy_when_prompt_queue_has_items(self, mock_store):
        mock_store.rpush(PROMPT_QUEUE_KEY, "job1")
        result = _get_sleep_interval(mock_store)
        # BUSY_SLEEP=10, ±10% jitter → [9.0, 11.0]
        assert 9.0 <= result <= 11.0

    def test_busy_when_recent_interaction(self, mock_store):
        mock_store.set(LAST_INTERACTION_KEY, str(time.time() - 30))  # 30s ago
        result = _get_sleep_interval(mock_store)
        assert 9.0 <= result <= 11.0

    def test_normal_when_gap_120_to_300(self, mock_store):
        mock_store.set(LAST_INTERACTION_KEY, str(time.time() - 200))  # 200s gap
        result = _get_sleep_interval(mock_store)
        # NORMAL_SLEEP=5, ±10% → [4.5, 5.5]
        assert 4.5 <= result <= 5.5

    def test_idle_when_gap_over_300(self, mock_store):
        mock_store.set(LAST_INTERACTION_KEY, str(time.time() - 600))  # 10min gap
        result = _get_sleep_interval(mock_store)
        # IDLE_SLEEP=1, ±10% → [0.9, 1.1]
        assert 0.9 <= result <= 1.1

    def test_idle_when_no_timestamp(self, mock_store):
        # No LAST_INTERACTION_KEY set
        result = _get_sleep_interval(mock_store)
        assert 0.9 <= result <= 1.1

    def test_normal_on_store_error(self):
        broken_store = MagicMock()
        broken_store.llen.side_effect = ConnectionError("dead")
        result = _get_sleep_interval(broken_store)
        # Fallback: NORMAL_SLEEP=5, ±10% → [4.5, 5.5]
        assert 4.5 <= result <= 5.5
