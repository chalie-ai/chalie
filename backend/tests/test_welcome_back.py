"""
Unit tests for the welcome-back brief.

Verifies that :func:`capabilities.welcome_back.maybe_send_welcome_back`
correctly detects absence gaps and pushes a catch-up digest to prompt-queue.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest


def _utc(h=12, m=0):
    """Return a timezone-aware UTC datetime at the given hour."""
    return datetime(2026, 3, 29, h, m, tzinfo=timezone.utc)


def _make_store(last_interaction_age=7200, flag_set=False, hints=None):
    """Build a fake MemoryStore with configurable state."""
    store = MagicMock()

    def _get(k):
        if k == "welcome_back:sent":
            return "1" if flag_set else None
        if k == "proactive:default:last_interaction_ts":
            return str(time.time() - last_interaction_age)
        return None

    store.get.side_effect = _get
    store.rpush = MagicMock()
    store.setex = MagicMock()
    if hints is None:
        hints = [json.dumps({"source": "caldav", "content": "3 events today"})]
    store.lrange.return_value = hints
    return store


@pytest.mark.unit
class TestWelcomeBack:
    """Test the welcome-back brief proactive module."""

    @pytest.fixture(autouse=True)
    def _patch_quiet(self):
        with patch(
            "capabilities.quiet_window.is_quiet_now", return_value=False
        ):
            yield

    @patch("services.memory_client.MemoryClientService")
    def test_fires_after_absence(self, mock_mem):
        """Brief fires when user absent >1h and world hints available."""
        from capabilities.welcome_back import maybe_send_welcome_back

        store = _make_store(last_interaction_age=7200)
        mock_mem.create_connection.return_value = store

        result = maybe_send_welcome_back(_utc(14, 0))

        assert result is True
        store.rpush.assert_called_once()
        payload = json.loads(store.rpush.call_args[0][1])
        assert "WELCOME BACK" in payload["prompt"]
        assert payload["metadata"]["source"] == "welcome_back"
        store.setex.assert_called_once()

    @patch("services.memory_client.MemoryClientService")
    def test_skips_when_recently_active(self, mock_mem):
        """Brief does not fire if user was active within 1 hour."""
        from capabilities.welcome_back import maybe_send_welcome_back

        store = _make_store(last_interaction_age=600)
        mock_mem.create_connection.return_value = store

        result = maybe_send_welcome_back(_utc(14, 0))

        assert result is False
        store.rpush.assert_not_called()

    @patch("services.memory_client.MemoryClientService")
    def test_skips_when_already_sent(self, mock_mem):
        """Brief does not fire if dedup flag is set."""
        from capabilities.welcome_back import maybe_send_welcome_back

        store = _make_store(last_interaction_age=7200, flag_set=True)
        mock_mem.create_connection.return_value = store

        result = maybe_send_welcome_back(_utc(14, 0))

        assert result is False
        store.rpush.assert_not_called()

    @patch("capabilities.quiet_window.is_quiet_now", return_value=True)
    def test_skips_during_quiet_hours(self, _quiet):
        """Brief does not fire during quiet window."""
        from capabilities.welcome_back import maybe_send_welcome_back

        result = maybe_send_welcome_back(_utc(23, 0))

        assert result is False

    @patch("services.memory_client.MemoryClientService")
    def test_skips_when_no_hints(self, mock_mem):
        """Brief does not fire if no world-state hints are cached."""
        from capabilities.welcome_back import maybe_send_welcome_back

        store = _make_store(last_interaction_age=7200, hints=[])
        mock_mem.create_connection.return_value = store

        result = maybe_send_welcome_back(_utc(14, 0))

        assert result is False
        store.rpush.assert_not_called()
