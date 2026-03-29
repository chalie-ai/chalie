"""Tests for :mod:`capabilities.evening_brief`."""

import datetime
import json
from unittest.mock import MagicMock, patch

import pytest

_UTC = datetime.timezone.utc
_EVENING = datetime.datetime(2026, 3, 28, 18, 0, tzinfo=_UTC)
_MORNING = datetime.datetime(2026, 3, 28, 9, 0, tzinfo=_UTC)


@pytest.mark.unit
@patch("capabilities.quiet_window.is_quiet_now", return_value=False)
@patch("capabilities.evening_brief._user_tz", return_value=_UTC)
@patch("capabilities.evening_brief._build_unanswered_section",
       return_value='Unanswered emails today (1):\n  - a@b: "Hi"')
@patch("capabilities.evening_brief._build_tomorrow_section",
       return_value="Tomorrow (1 events):\n  09:00 Standup")
@patch("services.memory_client.MemoryClientService")
def test_sends_brief_and_dedup(mock_mcs, _cal, _email, _tz, _quiet):
    """Brief fires in evening; dedup prevents second send."""
    store = MagicMock()
    store.get.return_value = None
    mock_mcs.create_connection.return_value = store

    from capabilities.evening_brief import maybe_send_evening_brief
    assert maybe_send_evening_brief(now=_EVENING) is True

    payload = json.loads(store.rpush.call_args[0][1])
    assert "EVENING BRIEF" in payload["prompt"]
    assert "Standup" in payload["prompt"]
    store.setex.assert_called_once()

    store.get.return_value = "1"
    assert maybe_send_evening_brief(now=_EVENING) is False


@pytest.mark.unit
@patch("capabilities.quiet_window.is_quiet_now", return_value=False)
@patch("capabilities.evening_brief._user_tz", return_value=_UTC)
def test_skips_before_evening(_tz, _quiet):
    """No brief before 17:00."""
    from capabilities.evening_brief import maybe_send_evening_brief
    assert maybe_send_evening_brief(now=_MORNING) is False
