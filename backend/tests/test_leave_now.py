"""Tests for :mod:`capabilities.leave_now`."""

import datetime
import json
from unittest.mock import MagicMock, patch

import pytest

_UTC = datetime.timezone.utc
_NOW = datetime.datetime(2026, 3, 29, 10, 0, tzinfo=_UTC)
_EVENT_IN_20 = (_NOW + datetime.timedelta(minutes=20)).isoformat()
_EVENT_IN_5 = (_NOW + datetime.timedelta(minutes=5)).isoformat()
_EVENT_IN_60 = (_NOW + datetime.timedelta(minutes=60)).isoformat()


def _meta(uid="uid-1", location="WeWork Sliema", all_day=False, summary="Team Sync"):
    return json.dumps({
        "uid": uid, "location": location, "all_day": all_day, "summary": summary,
    })


@pytest.mark.unit
@patch("capabilities.quiet_window.is_quiet_now", return_value=False)
@patch("capabilities.hook_dedup.is_fired", return_value=False)
@patch("capabilities.hook_dedup.mark_fired")
@patch("services.memory_client.MemoryClientService")
@patch("services.database_service.get_shared_db_service")
def test_fires_for_event_with_location(mock_db, mock_mcs, mock_mark, mock_is, _quiet):
    """Nudge enqueued for an event with a location in the departure window."""
    store = MagicMock()
    mock_mcs.create_connection.return_value = store

    conn = MagicMock()
    conn.cursor.return_value.execute.return_value.fetchall.return_value = [
        ("Team Sync @ WeWork Sliema", _EVENT_IN_20, _meta()),
    ]
    mock_db.return_value.connection.return_value.__enter__ = lambda s: conn
    mock_db.return_value.connection.return_value.__exit__ = MagicMock(return_value=False)

    from capabilities.leave_now import maybe_send_leave_now
    assert maybe_send_leave_now(now=_NOW) is True

    payload = json.loads(store.rpush.call_args[0][1])
    assert "LEAVE NOW" in payload["prompt"]
    assert "WeWork Sliema" in payload["prompt"]
    assert "Team Sync" in payload["prompt"]
    mock_mark.assert_called_once()


@pytest.mark.unit
@patch("capabilities.quiet_window.is_quiet_now", return_value=False)
@patch("capabilities.hook_dedup.is_fired", return_value=False)
@patch("services.memory_client.MemoryClientService")
@patch("services.database_service.get_shared_db_service")
def test_skips_no_location(mock_db, mock_mcs, mock_is, _quiet):
    """Events without a location are skipped."""
    store = MagicMock()
    mock_mcs.create_connection.return_value = store

    conn = MagicMock()
    conn.cursor.return_value.execute.return_value.fetchall.return_value = [
        ("Standup", _EVENT_IN_20, _meta(location="")),
    ]
    mock_db.return_value.connection.return_value.__enter__ = lambda s: conn
    mock_db.return_value.connection.return_value.__exit__ = MagicMock(return_value=False)

    from capabilities.leave_now import maybe_send_leave_now
    assert maybe_send_leave_now(now=_NOW) is False
    store.rpush.assert_not_called()


@pytest.mark.unit
@patch("capabilities.quiet_window.is_quiet_now", return_value=False)
@patch("capabilities.hook_dedup.is_fired", return_value=False)
@patch("services.memory_client.MemoryClientService")
@patch("services.database_service.get_shared_db_service")
def test_skips_all_day(mock_db, mock_mcs, mock_is, _quiet):
    """All-day events are skipped even if they have a location."""
    store = MagicMock()
    mock_mcs.create_connection.return_value = store

    conn = MagicMock()
    conn.cursor.return_value.execute.return_value.fetchall.return_value = [
        ("Conference", _EVENT_IN_20, _meta(all_day=True)),
    ]
    mock_db.return_value.connection.return_value.__enter__ = lambda s: conn
    mock_db.return_value.connection.return_value.__exit__ = MagicMock(return_value=False)

    from capabilities.leave_now import maybe_send_leave_now
    assert maybe_send_leave_now(now=_NOW) is False
    store.rpush.assert_not_called()


@pytest.mark.unit
@patch("capabilities.quiet_window.is_quiet_now", return_value=False)
@patch("capabilities.hook_dedup.is_fired", return_value=True)
@patch("services.memory_client.MemoryClientService")
@patch("services.database_service.get_shared_db_service")
def test_dedup_prevents_repeat(mock_db, mock_mcs, mock_is, _quiet):
    """Second call for the same event is suppressed by dedup."""
    store = MagicMock()
    mock_mcs.create_connection.return_value = store

    conn = MagicMock()
    conn.cursor.return_value.execute.return_value.fetchall.return_value = [
        ("Team Sync @ WeWork", _EVENT_IN_20, _meta()),
    ]
    mock_db.return_value.connection.return_value.__enter__ = lambda s: conn
    mock_db.return_value.connection.return_value.__exit__ = MagicMock(return_value=False)

    from capabilities.leave_now import maybe_send_leave_now
    assert maybe_send_leave_now(now=_NOW) is False
    store.rpush.assert_not_called()


@pytest.mark.unit
@patch("capabilities.quiet_window.is_quiet_now", return_value=True)
def test_quiet_window_suppresses(_quiet):
    """No nudges during quiet window."""
    from capabilities.leave_now import maybe_send_leave_now
    assert maybe_send_leave_now(now=_NOW) is False
