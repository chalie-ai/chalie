"""Tests for :mod:`capabilities.conflict_alert` -- conflict-alert proactive hook."""

import datetime
import json
from unittest.mock import MagicMock, patch

import pytest

_UTC = datetime.timezone.utc
_NOW = datetime.datetime(2026, 3, 29, 14, 0, tzinfo=_UTC)


def _make_db_mock(rows):
    """Return a mock get_shared_db_service with preset query results."""
    mock_db = MagicMock()
    conn = MagicMock()
    cur = MagicMock()
    cur.execute.return_value = cur
    cur.fetchall.return_value = rows
    conn.cursor.return_value = cur
    conn.__enter__ = lambda s: s
    conn.__exit__ = MagicMock(return_value=False)
    mock_db.return_value.connection.return_value = conn
    return mock_db


@pytest.mark.unit
@patch('capabilities.quiet_window.is_quiet_now', return_value=False)
@patch('services.memory_client.MemoryClientService')
@patch('services.database_service.get_shared_db_service')
def test_conflict_alert_fires(mock_db_fn, mock_mcs, _quiet):
    """Alert enqueued when an active conflict exists."""
    rows = [(
        'Schedule conflict: "Standup" and "Design Review" overlap',
        "caldav:conflict:uid1:uid2",
    )]
    mock_db_fn.side_effect = _make_db_mock(rows).side_effect
    mock_db_fn.return_value = _make_db_mock(rows).return_value

    store = MagicMock()
    store.get.return_value = None  # no dedup flag
    mock_mcs.create_connection.return_value = store

    from capabilities.conflict_alert import maybe_send_conflict_alert
    assert maybe_send_conflict_alert(now=_NOW) is True

    assert store.rpush.call_count == 1
    payload = json.loads(store.rpush.call_args[0][1])
    assert "Standup" in payload["prompt"]
    assert "Design Review" in payload["prompt"]
    assert payload["metadata"]["source"] == "conflict_alert"
    assert payload["metadata"]["canon_key"] == "uid1:uid2"
    store.setex.assert_called_once()


@pytest.mark.unit
@patch('capabilities.quiet_window.is_quiet_now', return_value=False)
@patch('services.memory_client.MemoryClientService')
@patch('services.database_service.get_shared_db_service')
def test_conflict_alert_dedup(mock_db_fn, mock_mcs, _quiet):
    """Same conflict pair does not fire twice on the same day."""
    rows = [(
        'Schedule conflict: "A" and "B" overlap',
        "caldav:conflict:uid1:uid2",
    )]
    mock_db_fn.side_effect = _make_db_mock(rows).side_effect
    mock_db_fn.return_value = _make_db_mock(rows).return_value

    store = MagicMock()
    store.get.return_value = "1"  # dedup flag already set
    mock_mcs.create_connection.return_value = store

    from capabilities.conflict_alert import maybe_send_conflict_alert
    assert maybe_send_conflict_alert(now=_NOW) is False
    store.rpush.assert_not_called()


@pytest.mark.unit
@patch('capabilities.quiet_window.is_quiet_now', return_value=True)
def test_conflict_alert_quiet_window(_quiet):
    """No alert during quiet hours."""
    from capabilities.conflict_alert import maybe_send_conflict_alert
    assert maybe_send_conflict_alert(now=_NOW) is False


@pytest.mark.unit
@patch('capabilities.quiet_window.is_quiet_now', return_value=False)
@patch('services.memory_client.MemoryClientService')
@patch('services.database_service.get_shared_db_service')
def test_conflict_alert_no_conflicts(mock_db_fn, mock_mcs, _quiet):
    """No alert when no conflicts exist."""
    mock_db_fn.side_effect = _make_db_mock([]).side_effect
    mock_db_fn.return_value = _make_db_mock([]).return_value

    store = MagicMock()
    mock_mcs.create_connection.return_value = store

    from capabilities.conflict_alert import maybe_send_conflict_alert
    assert maybe_send_conflict_alert(now=_NOW) is False
    store.rpush.assert_not_called()
