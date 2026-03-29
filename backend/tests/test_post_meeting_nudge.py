"""Tests for :mod:`capabilities.post_meeting_nudge`."""

import datetime
import json
from unittest.mock import MagicMock, patch

import pytest

_UTC = datetime.timezone.utc
_NOW = datetime.datetime(2026, 3, 29, 14, 3, tzinfo=_UTC)


def _make_db_row(summary, dtstart, dtend, uid, attendees):
    meta = json.dumps({
        "uid": uid, "dtstart": dtstart.isoformat(),
        "dtend": dtend.isoformat(), "all_day": False,
        "attendees": attendees,
    })
    return (summary, dtstart.isoformat(), meta)


def _mock_conn(rows):
    """Build a mock DB connection returning *rows*."""
    conn = MagicMock()
    cur = MagicMock()
    cur.execute.return_value = cur
    cur.fetchall.return_value = rows
    conn.cursor.return_value = cur
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)
    return conn


@pytest.mark.unit
@patch("capabilities.load_capabilities")
@patch("services.memory_client.MemoryClientService")
@patch("services.database_service.get_shared_db_service")
def test_nudge_fires_after_meeting_with_unanswered(
    mock_db, mock_mcs, mock_lc,
):
    """Nudge fires when a meeting ended ~2 min ago with unanswered emails."""
    dtstart = _NOW - datetime.timedelta(minutes=32)
    dtend = _NOW - datetime.timedelta(minutes=2)
    row = _make_db_row("Standup", dtstart, dtend, "e1", ["a@x.com"])

    store = MagicMock()
    store.get.return_value = None
    mock_mcs.create_connection.return_value = store
    mock_db.return_value.connection.return_value = _mock_conn([row])

    imap_cap = MagicMock()
    imap_cap.is_connected.return_value = True
    imap_cap.get_tools.return_value = [
        {"name": "imap_search_email",
         "handler": MagicMock(return_value={
             "emails": [{"subject": "API creds",
                         "date": "2026-03-27T10:00:00"}]})}]
    mock_lc.return_value = {"imap": imap_cap}

    from capabilities.post_meeting_nudge import (
        maybe_send_post_meeting_nudge,
    )
    assert maybe_send_post_meeting_nudge(now=_NOW) is True
    payload = json.loads(store.rpush.call_args[0][1])
    assert "API creds" in payload["prompt"]
    store.setex.assert_called_once()


@pytest.mark.unit
@patch("capabilities.load_capabilities")
@patch("services.memory_client.MemoryClientService")
@patch("services.database_service.get_shared_db_service")
def test_nudge_skips_no_unanswered(mock_db, mock_mcs, mock_lc):
    """No nudge when attendees have no unanswered emails."""
    dtstart = _NOW - datetime.timedelta(minutes=32)
    dtend = _NOW - datetime.timedelta(minutes=2)
    row = _make_db_row("Standup", dtstart, dtend, "e2", ["a@x.com"])

    store = MagicMock()
    store.get.return_value = None
    mock_mcs.create_connection.return_value = store
    mock_db.return_value.connection.return_value = _mock_conn([row])

    imap_cap = MagicMock()
    imap_cap.is_connected.return_value = True
    imap_cap.get_tools.return_value = [
        {"name": "imap_search_email",
         "handler": MagicMock(return_value={"emails": []})}]
    mock_lc.return_value = {"imap": imap_cap}

    from capabilities.post_meeting_nudge import (
        maybe_send_post_meeting_nudge,
    )
    assert maybe_send_post_meeting_nudge(now=_NOW) is False
    store.rpush.assert_not_called()


@pytest.mark.unit
@patch("services.memory_client.MemoryClientService")
@patch("services.database_service.get_shared_db_service")
def test_nudge_skips_dedup(mock_db, mock_mcs):
    """Nudge is skipped when dedup flag already set."""
    dtstart = _NOW - datetime.timedelta(minutes=32)
    dtend = _NOW - datetime.timedelta(minutes=2)
    row = _make_db_row("Standup", dtstart, dtend, "e3", ["a@x.com"])

    store = MagicMock()
    store.get.return_value = "1"
    mock_mcs.create_connection.return_value = store
    mock_db.return_value.connection.return_value = _mock_conn([row])

    from capabilities.post_meeting_nudge import (
        maybe_send_post_meeting_nudge,
    )
    assert maybe_send_post_meeting_nudge(now=_NOW) is False
    store.rpush.assert_not_called()


@pytest.mark.unit
@patch("services.memory_client.MemoryClientService")
@patch("services.database_service.get_shared_db_service")
def test_nudge_skips_outside_window(mock_db, mock_mcs):
    """Nudge is skipped when meeting ended more than 5 minutes ago."""
    dtstart = _NOW - datetime.timedelta(hours=1, minutes=30)
    dtend = _NOW - datetime.timedelta(minutes=10)
    row = _make_db_row("Old meeting", dtstart, dtend, "e4", ["a@x.com"])

    store = MagicMock()
    store.get.return_value = None
    mock_mcs.create_connection.return_value = store
    mock_db.return_value.connection.return_value = _mock_conn([row])

    from capabilities.post_meeting_nudge import (
        maybe_send_post_meeting_nudge,
    )
    assert maybe_send_post_meeting_nudge(now=_NOW) is False
    store.rpush.assert_not_called()
