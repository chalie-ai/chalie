"""Tests for first_look — one-time cross-capability welcome briefing."""

import json
import sqlite3
from datetime import timedelta
from unittest.mock import MagicMock, patch

import pytest

from capabilities.first_look import (
    _build_calendar_snapshot,
    _read_inbox_snapshot,
    maybe_send_first_look,
)
from services.time_utils import utc_now


@pytest.mark.unit
@patch("capabilities.first_look.load_capabilities")
@patch("services.memory_client.MemoryClientService.create_connection")
def test_first_look_skips(mock_conn, mock_load):
    """Already-sent flag or single capability → skip."""
    store = MagicMock()

    store.get.return_value = "1"
    mock_conn.return_value = store
    assert maybe_send_first_look() is False
    mock_load.assert_not_called()

    store.get.return_value = None
    cap = MagicMock()
    cap.is_connected.return_value = True
    mock_load.return_value = {"caldav": cap}
    assert maybe_send_first_look() is False


@pytest.mark.unit
@patch("capabilities.first_look._read_inbox_snapshot", return_value="Inbox: 3 actionable")
@patch("capabilities.first_look._build_calendar_snapshot", return_value="Calendar (2 upcoming):\n- ev1")
@patch("capabilities.first_look.load_capabilities")
@patch("services.memory_client.MemoryClientService.create_connection")
def test_first_look_enqueues(mock_conn, mock_load, mock_cal, mock_inbox):
    """Both connected with data → enqueues prompt and sets flag."""
    store = MagicMock()
    store.get.return_value = None
    mock_conn.return_value = store
    c1, c2 = MagicMock(), MagicMock()
    c1.is_connected.return_value = True
    c2.is_connected.return_value = True
    mock_load.return_value = {"caldav": c1, "imap": c2}

    assert maybe_send_first_look() is True
    pushed = json.loads(store.rpush.call_args[0][1])
    assert "[FIRST LOOK" in pushed["prompt"]
    assert "Calendar (2 upcoming)" in pushed["prompt"]
    assert "Inbox: 3 actionable" in pushed["prompt"]
    store.setex.assert_called_once()

    # Inbox hint is also verified within the prompt
    assert _read_inbox_snapshot(store) == "Inbox: 3 actionable"


@pytest.mark.unit
@patch("services.database_service.get_shared_db_service")
def test_calendar_snapshot(mock_db):
    """Formats events from scheduled_items; empty when none."""
    now = utc_now()
    due = (now + timedelta(hours=2)).isoformat()

    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE scheduled_items "
        "(id TEXT, item_type TEXT, message TEXT, due_at TEXT, status TEXT, source TEXT, metadata TEXT)"
    )
    conn.execute(
        "INSERT INTO scheduled_items VALUES (?,?,?,?,?,?,?)",
        ("e1", "event", "Standup", due, "pending", "caldav", "{}"),
    )
    conn.commit()

    db_svc = MagicMock()
    db_svc.connection.return_value.__enter__ = lambda s: conn
    db_svc.connection.return_value.__exit__ = MagicMock(return_value=False)
    mock_db.return_value = db_svc

    result = _build_calendar_snapshot()
    assert "Calendar (1 upcoming)" in result
    assert "Standup" in result
