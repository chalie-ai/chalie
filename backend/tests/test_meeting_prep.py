"""Tests for :mod:`capabilities.meeting_prep` -- pre-meeting brief."""

import datetime
import json
from unittest.mock import MagicMock, patch

import pytest

_UTC = datetime.timezone.utc
_NOW = datetime.datetime(2026, 3, 28, 13, 45, tzinfo=_UTC)


@pytest.mark.unit
@patch('capabilities.load_capabilities', return_value={})
@patch('capabilities.contact_resolver.resolve',
       return_value=[{"email": "s@x.com", "name": "Sarah"}])
@patch('services.memory_client.MemoryClientService')
@patch('services.database_service.get_shared_db_service')
def test_meeting_prep_sends_brief(mock_db, mock_mcs, _r, _lc):
    dtstart = _NOW + datetime.timedelta(minutes=15)
    meta = json.dumps({"uid": "e1", "dtstart": dtstart.isoformat(),
                        "location": "Room 3", "all_day": False,
                        "attendees": ["s@x.com"]})
    store = MagicMock()
    store.get.return_value = None
    mock_mcs.create_connection.return_value = store
    conn = MagicMock()
    cur = MagicMock()
    cur.execute.return_value = cur
    cur.fetchall.return_value = [("Budget Review [Work]", dtstart.isoformat(), meta)]
    conn.cursor.return_value = cur
    conn.__enter__ = lambda s: s
    conn.__exit__ = MagicMock(return_value=False)
    mock_db.return_value.connection.return_value = conn

    from capabilities.meeting_prep import maybe_send_meeting_prep
    assert maybe_send_meeting_prep(now=_NOW) is True
    p = json.loads(store.rpush.call_args[0][1])
    assert "Budget Review" in p["prompt"]
    assert "Room 3" in p["prompt"]
    assert p["metadata"]["source"] == "meeting_prep"


@pytest.mark.unit
@patch('capabilities.load_capabilities')
@patch('capabilities.contact_resolver.resolve',
       return_value=[{"email": "s@x.com", "name": "Sarah"}])
@patch('services.memory_client.MemoryClientService')
@patch('services.database_service.get_shared_db_service')
def test_meeting_prep_includes_needs_reply(mock_db, mock_mcs, _r, mock_lc):
    """Brief includes 'Needs reply' when attendee has unanswered emails."""
    dtstart = _NOW + datetime.timedelta(minutes=15)
    meta = json.dumps({"uid": "e2", "dtstart": dtstart.isoformat(),
                        "location": "", "all_day": False,
                        "attendees": ["s@x.com"]})
    store = MagicMock()
    store.get.return_value = None
    mock_mcs.create_connection.return_value = store
    conn = MagicMock()
    cur = MagicMock()
    cur.execute.return_value = cur
    cur.fetchall.return_value = [("Budget Review", dtstart.isoformat(), meta)]
    conn.cursor.return_value = cur
    conn.__enter__ = lambda s: s
    conn.__exit__ = MagicMock(return_value=False)
    mock_db.return_value.connection.return_value = conn

    # Mock IMAP capability with search tool that returns unanswered emails
    def _fake_search(_topic, params):
        if params.get("unanswered"):
            return {"emails": [{"subject": "Q2 Budget", "date": "2026-03-27T10:00:00"}]}
        return {"emails": [{"subject": "Q2 Budget"}]}

    imap_cap = MagicMock()
    imap_cap.is_connected.return_value = True
    imap_cap.get_tools.return_value = [
        {"name": "imap_search_email", "handler": _fake_search}]
    mock_lc.return_value = {"imap": imap_cap}

    from capabilities.meeting_prep import maybe_send_meeting_prep
    assert maybe_send_meeting_prep(now=_NOW) is True
    prompt = json.loads(store.rpush.call_args[0][1])["prompt"]
    assert "Needs reply" in prompt
    assert "Q2 Budget" in prompt


@pytest.mark.unit
@patch('capabilities.load_capabilities')
@patch('capabilities.contact_resolver.resolve',
       return_value=[{"email": "s@x.com", "name": "Sarah"}])
@patch('services.memory_client.MemoryClientService')
@patch('services.database_service.get_shared_db_service')
def test_meeting_prep_no_needs_reply_when_all_answered(mock_db, mock_mcs, _r, mock_lc):
    """Brief omits 'Needs reply' section when no unanswered emails."""
    dtstart = _NOW + datetime.timedelta(minutes=15)
    meta = json.dumps({"uid": "e3", "dtstart": dtstart.isoformat(),
                        "location": "", "all_day": False,
                        "attendees": ["s@x.com"]})
    store = MagicMock()
    store.get.return_value = None
    mock_mcs.create_connection.return_value = store
    conn = MagicMock()
    cur = MagicMock()
    cur.execute.return_value = cur
    cur.fetchall.return_value = [("Standup", dtstart.isoformat(), meta)]
    conn.cursor.return_value = cur
    conn.__enter__ = lambda s: s
    conn.__exit__ = MagicMock(return_value=False)
    mock_db.return_value.connection.return_value = conn

    def _fake_search(_topic, params):
        if params.get("unanswered"):
            return {"emails": []}  # all replied to
        return {"emails": [{"subject": "Notes"}]}

    imap_cap = MagicMock()
    imap_cap.is_connected.return_value = True
    imap_cap.get_tools.return_value = [
        {"name": "imap_search_email", "handler": _fake_search}]
    mock_lc.return_value = {"imap": imap_cap}

    from capabilities.meeting_prep import maybe_send_meeting_prep
    assert maybe_send_meeting_prep(now=_NOW) is True
    prompt = json.loads(store.rpush.call_args[0][1])["prompt"]
    assert "Needs reply" not in prompt
    assert "Notes" in prompt


@pytest.mark.unit
@patch('services.memory_client.MemoryClientService')
@patch('services.database_service.get_shared_db_service')
def test_meeting_prep_skips(mock_db, mock_mcs):
    dtstart = _NOW + datetime.timedelta(minutes=15)
    meta_dedup = json.dumps({"uid": "e1", "dtstart": dtstart.isoformat(),
                              "all_day": False, "attendees": ["a@b.com"]})
    meta_empty = json.dumps({"uid": "e2", "dtstart": dtstart.isoformat(),
                              "all_day": False, "attendees": []})
    store = MagicMock()
    # First call: dedup flag set; second call: no flag but no attendees
    store.get.side_effect = ["1", None]
    mock_mcs.create_connection.return_value = store
    conn = MagicMock()
    cur = MagicMock()
    cur.execute.return_value = cur
    cur.fetchall.return_value = [
        ("Meeting A", dtstart.isoformat(), meta_dedup),
        ("Meeting B", dtstart.isoformat(), meta_empty),
    ]
    conn.cursor.return_value = cur
    conn.__enter__ = lambda s: s
    conn.__exit__ = MagicMock(return_value=False)
    mock_db.return_value.connection.return_value = conn

    from capabilities.meeting_prep import maybe_send_meeting_prep
    assert maybe_send_meeting_prep(now=_NOW) is False
    store.rpush.assert_not_called()
