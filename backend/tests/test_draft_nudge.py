"""Tests for :mod:`capabilities.draft_nudge`."""

import datetime
import json
from unittest.mock import MagicMock, patch

import pytest

_UTC = datetime.timezone.utc
_NOW = datetime.datetime(2026, 3, 29, 14, 0, tzinfo=_UTC)

_STALE_EMAIL = {
    "uid": 101, "subject": "Q3 budget review",
    "from_name": "Sarah Chen", "from_addr": "sarah@example.com",
    "date": (_NOW - datetime.timedelta(hours=6)).isoformat(),
    "triage": "actionable", "is_thread": False,
}
_FRESH_EMAIL = {
    "uid": 102, "subject": "Lunch plans",
    "from_name": "Alex", "from_addr": "alex@example.com",
    "date": (_NOW - datetime.timedelta(hours=1)).isoformat(),
    "triage": "actionable", "is_thread": False,
}


def _mock_imap(emails):
    cap = MagicMock()
    cap.is_connected.return_value = True
    cap.get_tools.return_value = [
        {"name": "imap_search_email",
         "handler": MagicMock(return_value={"emails": emails})},
    ]
    return cap


@pytest.mark.unit
@patch("capabilities.quiet_window.is_quiet_now", return_value=False)
@patch("capabilities.load_capabilities")
@patch("services.memory_client.MemoryClientService")
def test_sends_nudge_for_stale(mock_mcs, mock_caps, _q):
    store = MagicMock()
    store.get.return_value = None
    mock_mcs.create_connection.return_value = store
    mock_caps.return_value = {"imap": _mock_imap([_STALE_EMAIL])}

    from capabilities.draft_nudge import maybe_send_draft_nudge
    assert maybe_send_draft_nudge(now=_NOW) is True
    assert "Sarah Chen" in json.loads(store.rpush.call_args[0][1])["prompt"]


@pytest.mark.unit
@patch("capabilities.quiet_window.is_quiet_now", return_value=False)
@patch("capabilities.load_capabilities")
@patch("services.memory_client.MemoryClientService")
def test_skips_fresh(mock_mcs, mock_caps, _q):
    store = MagicMock()
    store.get.return_value = None
    mock_mcs.create_connection.return_value = store
    mock_caps.return_value = {"imap": _mock_imap([_FRESH_EMAIL])}

    from capabilities.draft_nudge import maybe_send_draft_nudge
    assert maybe_send_draft_nudge(now=_NOW) is False


@pytest.mark.unit
@patch("capabilities.quiet_window.is_quiet_now", return_value=False)
@patch("services.memory_client.MemoryClientService")
def test_dedup(mock_mcs, _q):
    store = MagicMock()
    store.get.return_value = "1"
    mock_mcs.create_connection.return_value = store

    from capabilities.draft_nudge import maybe_send_draft_nudge
    assert maybe_send_draft_nudge(now=_NOW) is False


@pytest.mark.unit
@patch("capabilities.quiet_window.is_quiet_now", return_value=True)
def test_quiet_suppresses(_q):
    from capabilities.draft_nudge import maybe_send_draft_nudge
    assert maybe_send_draft_nudge(now=_NOW) is False


@pytest.mark.unit
@patch("capabilities.quiet_window.is_quiet_now", return_value=False)
@patch("capabilities.load_capabilities", return_value={})
@patch("services.memory_client.MemoryClientService")
def test_no_imap(mock_mcs, _caps, _q):
    store = MagicMock()
    store.get.return_value = None
    mock_mcs.create_connection.return_value = store

    from capabilities.draft_nudge import maybe_send_draft_nudge
    assert maybe_send_draft_nudge(now=_NOW) is False
