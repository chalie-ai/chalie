"""Tests for :mod:`capabilities.urgent_inbox`."""

import datetime
import json
from unittest.mock import MagicMock, patch

import pytest

_UTC = datetime.timezone.utc
_NOW = datetime.datetime(2026, 3, 29, 14, 0, tzinfo=_UTC)

_RECENT_KNOWN = {
    "uid": 201, "subject": "Contract review",
    "from_name": "Sarah Chen", "from_addr": "sarah@example.com",
    "date": (_NOW - datetime.timedelta(minutes=5)).isoformat(),
    "triage": "actionable", "is_thread": False,
}
_OLD_EMAIL = {
    "uid": 202, "subject": "Lunch plans",
    "from_name": "Alex", "from_addr": "alex@example.com",
    "date": (_NOW - datetime.timedelta(hours=2)).isoformat(),
    "triage": "actionable", "is_thread": False,
}
_UNKNOWN_SENDER = {
    "uid": 203, "subject": "Newsletter",
    "from_name": "News Bot", "from_addr": "bot@spam.com",
    "date": (_NOW - datetime.timedelta(minutes=3)).isoformat(),
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
@patch("capabilities.contact_resolver.resolve")
@patch("capabilities.quiet_window.is_quiet_now", return_value=False)
@patch("capabilities.load_capabilities")
@patch("services.memory_client.MemoryClientService")
def test_fires_for_recent_known_contact(mock_mcs, mock_caps, _q, mock_resolve):
    store = MagicMock()
    store.get.return_value = None
    mock_mcs.create_connection.return_value = store
    mock_caps.return_value = {"imap": _mock_imap([_RECENT_KNOWN])}
    mock_resolve.return_value = [{"email": "sarah@example.com", "name": "Sarah Chen"}]

    from capabilities.urgent_inbox import maybe_send_urgent_inbox
    assert maybe_send_urgent_inbox(now=_NOW) is True
    assert "Sarah Chen" in json.loads(store.rpush.call_args[0][1])["prompt"]


@pytest.mark.unit
@patch("capabilities.contact_resolver.resolve")
@patch("capabilities.quiet_window.is_quiet_now", return_value=False)
@patch("capabilities.load_capabilities")
@patch("services.memory_client.MemoryClientService")
def test_skips_old_email(mock_mcs, mock_caps, _q, mock_resolve):
    store = MagicMock()
    store.get.return_value = None
    mock_mcs.create_connection.return_value = store
    mock_caps.return_value = {"imap": _mock_imap([_OLD_EMAIL])}
    mock_resolve.return_value = [{"email": "alex@example.com", "name": "Alex"}]

    from capabilities.urgent_inbox import maybe_send_urgent_inbox
    assert maybe_send_urgent_inbox(now=_NOW) is False


@pytest.mark.unit
@patch("capabilities.contact_resolver.resolve", return_value=[])
@patch("capabilities.quiet_window.is_quiet_now", return_value=False)
@patch("capabilities.load_capabilities")
@patch("services.memory_client.MemoryClientService")
def test_skips_unknown_sender(mock_mcs, mock_caps, _q, _resolve):
    store = MagicMock()
    store.get.return_value = None
    mock_mcs.create_connection.return_value = store
    mock_caps.return_value = {"imap": _mock_imap([_UNKNOWN_SENDER])}

    from capabilities.urgent_inbox import maybe_send_urgent_inbox
    assert maybe_send_urgent_inbox(now=_NOW) is False


@pytest.mark.unit
@patch("capabilities.contact_resolver.resolve")
@patch("capabilities.quiet_window.is_quiet_now", return_value=False)
@patch("capabilities.load_capabilities")
@patch("services.memory_client.MemoryClientService")
def test_dedup_by_uid(mock_mcs, mock_caps, _q, mock_resolve):
    seen = {"urgent_inbox:201": "1"}
    store = MagicMock()
    store.get.side_effect = seen.get
    mock_mcs.create_connection.return_value = store
    mock_caps.return_value = {"imap": _mock_imap([_RECENT_KNOWN])}
    mock_resolve.return_value = [{"email": "sarah@example.com", "name": "Sarah Chen"}]

    from capabilities.urgent_inbox import maybe_send_urgent_inbox
    assert maybe_send_urgent_inbox(now=_NOW) is False


@pytest.mark.unit
def test_suppressed_when_quiet_or_no_imap():
    from capabilities.urgent_inbox import maybe_send_urgent_inbox

    with patch("capabilities.quiet_window.is_quiet_now", return_value=True):
        assert maybe_send_urgent_inbox(now=_NOW) is False

    with patch("capabilities.quiet_window.is_quiet_now", return_value=False), \
         patch("capabilities.load_capabilities", return_value={}), \
         patch("services.memory_client.MemoryClientService") as mcs:
        mcs.create_connection.return_value = MagicMock()
        assert maybe_send_urgent_inbox(now=_NOW) is False
