"""Tests for :mod:`capabilities.morning_reply_drafts`."""

import datetime
import json
from unittest.mock import MagicMock, patch

import pytest

_UTC = datetime.timezone.utc
_NOW = datetime.datetime(2026, 3, 29, 8, 0, tzinfo=_UTC)

_RECENT_EMAIL = {
    "uid": 201, "subject": "Q3 roadmap draft",
    "from_name": "Sarah Chen", "from_addr": "sarah@example.com",
    "date": (_NOW - datetime.timedelta(hours=3)).isoformat(),
}
_OLD_EMAIL = {
    "uid": 202, "subject": "Old thread",
    "from_name": "Alex", "from_addr": "alex@example.com",
    "date": (_NOW - datetime.timedelta(hours=18)).isoformat(),
}
_SECOND_RECENT = {
    "uid": 203, "subject": "Deployment plan",
    "from_name": "Jordan", "from_addr": "jordan@example.com",
    "date": (_NOW - datetime.timedelta(hours=1)).isoformat(),
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
@patch("capabilities.hook_dedup.is_fired", return_value=False)
@patch("capabilities.quiet_window.is_quiet_now", return_value=False)
@patch("capabilities.load_capabilities")
@patch("services.memory_client.MemoryClientService")
def test_sends_drafts_for_recent_emails(mock_mcs, mock_caps, _q, _dedup):
    store = MagicMock()
    mock_mcs.create_connection.return_value = store
    mock_caps.return_value = {"imap": _mock_imap([_RECENT_EMAIL, _SECOND_RECENT])}

    from capabilities.morning_reply_drafts import maybe_send_morning_reply_drafts
    assert maybe_send_morning_reply_drafts(now=_NOW) is True

    payload = json.loads(store.rpush.call_args[0][1])
    assert "Sarah Chen" in payload["prompt"]
    assert "Jordan" in payload["prompt"]
    assert "MORNING REPLY DRAFTS" in payload["prompt"]
    assert payload["metadata"]["source"] == "morning_reply_drafts"


@pytest.mark.unit
@patch("capabilities.hook_dedup.is_fired", return_value=False)
@patch("capabilities.quiet_window.is_quiet_now", return_value=False)
@patch("capabilities.load_capabilities")
@patch("services.memory_client.MemoryClientService")
def test_persists_uid_mapping_in_memorystore(mock_mcs, mock_caps, _q, _dedup):
    """UID mapping must be stored so 'send 1' resolves to a real email."""
    store = MagicMock()
    mock_mcs.create_connection.return_value = store
    mock_caps.return_value = {"imap": _mock_imap([_RECENT_EMAIL, _SECOND_RECENT])}

    from capabilities.morning_reply_drafts import maybe_send_morning_reply_drafts
    assert maybe_send_morning_reply_drafts(now=_NOW) is True

    # Verify MemoryStore.set was called with the draft mapping
    set_call = store.set.call_args
    assert set_call is not None
    key, value = set_call[0][0], set_call[0][1]
    assert key == f"morning_drafts:{_NOW.strftime('%Y-%m-%d')}"
    draft_map = json.loads(value)
    assert len(draft_map) == 2
    assert draft_map[0]["uid"] == 201
    assert draft_map[0]["from_addr"] == "sarah@example.com"
    assert draft_map[1]["uid"] == 203
    assert draft_map[1]["from_addr"] == "jordan@example.com"


@pytest.mark.unit
@patch("capabilities.hook_dedup.is_fired", return_value=False)
@patch("capabilities.quiet_window.is_quiet_now", return_value=False)
@patch("capabilities.load_capabilities")
@patch("services.memory_client.MemoryClientService")
def test_prompt_includes_uid_mapping(mock_mcs, mock_caps, _q, _dedup):
    """The LLM prompt must contain UID refs so it can dispatch on 'send N'."""
    store = MagicMock()
    mock_mcs.create_connection.return_value = store
    mock_caps.return_value = {"imap": _mock_imap([_RECENT_EMAIL])}

    from capabilities.morning_reply_drafts import maybe_send_morning_reply_drafts
    assert maybe_send_morning_reply_drafts(now=_NOW) is True

    payload = json.loads(store.rpush.call_args[0][1])
    assert "UID:201" in payload["prompt"]
    assert "imap_draft_reply" in payload["prompt"]


@pytest.mark.unit
@patch("capabilities.hook_dedup.is_fired", return_value=False)
@patch("capabilities.quiet_window.is_quiet_now", return_value=False)
@patch("capabilities.load_capabilities")
@patch("services.memory_client.MemoryClientService")
def test_skips_old_emails(mock_mcs, mock_caps, _q, _dedup):
    store = MagicMock()
    mock_mcs.create_connection.return_value = store
    mock_caps.return_value = {"imap": _mock_imap([_OLD_EMAIL])}

    from capabilities.morning_reply_drafts import maybe_send_morning_reply_drafts
    assert maybe_send_morning_reply_drafts(now=_NOW) is False


@pytest.mark.unit
@patch("capabilities.quiet_window.is_quiet_now", return_value=False)
@patch("capabilities.hook_dedup.is_fired", return_value=True)
def test_dedup_and_quiet(_dedup, _q):
    from capabilities.morning_reply_drafts import maybe_send_morning_reply_drafts
    assert maybe_send_morning_reply_drafts(now=_NOW) is False

    with patch("capabilities.quiet_window.is_quiet_now", return_value=True), \
         patch("capabilities.hook_dedup.is_fired", return_value=False):
        assert maybe_send_morning_reply_drafts(now=_NOW) is False


@pytest.mark.unit
@patch("capabilities.hook_dedup.is_fired", return_value=False)
@patch("capabilities.quiet_window.is_quiet_now", return_value=False)
@patch("capabilities.load_capabilities")
@patch("services.memory_client.MemoryClientService")
def test_outside_morning_window_and_no_imap(mock_mcs, mock_caps, _q, _dedup):
    afternoon = datetime.datetime(2026, 3, 29, 15, 0, tzinfo=_UTC)
    store = MagicMock()
    mock_mcs.create_connection.return_value = store
    mock_caps.return_value = {"imap": _mock_imap([_RECENT_EMAIL])}

    from capabilities.morning_reply_drafts import maybe_send_morning_reply_drafts
    assert maybe_send_morning_reply_drafts(now=afternoon) is False

    mock_caps.return_value = {}
    assert maybe_send_morning_reply_drafts(now=_NOW) is False
