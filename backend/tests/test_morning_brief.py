"""Tests for :mod:`capabilities.morning_brief` — fused daily digest."""

import datetime
import json
from unittest.mock import MagicMock, patch

import pytest

_UTC = datetime.timezone.utc


def _make_event(**overrides):
    base = {
        'uid': 'evt-1', 'summary': 'Team standup',
        'dtstart': datetime.datetime(2026, 3, 28, 9, 0, tzinfo=_UTC),
        'dtend': datetime.datetime(2026, 3, 28, 9, 30, tzinfo=_UTC),
        'location': None, 'attendees': [], 'recurrence': None,
        'all_day': False, 'calendar_name': 'Work',
    }
    base.update(overrides)
    return base


class TestMorningBriefFusion:
    """Test cross-capability fusion (calendar + email in one brief)."""

    @pytest.mark.unit
    @patch('capabilities.morning_brief._read_cached_inbox_hint',
           return_value='Inbox: 3 actionable (top: boss@work.com), 5 informational.')
    @patch('services.memory_client.MemoryClientService')
    def test_fused_with_email(self, mock_mcs, _mock_hint):
        store = MagicMock()
        store.get.return_value = None
        mock_mcs.create_connection.return_value = store

        from capabilities.morning_brief import maybe_send_morning_brief
        now = datetime.datetime(2026, 3, 28, 6, 0, tzinfo=_UTC)
        result = maybe_send_morning_brief([_make_event()], now)

        assert result is True
        payload = json.loads(store.rpush.call_args[0][1])
        assert "Team standup" in payload['prompt']
        assert "Inbox: 3 actionable" in payload['prompt']
        assert "email highlights" in payload['prompt']

    @pytest.mark.unit
    @patch('capabilities.morning_brief._read_cached_inbox_hint', return_value='')
    @patch('services.memory_client.MemoryClientService')
    def test_calendar_only_when_no_email(self, mock_mcs, _mock_hint):
        store = MagicMock()
        store.get.return_value = None
        mock_mcs.create_connection.return_value = store

        from capabilities.morning_brief import maybe_send_morning_brief
        now = datetime.datetime(2026, 3, 28, 6, 0, tzinfo=_UTC)
        result = maybe_send_morning_brief([_make_event()], now)

        assert result is True
        payload = json.loads(store.rpush.call_args[0][1])
        assert "Team standup" in payload['prompt']
        assert "Inbox" not in payload['prompt']

    @pytest.mark.unit
    def test_inbox_hint_reads_imap_signal(self):
        store = MagicMock()
        store.lrange.return_value = [
            json.dumps({"source": "caldav", "content": "3 meetings"}),
            json.dumps({"source": "imap", "content": "Inbox: 2 actionable."}),
        ]
        from capabilities.morning_brief import _read_cached_inbox_hint
        assert _read_cached_inbox_hint(store) == "Inbox: 2 actionable."
