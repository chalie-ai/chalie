"""Tests for :mod:`capabilities.morning_brief` — fused daily digest."""

import datetime
import json
from unittest.mock import MagicMock, patch

import pytest

_UTC = datetime.timezone.utc


def _make_event(**overrides):
    base = {
        'summary': 'Team standup',
        'dtstart': datetime.datetime(2026, 3, 28, 9, 0, tzinfo=_UTC),
        'dtend': datetime.datetime(2026, 3, 28, 9, 30, tzinfo=_UTC),
        'location': None,
    }
    base.update(overrides)
    return base


@patch('capabilities.morning_brief._user_tz', return_value=_UTC)
@patch('capabilities.quiet_window.is_quiet_now', return_value=False)
class TestMorningBriefFusion:
    """Test cross-capability fusion (calendar + email in one brief)."""

    @pytest.mark.unit
    @patch('capabilities.morning_brief._read_cached_inbox_hint',
           return_value='Inbox: 3 actionable (top: boss@work.com), '
                        '5 informational.')
    @patch('capabilities.morning_brief._fetch_today_events')
    @patch('services.memory_client.MemoryClientService')
    def test_fused_with_email(self, mock_mcs, mock_fetch, _hint,
                              _quiet, _tz):
        mock_fetch.return_value = [_make_event()]
        store = MagicMock()
        store.get.return_value = None
        mock_mcs.create_connection.return_value = store

        from capabilities.morning_brief import maybe_send_morning_brief
        now = datetime.datetime(2026, 3, 28, 7, 0, tzinfo=_UTC)
        result = maybe_send_morning_brief(now)

        assert result is True
        payload = json.loads(store.rpush.call_args[0][1])
        assert "Team standup" in payload['prompt']
        assert "Inbox: 3 actionable" in payload['prompt']
        assert "email highlights" in payload['prompt']

    @pytest.mark.unit
    @patch('capabilities.morning_brief._read_cached_inbox_hint',
           return_value='')
    @patch('capabilities.morning_brief._fetch_today_events')
    @patch('services.memory_client.MemoryClientService')
    def test_calendar_only_when_no_email(self, mock_mcs, mock_fetch,
                                         _hint, _quiet, _tz):
        mock_fetch.return_value = [_make_event()]
        store = MagicMock()
        store.get.return_value = None
        mock_mcs.create_connection.return_value = store

        from capabilities.morning_brief import maybe_send_morning_brief
        now = datetime.datetime(2026, 3, 28, 7, 0, tzinfo=_UTC)
        result = maybe_send_morning_brief(now)

        assert result is True
        payload = json.loads(store.rpush.call_args[0][1])
        assert "Team standup" in payload['prompt']
        assert "Inbox" not in payload['prompt']

    @pytest.mark.unit
    @patch('capabilities.morning_brief._read_cached_inbox_hint',
           return_value='')
    @patch('capabilities.morning_brief._fetch_today_events')
    @patch('services.memory_client.MemoryClientService')
    def test_back_to_back_included_in_brief(self, mock_mcs, mock_fetch,
                                             _hint, _quiet, _tz):
        mock_fetch.return_value = [
            _make_event(
                summary='Standup',
                dtstart=datetime.datetime(2026, 3, 28, 9, 0, tzinfo=_UTC),
                dtend=datetime.datetime(2026, 3, 28, 9, 30, tzinfo=_UTC),
            ),
            _make_event(
                summary='Design Review',
                dtstart=datetime.datetime(2026, 3, 28, 9, 32, tzinfo=_UTC),
                dtend=datetime.datetime(2026, 3, 28, 10, 0, tzinfo=_UTC),
            ),
        ]
        store = MagicMock()
        store.get.return_value = None
        mock_mcs.create_connection.return_value = store

        from capabilities.morning_brief import maybe_send_morning_brief
        now = datetime.datetime(2026, 3, 28, 7, 0, tzinfo=_UTC)
        result = maybe_send_morning_brief(now)

        assert result is True
        payload = json.loads(store.rpush.call_args[0][1])
        prompt = payload['prompt']
        assert "Tight transitions" in prompt and "Standup" in prompt

    @pytest.mark.unit
    def test_inbox_hint_reads_imap_signal(self, _quiet, _tz):
        store = MagicMock()
        store.lrange.return_value = [
            json.dumps({"source": "caldav", "content": "3 meetings"}),
            json.dumps({"source": "imap", "content": "Inbox: 2 actionable."}),
        ]
        from capabilities.morning_brief import _read_cached_inbox_hint
        assert _read_cached_inbox_hint(store) == "Inbox: 2 actionable."

    @pytest.mark.unit
    @patch('services.memory_client.MemoryClientService')
    def test_outside_morning_window_skipped(self, mock_mcs, _quiet, _tz):
        """Morning brief only fires during 06:00-10:00 local."""
        from capabilities.morning_brief import maybe_send_morning_brief
        # 3 AM — too early
        assert maybe_send_morning_brief(
            datetime.datetime(2026, 3, 28, 3, 0, tzinfo=_UTC)) is False
        # 11 AM — too late
        assert maybe_send_morning_brief(
            datetime.datetime(2026, 3, 28, 11, 0, tzinfo=_UTC)) is False
        # MemoryClientService should NOT have been called (early exit)
        mock_mcs.create_connection.assert_not_called()
