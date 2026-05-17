"""
Tests for backend/services/scheduler_service.py

Covers _next_due (recurrence mapping), _in_quiet_hours, and _fire_item.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
import services.scheduler_service as svc


@pytest.mark.unit
class TestNextDue:

    def test_simple_recurrences(self):
        base = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
        assert svc._next_due({"due_at": base, "recurrence": "daily"}).day == 16
        assert svc._next_due({"due_at": base, "recurrence": "weekly"}).day == 22
        assert svc._next_due({"due_at": base, "recurrence": "hourly"}).hour == 11

    def test_monthly_clamps_day(self):
        item = {"due_at": datetime(2024, 1, 31, 10, 0, tzinfo=timezone.utc), "recurrence": "monthly"}
        result = svc._next_due(item)
        assert result.month == 2 and result.day == 29

    def test_interval_minutes(self):
        item = {"due_at": datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc), "recurrence": "interval:30"}
        assert svc._next_due(item).minute == 30

    def test_none_and_unknown(self):
        assert svc._next_due({"due_at": "2024-01-15T10:00:00+00:00", "recurrence": None}) is None
        assert svc._next_due({"due_at": "2024-01-15T10:00:00+00:00", "recurrence": "unknown"}) is None


@pytest.mark.unit
class TestInQuietHours:

    def _patch_local_now(self, hour):
        return patch("services.locale_service.local_now",
                     return_value=datetime(2024, 6, 15, hour, 0, tzinfo=timezone.utc))

    def test_no_window_not_quiet(self):
        assert svc._in_quiet_hours({}) is False

    def test_inside_vs_outside_window(self):
        item = {"window_start": "09:00", "window_end": "17:00"}
        with self._patch_local_now(12):
            assert svc._in_quiet_hours(item) is False
        with self._patch_local_now(20):
            assert svc._in_quiet_hours(item) is True


@pytest.mark.unit
class TestFireItem:

    def _make_item(self, item_type='notification', is_prompt=False, message='Test', **extra):
        item = {"id": "test1", "item_type": item_type, "message": message,
                "due_at": datetime(2024, 1, 15, 9, 0, tzinfo=timezone.utc),
                "topic": "general", "is_prompt": is_prompt}
        item.update(extra)
        return item

    def test_notification_uses_output_service(self):
        with patch('services.output_service.OutputService') as mock_cls:
            mock_output = MagicMock()
            mock_cls.return_value = mock_output
            svc._fire_item(self._make_item())
            assert mock_output.enqueue_text.call_args[1]["mode"] == "NOTIFICATION"

    def test_prompt_spawns_thread(self):
        mock_t = MagicMock()
        with patch('threading.Thread', return_value=mock_t):
            svc._fire_item(self._make_item('prompt', is_prompt=True, message='Do it'))
        mock_t.start.assert_called_once()

    def test_system_dispatches_handler(self):
        handler = MagicMock()
        svc.register_system_handler('test-src', handler)
        svc._fire_item(self._make_item('system', channel='test-src'))
        handler.assert_called_once()
