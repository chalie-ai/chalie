"""
Tests for backend/services/scheduler_service.py

Covers pure-computation functions (_calculate_next_due, _build_recurrence, _fire_item).
Poll/DB tests require SQLite and are covered by integration tests.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
import services.scheduler_service as scheduler_svc


@pytest.mark.unit
class TestCalculateNextDue:
    """Test _calculate_next_due recurrence logic."""

    def test_daily(self):
        due_at = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
        next_due = scheduler_svc._calculate_next_due(due_at, "daily")
        assert next_due == datetime(2024, 1, 16, 10, 0, tzinfo=timezone.utc)

    def test_weekly(self):
        due_at = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
        next_due = scheduler_svc._calculate_next_due(due_at, "weekly")
        assert next_due == datetime(2024, 1, 22, 10, 0, tzinfo=timezone.utc)

    def test_monthly(self):
        due_at = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
        next_due = scheduler_svc._calculate_next_due(due_at, "monthly")
        assert next_due == datetime(2024, 2, 15, 10, 0, tzinfo=timezone.utc)

    def test_monthly_clamping(self):
        """Jan 31 → Feb should clamp to Feb 29 in leap year."""
        due_at = datetime(2024, 1, 31, 10, 0, tzinfo=timezone.utc)
        next_due = scheduler_svc._calculate_next_due(due_at, "monthly")
        assert next_due.month == 2
        assert next_due.day == 29  # 2024 is leap year

    def test_weekdays_skips_weekend(self):
        """Friday should advance to Monday."""
        friday = datetime(2024, 1, 19, 10, 0, tzinfo=timezone.utc)
        next_due = scheduler_svc._calculate_next_due(friday, "weekdays")
        assert next_due.weekday() == 0  # Monday

    def test_interval(self):
        due_at = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
        next_due = scheduler_svc._calculate_next_due(due_at, "interval:30")
        assert next_due == datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc)

    def test_interval_60_min(self):
        due_at = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
        next_due = scheduler_svc._calculate_next_due(due_at, "interval:60")
        assert next_due == datetime(2024, 1, 15, 11, 0, tzinfo=timezone.utc)

    def test_hourly_no_window(self):
        due_at = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
        next_due = scheduler_svc._calculate_next_due(due_at, "hourly")
        assert next_due == datetime(2024, 1, 15, 11, 0, tzinfo=timezone.utc)

    def test_hourly_within_window(self):
        due_at = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
        next_due = scheduler_svc._calculate_next_due(due_at, "hourly", "09:00", "17:00")
        assert next_due == datetime(2024, 1, 15, 11, 0, tzinfo=timezone.utc)

    def test_hourly_past_window_end(self):
        """Past window end should advance to next day's window start."""
        due_at = datetime(2024, 1, 15, 17, 0, tzinfo=timezone.utc)
        next_due = scheduler_svc._calculate_next_due(due_at, "hourly", "09:00", "17:00")
        assert next_due.day == 16
        assert next_due.hour == 9

    def test_unknown_recurrence_returns_none(self):
        due_at = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
        result = scheduler_svc._calculate_next_due(due_at, "unknown")
        assert result is None


@pytest.mark.unit
class TestBuildRecurrence:
    """Test _build_recurrence next-occurrence generation."""

    def test_no_recurrence_returns_none(self):
        item = {
            "id": "test1", "item_type": "notification", "message": "hello",
            "due_at": datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc),
            "recurrence": None, "topic": "general", "created_by_session": None,
            "group_id": "test1", "is_prompt": False,
        }
        result = scheduler_svc._build_recurrence(item, datetime.now(timezone.utc))
        assert result is None

    def test_daily_next_occurrence(self):
        item = {
            "id": "abc12345", "item_type": "notification", "message": "Daily standup",
            "due_at": datetime(2024, 1, 15, 9, 0, tzinfo=timezone.utc),
            "recurrence": "daily", "topic": "work", "created_by_session": None,
            "group_id": "abc12345", "is_prompt": False,
        }
        result = scheduler_svc._build_recurrence(item, datetime(2024, 1, 15, 9, 1, tzinfo=timezone.utc))
        assert result is not None
        assert result["due_at"] == datetime(2024, 1, 16, 9, 0, tzinfo=timezone.utc)
        assert result["group_id"] == "abc12345"
        assert result["id"] != "abc12345"

    def test_prompt_type_preserved(self):
        item = {
            "id": "xyz99", "item_type": "prompt", "message": "Check my progress",
            "due_at": datetime(2024, 1, 15, 9, 0, tzinfo=timezone.utc),
            "recurrence": "daily", "topic": "goals", "created_by_session": None,
            "group_id": "xyz99", "is_prompt": True,
        }
        result = scheduler_svc._build_recurrence(item, datetime(2024, 1, 15, 9, 1, tzinfo=timezone.utc))
        assert result["item_type"] == "prompt"
        assert result["is_prompt"] is True


@pytest.mark.unit
class TestFireItem:
    """Test _fire_item delivery routing."""

    def _make_item(self, item_type='notification', is_prompt=False, message='Test', **extra):
        item = {
            "id": "test1", "item_type": item_type, "message": message,
            "due_at": datetime(2024, 1, 15, 9, 0, tzinfo=timezone.utc),
            "topic": "general", "is_prompt": is_prompt,
        }
        item.update(extra)
        return item

    def test_notification_uses_output_service(self):
        with patch('services.output_service.OutputService') as mock_cls:
            mock_output = MagicMock()
            mock_cls.return_value = mock_output
            scheduler_svc._fire_item(self._make_item())
            assert mock_output.enqueue_text.called
            assert mock_output.enqueue_text.call_args[1]["mode"] == "NOTIFICATION"

    def test_prompt_spawns_daemon_thread(self):
        mock_t = MagicMock()
        with patch('threading.Thread', return_value=mock_t):
            scheduler_svc._fire_item(self._make_item('prompt', is_prompt=True, message='Do review'))
        mock_t.start.assert_called_once()

    def test_prompt_does_not_use_prompt_queue(self):
        """Regression: prompt items must NOT touch PromptQueue."""
        mock_t = MagicMock()
        with patch('threading.Thread', return_value=mock_t):
            scheduler_svc._fire_item(self._make_item('prompt', is_prompt=True, message='Check'))
        mock_t.start.assert_called_once()

    def test_empty_prompt_skipped(self):
        for msg in ['', '   ', '\n\t']:
            with patch('threading.Thread') as mock_t:
                scheduler_svc._fire_item(self._make_item('prompt', is_prompt=True, message=msg))
            mock_t.assert_not_called()

    def test_system_item_dispatches_handler(self):
        handler = MagicMock()
        scheduler_svc.register_system_handler('test-src', handler)
        scheduler_svc._fire_item(self._make_item('system', channel='test-src'))
        handler.assert_called_once()

    def test_system_item_no_handler_does_not_raise(self):
        scheduler_svc._fire_item(self._make_item('system', channel='nonexistent-xyz'))

    def test_error_message_is_sanitized(self):
        """On failure, user sees generic message — not raw exception.

        The scheduler constructs ScheduledMessageProcessor(raw_input=..., metadata=...)
        and calls .send() (v2 API, returns str). When .send() raises, the error
        message must be replaced with a safe generic string before reaching the user.
        """
        item = self._make_item('prompt', is_prompt=True, message='Fail task')
        captured_target = {}

        def _capture(*, target, **kw):
            captured_target['fn'] = target
            return MagicMock()

        with patch('threading.Thread', side_effect=_capture):
            scheduler_svc._fire_item(item)

        with patch('services.scheduled_message_processor.ScheduledMessageProcessor') as mock_proc, \
             patch('services.output_service.OutputService') as mock_out:
            # v2 API: .send() raises, not .process()
            mock_proc.return_value.send.side_effect = RuntimeError("secret error")
            mock_output = MagicMock()
            mock_out.return_value = mock_output
            captured_target['fn']()

        call_kwargs = mock_output.enqueue_proactive.call_args[1]
        assert 'secret error' not in call_kwargs['response']
        assert 'could not be completed' in call_kwargs['response']


@pytest.mark.unit
class TestPromptSemaphore:

    def test_semaphore_exists(self):
        import threading
        assert isinstance(scheduler_svc._PROMPT_SEMAPHORE, threading.Semaphore)
