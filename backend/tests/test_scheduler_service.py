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
        assert next_due.weekday() < 5  # not Saturday or Sunday
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
        assert result["item_type"] == "notification"
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
        assert result is not None
        assert result["item_type"] == "prompt"
        assert result["is_prompt"] is True


@pytest.mark.unit
class TestFireItem:
    """Test _fire_item delivery routing."""

    def test_notification_uses_output_service(self):
        """Notification items should bypass LLM and go directly to OutputService."""
        item = {
            "id": "notif1",
            "item_type": "notification",
            "message": "Take your medicine",
            "due_at": datetime(2024, 1, 15, 9, 0, tzinfo=timezone.utc),
            "topic": "health",
            "is_prompt": False,
        }
        with patch('services.output_service.OutputService') as mock_output_cls:
            mock_output = MagicMock()
            mock_output_cls.return_value = mock_output
            scheduler_svc._fire_item(item)
            assert mock_output.enqueue_text.called
            call_kwargs = mock_output.enqueue_text.call_args[1]
            assert call_kwargs["mode"] == "NOTIFICATION"
            assert call_kwargs["response"] == "Take your medicine"

    def test_prompt_spawns_daemon_thread(self):
        """Prompt items must dispatch via a daemon thread — not PromptQueue."""
        import threading as _threading

        item = {
            "id": "prompt1",
            "item_type": "prompt",
            "message": "How did I do this week?",
            "due_at": datetime(2024, 1, 15, 9, 0, tzinfo=timezone.utc),
            "topic": "reflection",
            "is_prompt": True,
        }

        threads_started = []
        original_thread = _threading.Thread

        def _capture_thread(*args, **kwargs):
            t = original_thread(*args, **kwargs)
            threads_started.append(t)
            return t

        # threading is imported locally inside _fire_item(); patch it at the
        # threading module level so the inline import picks up the mock.
        with patch('threading.Thread', side_effect=_capture_thread):
            scheduler_svc._fire_item(item)

        assert len(threads_started) == 1, "Exactly one daemon thread must be started"
        t = threads_started[0]
        assert t.daemon is True, "Thread must be a daemon so it does not block shutdown"

    def test_prompt_thread_name_contains_item_id(self):
        """The spawned thread name must include the item id for traceability."""
        import threading as _threading

        item = {
            "id": "myitem99",
            "item_type": "prompt",
            "message": "Run the weekly review",
            "due_at": datetime(2024, 1, 15, 9, 0, tzinfo=timezone.utc),
            "topic": "reviews",
            "is_prompt": True,
        }

        threads_started = []
        original_thread = _threading.Thread

        def _capture_thread(*args, **kwargs):
            t = original_thread(*args, **kwargs)
            threads_started.append(t)
            return t

        with patch('threading.Thread', side_effect=_capture_thread):
            scheduler_svc._fire_item(item)

        assert threads_started, "A thread must have been created"
        assert 'myitem99' in threads_started[0].name

    def test_prompt_does_not_use_prompt_queue(self):
        """PromptQueue must NOT be imported or called for prompt items.

        The implementation spawns a daemon thread; we block the thread from
        actually running and assert that start() was called exactly once —
        proving the prompt path goes through the thread, not PromptQueue.
        """
        item = {
            "id": "prompt2",
            "item_type": "prompt",
            "message": "Daily check-in",
            "due_at": datetime(2024, 1, 15, 9, 0, tzinfo=timezone.utc),
            "topic": "general",
            "is_prompt": True,
        }

        mock_t = MagicMock()
        mock_t.daemon = False  # will be set by the implementation

        with patch('threading.Thread', return_value=mock_t):
            scheduler_svc._fire_item(item)

        mock_t.start.assert_called_once()

    def test_non_prompt_item_uses_output_service_directly(self):
        """Non-prompt items (e.g. reminder) still call OutputService.enqueue_text — no thread."""
        item = {
            "id": "reminder1",
            "item_type": "reminder",
            "message": "Stand up meeting",
            "due_at": datetime(2024, 1, 15, 9, 0, tzinfo=timezone.utc),
            "topic": "work",
            "is_prompt": False,
        }

        with patch('services.output_service.OutputService') as mock_output_cls, \
             patch('threading.Thread') as mock_thread_cls:
            mock_output = MagicMock()
            mock_output_cls.return_value = mock_output
            scheduler_svc._fire_item(item)

        # Direct delivery — no thread spawned
        mock_thread_cls.assert_not_called()
        assert mock_output.enqueue_text.called
        call_kwargs = mock_output.enqueue_text.call_args[1]
        assert call_kwargs["response"] == "Stand up meeting"

    def test_system_item_dispatches_to_registered_handler(self):
        """System items must call the registered handler — not OutputService, not a thread."""
        handler = MagicMock()
        scheduler_svc.register_system_handler('test-source', handler)

        item = {
            "id": "sys1",
            "item_type": "system",
            "message": "",
            "due_at": datetime(2024, 1, 15, 9, 0, tzinfo=timezone.utc),
            "channel": "test-source",
            "is_prompt": False,
        }

        with patch('services.output_service.OutputService') as mock_output_cls, \
             patch('threading.Thread') as mock_thread_cls:
            scheduler_svc._fire_item(item)

        handler.assert_called_once()
        mock_output_cls.assert_not_called()
        mock_thread_cls.assert_not_called()

    def test_system_item_with_no_handler_does_not_raise(self):
        """A system item with an unregistered channel must log a warning, not raise."""
        item = {
            "id": "sys2",
            "item_type": "system",
            "message": "",
            "due_at": datetime(2024, 1, 15, 9, 0, tzinfo=timezone.utc),
            "channel": "nonexistent-handler-xyz",
            "is_prompt": False,
        }

        # Must complete without raising
        scheduler_svc._fire_item(item)

    def test_empty_prompt_message_skipped(self):
        """Prompt items with empty/whitespace message must be skipped — no thread spawned."""
        for empty_msg in ['', '   ', '\n\t']:
            item = {
                "id": "empty1",
                "item_type": "prompt",
                "message": empty_msg,
                "due_at": datetime(2024, 1, 15, 9, 0, tzinfo=timezone.utc),
                "topic": "general",
                "is_prompt": True,
            }

            with patch('threading.Thread') as mock_thread_cls:
                scheduler_svc._fire_item(item)

            mock_thread_cls.assert_not_called()

    def test_prompt_error_message_is_sanitized(self):
        """When the daemon thread fails, the user must see a generic error — not raw exception."""
        item = {
            "id": "fail1",
            "item_type": "prompt",
            "message": "Do something that fails",
            "due_at": datetime(2024, 1, 15, 9, 0, tzinfo=timezone.utc),
            "topic": "general",
            "is_prompt": True,
        }

        # Capture the _run function passed to Thread, then execute it directly
        captured_target = {}

        def _capture_thread(*args, **kwargs):
            captured_target['fn'] = kwargs.get('target')
            t = MagicMock()
            return t

        with patch('threading.Thread', side_effect=_capture_thread):
            scheduler_svc._fire_item(item)

        assert 'fn' in captured_target

        # Execute _run — the processor will raise
        with patch('services.scheduled_message_processor.ScheduledMessageProcessor') as mock_proc_cls, \
             patch('services.output_service.OutputService') as mock_output_cls:
            mock_proc_cls.return_value.process.side_effect = RuntimeError("secret internal error")
            mock_output = MagicMock()
            mock_output_cls.return_value = mock_output

            captured_target['fn']()

        # Error message must be generic — no raw exception text
        call_kwargs = mock_output.enqueue_proactive.call_args[1]
        assert 'secret internal error' not in call_kwargs['response']
        assert 'could not be completed' in call_kwargs['response']


@pytest.mark.unit
class TestPromptSemaphore:
    """Test the concurrent thread cap via _PROMPT_SEMAPHORE."""

    def test_semaphore_exists_and_is_threading_semaphore(self):
        """Module-level _PROMPT_SEMAPHORE must exist as a threading.Semaphore."""
        import threading as _threading
        assert hasattr(scheduler_svc, '_PROMPT_SEMAPHORE')
        sem = scheduler_svc._PROMPT_SEMAPHORE
        assert isinstance(sem, _threading.Semaphore)
