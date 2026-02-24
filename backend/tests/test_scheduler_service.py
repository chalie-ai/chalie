"""
Tests for backend/services/scheduler_service.py

Covers pure-computation functions (_calculate_next_due, _build_recurrence, _fire_item).
Poll/DB tests require PostgreSQL and are covered by integration tests.
"""

import pytest
from datetime import datetime, timedelta, timezone
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

    def test_prompt_uses_llm_pipeline(self):
        """Prompt items should be routed through the LLM pipeline via PromptQueue."""
        item = {
            "id": "prompt1",
            "item_type": "prompt",
            "message": "How did I do this week?",
            "due_at": datetime(2024, 1, 15, 9, 0, tzinfo=timezone.utc),
            "topic": "reflection",
            "is_prompt": True,
        }
        with patch('services.prompt_queue.PromptQueue') as mock_queue_cls, \
             patch('services.client_context_service.ClientContextService') as mock_ctx, \
             patch('workers.digest_worker.digest_worker'):
            mock_queue = MagicMock()
            mock_queue_cls.return_value = mock_queue
            mock_ctx.return_value.format_for_prompt.return_value = ""
            scheduler_svc._fire_item(item)
            assert mock_queue.enqueue.called
            enqueue_args = mock_queue.enqueue.call_args[0]
            assert enqueue_args[0] == "How did I do this week?"
