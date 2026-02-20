"""
Tests for backend/tools/scheduler/handler.py
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch
from tools.scheduler.handler import execute
import tools.scheduler.handler as scheduler_handler


@pytest.mark.unit
class TestSchedulerToolHandler:
    """Test scheduler tool handler."""

    @pytest.fixture(autouse=True)
    def setup_temp_db(self, tmp_path, monkeypatch):
        """Setup temporary database directory for each test."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        db_path = data_dir / "scheduler.db"
        monkeypatch.setattr(scheduler_handler, '_DATA_DIR', data_dir)
        monkeypatch.setattr(scheduler_handler, '_DB_PATH', db_path)

    def test_create_reminder(self):
        """Create a reminder should return status=created."""
        due_at = (datetime.now() + timedelta(hours=1)).isoformat()
        result = execute("test_topic", {
            "action": "create",
            "type": "reminder",
            "message": "Test reminder",
            "due_at": due_at
        })

        assert result['status'] == "created"
        assert 'id' in result
        assert 'due_at' in result

    def test_create_task(self):
        """Create a task should work."""
        due_at = (datetime.now() + timedelta(hours=2)).isoformat()
        result = execute("test_topic", {
            "action": "create",
            "type": "task",
            "message": "Test task",
            "due_at": due_at
        })

        assert result['status'] == "created"

    def test_create_missing_message(self):
        """Create without message should error."""
        due_at = (datetime.now() + timedelta(hours=1)).isoformat()
        result = execute("test_topic", {
            "action": "create",
            "type": "reminder",
            "due_at": due_at
        })

        assert "error" in result

    def test_create_missing_due_at(self):
        """Create without due_at should error."""
        result = execute("test_topic", {
            "action": "create",
            "type": "reminder",
            "message": "Test"
        })

        assert "error" in result

    def test_create_past_due_at(self):
        """Create with past due_at should error."""
        past_time = (datetime.now() - timedelta(hours=1)).isoformat()
        result = execute("test_topic", {
            "action": "create",
            "type": "reminder",
            "message": "Past reminder",
            "due_at": past_time
        })

        assert "error" in result
        assert "future" in result['error'].lower()

    def test_create_too_far_future(self):
        """Create with due_at > 365 days should error."""
        far_future = (datetime.now() + timedelta(days=400)).isoformat()
        result = execute("test_topic", {
            "action": "create",
            "type": "reminder",
            "message": "Too far",
            "due_at": far_future
        })

        assert "error" in result
        assert "365" in result['error']

    def test_create_invalid_recurrence(self):
        """Create with invalid recurrence should error."""
        due_at = (datetime.now() + timedelta(hours=1)).isoformat()
        result = execute("test_topic", {
            "action": "create",
            "type": "reminder",
            "message": "Bad recurrence",
            "due_at": due_at,
            "recurrence": "invalid"
        })

        assert "error" in result

    def test_create_valid_recurrences(self):
        """Valid recurrence values should be accepted."""
        due_at = (datetime.now() + timedelta(hours=1)).isoformat()
        for recurrence in ["daily", "weekly", "monthly", "weekdays"]:
            result = execute("test_topic", {
                "action": "create",
                "type": "reminder",
                "message": f"Recurrence: {recurrence}",
                "due_at": due_at,
                "recurrence": recurrence
            })

            assert result['status'] == "created"

    def test_list_pending_items(self):
        """List should return pending items."""
        due_at = (datetime.now() + timedelta(hours=1)).isoformat()
        execute("test_topic", {
            "action": "create",
            "type": "reminder",
            "message": "Test 1",
            "due_at": due_at
        })

        result = execute("test_topic", {"action": "list"})

        assert 'items' in result
        assert result['count'] >= 1
        assert any(item['message'] == 'Test 1' for item in result['items'])

    def test_list_empty(self):
        """List with no items should return empty."""
        result = execute("test_topic", {"action": "list"})

        assert result['items'] == []
        assert result['count'] == 0

    def test_cancel_existing_item(self):
        """Cancel should mark item as cancelled."""
        # Create item
        due_at = (datetime.now() + timedelta(hours=1)).isoformat()
        create_result = execute("test_topic", {
            "action": "create",
            "type": "reminder",
            "message": "To cancel",
            "due_at": due_at
        })

        item_id = create_result['id']

        # Cancel it
        cancel_result = execute("test_topic", {
            "action": "cancel",
            "item_id": item_id
        })

        assert cancel_result['status'] == "cancelled"

        # Verify it's gone from list
        list_result = execute("test_topic", {"action": "list"})
        assert not any(item['id'] == item_id for item in list_result['items'])

    def test_cancel_nonexistent_item(self):
        """Cancel nonexistent item should error."""
        result = execute("test_topic", {
            "action": "cancel",
            "item_id": "nonexistent"
        })

        assert "error" in result

    def test_cancel_missing_item_id(self):
        """Cancel without item_id should error."""
        result = execute("test_topic", {"action": "cancel"})

        assert "error" in result

    def test_unknown_action(self):
        """Unknown action should error."""
        result = execute("test_topic", {"action": "invalid"})

        assert "error" in result
        assert "Unknown action" in result['error']

    def test_invalid_type(self):
        """Invalid type should error."""
        due_at = (datetime.now() + timedelta(hours=1)).isoformat()
        result = execute("test_topic", {
            "action": "create",
            "type": "invalid",
            "message": "Test",
            "due_at": due_at
        })

        assert "error" in result
        assert "reminder" in result['error'] or "task" in result['error']
