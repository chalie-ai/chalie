"""
Unit tests for Google Account tool handler — tests action routing and handler logic.

Uses mocked HTTP responses (no real Google API calls).
"""
import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add tools directory to path for imports
TOOL_DIR = Path(__file__).parent.parent / "tools" / "google_account"
sys.path.insert(0, str(TOOL_DIR))


@pytest.mark.unit
class TestHandlerRouting:
    """Test that the handler routes actions correctly."""

    def test_missing_action(self):
        from handler import execute
        result = execute(params={}, settings={"_oauth_access_token": "tok"}, telemetry={})
        assert "error" in result
        assert "action" in result["error"].lower()

    def test_unknown_action(self):
        from handler import execute
        result = execute(
            params={"action": "unknown_action"},
            settings={"_oauth_access_token": "tok"},
            telemetry={},
        )
        assert "error" in result
        assert "unknown_action" in result["error"].lower()

    def test_missing_oauth_token(self):
        from handler import execute
        result = execute(
            params={"action": "gmail_unread"},
            settings={},
            telemetry={},
        )
        assert "error" in result
        assert "not connected" in result["error"].lower()

    def test_all_actions_recognized(self):
        from handler import ALL_ACTIONS
        expected = {
            "gmail_unread", "gmail_search", "gmail_read", "gmail_send", "gmail_reply",
            "calendar_today", "calendar_tomorrow", "calendar_week",
            "calendar_search", "calendar_create", "calendar_update", "calendar_delete",
            "tasks_list", "tasks_get", "tasks_create", "tasks_complete",
        }
        assert ALL_ACTIONS == expected


@pytest.mark.unit
class TestGmailHandler:
    """Test Gmail handler methods."""

    def _mock_response(self, json_data, status=200):
        resp = MagicMock()
        resp.json.return_value = json_data
        resp.status_code = status
        resp.raise_for_status = MagicMock()
        return resp

    def test_get_unread(self):
        from gmail_handler import GmailHandler

        msg_list = {"messages": [{"id": "msg1"}], "resultSizeEstimate": 1}
        msg_detail = {
            "id": "msg1",
            "snippet": "Hello world",
            "payload": {
                "headers": [
                    {"name": "From", "value": "alice@example.com"},
                    {"name": "Subject", "value": "Test email"},
                    {"name": "Date", "value": "Mon, 1 Jan 2024 12:00:00"},
                ]
            }
        }

        with patch('requests.get') as mock_get:
            mock_get.side_effect = [
                self._mock_response(msg_list),
                self._mock_response(msg_detail),
            ]
            handler = GmailHandler("test-token")
            result = handler.get_unread(5)

        assert result["type"] == "email_list"
        assert result["unread_count"] == 1
        assert len(result["emails"]) == 1
        assert result["emails"][0]["from"] == "alice@example.com"
        assert "text" in result

    def test_search(self):
        from gmail_handler import GmailHandler

        with patch('requests.get') as mock_get:
            mock_get.return_value = self._mock_response({
                "messages": [], "resultSizeEstimate": 0
            })
            handler = GmailHandler("test-token")
            result = handler.search("from:bob", 10)

        assert result["type"] == "email_list"
        assert result["query"] == "from:bob"
        assert result["result_count"] == 0

    def test_search_missing_query(self):
        from gmail_handler import GmailHandler
        handler = GmailHandler("test-token")
        result = handler.search("", 10)
        assert "error" in result

    def test_send(self):
        from gmail_handler import GmailHandler

        with patch('requests.post') as mock_post:
            mock_post.return_value = self._mock_response({"id": "sent1"})
            handler = GmailHandler("test-token")
            result = handler.send(to="bob@example.com", subject="Hi", body="Hello")

        assert result["type"] == "email_sent"
        assert result["to"] == "bob@example.com"

    def test_send_missing_to(self):
        from gmail_handler import GmailHandler
        handler = GmailHandler("test-token")
        result = handler.send(to="", subject="Hi", body="Hello")
        assert "error" in result

    def test_read_message(self):
        from gmail_handler import GmailHandler
        import base64

        body_data = base64.urlsafe_b64encode(b"Email body text").decode()
        msg = {
            "id": "msg1",
            "payload": {
                "mimeType": "text/plain",
                "headers": [
                    {"name": "From", "value": "alice@example.com"},
                    {"name": "To", "value": "bob@example.com"},
                    {"name": "Subject", "value": "Test"},
                    {"name": "Date", "value": "Mon, 1 Jan 2024"},
                ],
                "body": {"data": body_data},
            }
        }

        with patch('requests.get') as mock_get:
            mock_get.return_value = self._mock_response(msg)
            handler = GmailHandler("test-token")
            result = handler.read_message("msg1")

        assert result["type"] == "email_detail"
        assert "Email body text" in result["body"]


@pytest.mark.unit
class TestCalendarHandler:
    """Test Calendar handler methods."""

    def _mock_response(self, json_data):
        resp = MagicMock()
        resp.json.return_value = json_data
        resp.raise_for_status = MagicMock()
        return resp

    def test_get_today(self):
        from calendar_handler import CalendarHandler

        events_data = {
            "items": [
                {
                    "id": "evt1",
                    "summary": "Team standup",
                    "start": {"dateTime": "2024-01-01T09:00:00+01:00"},
                    "end": {"dateTime": "2024-01-01T09:30:00+01:00"},
                    "status": "confirmed",
                }
            ]
        }

        with patch('requests.get') as mock_get:
            mock_get.return_value = self._mock_response(events_data)
            handler = CalendarHandler("test-token")
            result = handler.get_today(10)

        assert result["type"] == "calendar_list"
        assert result["event_count"] == 1
        assert result["events"][0]["title"] == "Team standup"

    def test_create_event(self):
        from calendar_handler import CalendarHandler

        created = {
            "id": "new-evt",
            "summary": "Lunch",
            "start": {"dateTime": "2024-01-01T12:00:00Z"},
            "end": {"dateTime": "2024-01-01T13:00:00Z"},
            "status": "confirmed",
        }

        with patch('requests.post') as mock_post:
            mock_post.return_value = self._mock_response(created)
            handler = CalendarHandler("test-token")
            result = handler.create_event(
                title="Lunch",
                start="2024-01-01T12:00:00Z",
                end="2024-01-01T13:00:00Z",
            )

        assert result["type"] == "calendar_created"
        assert result["event"]["title"] == "Lunch"

    def test_create_event_missing_title(self):
        from calendar_handler import CalendarHandler
        handler = CalendarHandler("test-token")
        result = handler.create_event(title="", start="2024-01-01T12:00:00Z", end="2024-01-01T13:00:00Z")
        assert "error" in result

    def test_delete_event(self):
        from calendar_handler import CalendarHandler

        with patch('requests.get') as mock_get, \
             patch('requests.delete') as mock_delete:
            mock_get.return_value = self._mock_response({"summary": "Old meeting"})
            mock_delete.return_value = MagicMock(raise_for_status=MagicMock())
            handler = CalendarHandler("test-token")
            result = handler.delete_event("evt-123")

        assert result["type"] == "calendar_deleted"
        assert result["title"] == "Old meeting"

    def test_all_day_event_parsing(self):
        from calendar_handler import CalendarHandler

        events_data = {
            "items": [
                {
                    "id": "ad1",
                    "summary": "Holiday",
                    "start": {"date": "2024-01-01"},
                    "end": {"date": "2024-01-02"},
                }
            ]
        }

        with patch('requests.get') as mock_get:
            mock_get.return_value = self._mock_response(events_data)
            handler = CalendarHandler("test-token")
            result = handler.get_today(10)

        assert result["events"][0]["is_all_day"] is True
        assert result["events"][0]["start_display"] == "All day"


@pytest.mark.unit
class TestTasksHandler:
    """Test Tasks handler methods."""

    def _mock_response(self, json_data):
        resp = MagicMock()
        resp.json.return_value = json_data
        resp.raise_for_status = MagicMock()
        return resp

    def test_list_task_lists(self):
        from tasks_handler import TasksHandler

        with patch('requests.get') as mock_get:
            mock_get.return_value = self._mock_response({
                "items": [
                    {"id": "list1", "title": "My Tasks", "updated": "2024-01-01T00:00:00Z"},
                ]
            })
            handler = TasksHandler("test-token")
            result = handler.list_task_lists()

        assert result["type"] == "task_lists"
        assert len(result["lists"]) == 1
        assert result["lists"][0]["title"] == "My Tasks"

    def test_get_tasks(self):
        from tasks_handler import TasksHandler

        with patch('requests.get') as mock_get:
            mock_get.return_value = self._mock_response({
                "items": [
                    {"id": "t1", "title": "Buy groceries", "status": "needsAction"},
                    {"id": "t2", "title": "Call dentist", "status": "needsAction", "due": "2024-01-15T00:00:00Z"},
                ]
            })
            handler = TasksHandler("test-token")
            result = handler.get_tasks("", 20)

        assert result["type"] == "task_list"
        assert result["task_count"] == 2
        assert result["tasks"][0]["title"] == "Buy groceries"
        assert result["tasks"][1]["due"] == "2024-01-15T00:00:00Z"

    def test_create_task(self):
        from tasks_handler import TasksHandler

        with patch('requests.post') as mock_post:
            mock_post.return_value = self._mock_response({
                "id": "new-task",
                "title": "Do laundry",
                "status": "needsAction",
            })
            handler = TasksHandler("test-token")
            result = handler.create_task(title="Do laundry")

        assert result["type"] == "task_created"
        assert result["task"]["title"] == "Do laundry"

    def test_create_task_missing_title(self):
        from tasks_handler import TasksHandler
        handler = TasksHandler("test-token")
        result = handler.create_task(title="")
        assert "error" in result

    def test_complete_task(self):
        from tasks_handler import TasksHandler

        with patch('requests.get') as mock_get, \
             patch('requests.patch') as mock_patch:
            mock_get.return_value = self._mock_response({"title": "Done task"})
            mock_patch.return_value = self._mock_response({"id": "t1", "status": "completed"})
            handler = TasksHandler("test-token")
            result = handler.complete_task("t1")

        assert result["type"] == "task_completed"
        assert result["title"] == "Done task"


@pytest.mark.unit
class TestCardTemplates:
    """Test that card rendering produces non-empty HTML."""

    def test_email_list_card(self):
        from card_templates import render_card
        data = {
            "type": "email_list",
            "unread_count": 2,
            "emails": [
                {"id": "1", "from": "alice@ex.com", "subject": "Hi", "date": "Mon", "snippet": "Hello"},
            ],
        }
        html = render_card("gmail_unread", data)
        assert len(html) > 50
        assert "Gmail" in html
        assert "alice" in html

    def test_calendar_card(self):
        from card_templates import render_card
        data = {
            "type": "calendar_list",
            "label": "today",
            "event_count": 1,
            "events": [
                {"id": "e1", "title": "Meeting", "start_display": "09:00", "end_display": "10:00",
                 "is_all_day": False, "location": "Room A", "description": "", "attendees": [], "status": "confirmed"},
            ],
        }
        html = render_card("calendar_today", data)
        assert len(html) > 50
        assert "Calendar" in html
        assert "Meeting" in html

    def test_tasks_card(self):
        from card_templates import render_card
        data = {
            "type": "task_list",
            "list_title": "My Tasks",
            "task_count": 1,
            "tasks": [
                {"id": "t1", "title": "Buy milk", "completed": False, "due": "", "notes": "", "status": "needsAction", "updated": ""},
            ],
        }
        html = render_card("tasks_get", data)
        assert len(html) > 50
        assert "Buy milk" in html

    def test_confirmation_card(self):
        from card_templates import render_card
        data = {"type": "email_sent", "text": "Email sent to bob@ex.com"}
        html = render_card("gmail_send", data)
        assert len(html) > 30
        assert "bob@ex.com" in html

    def test_error_card(self):
        from card_templates import render_card
        data = {"error": "Something went wrong"}
        html = render_card("gmail_unread", data)
        assert "Something went wrong" in html
