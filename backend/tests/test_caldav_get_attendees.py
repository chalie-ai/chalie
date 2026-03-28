"""Unit tests for the ``caldav_get_attendees`` tool handler.

Covers:
    1. Event found with attendees — resolved via ContactResolver.
    2. Event found with no attendees — returns empty list.
    3. Event not found — returns error.
    4. Unresolved email — still returned with empty name.
    5. Not connected — returns error.

All tests are marked ``@pytest.mark.unit``.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


def _make_capability():
    from capabilities.caldav_capability.capability import CaldavCapability
    return CaldavCapability()


def _find_attendees_handler(cap):
    """Extract the caldav_get_attendees handler from get_tools()."""
    tools = cap.get_tools()
    for t in tools:
        if t["name"] == "caldav_get_attendees":
            return t["handler"]
    raise AssertionError("caldav_get_attendees tool not found in get_tools()")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _insert_event(conn, uid, title, attendees):
    """Insert a fake scheduled_items row for testing."""
    meta = json.dumps({"attendees": attendees, "calendar_name": "Work"})
    conn.execute(
        "INSERT INTO scheduled_items "
        "(external_uid, message, due_at, metadata, source, item_type, status) "
        "VALUES (?, ?, '2026-04-01T10:00:00+00:00', ?, 'caldav', 'event', 'pending')",
        (f"caldav:{uid}", title, meta),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCaldavGetAttendees:

    def test_resolved_attendees(self):
        """Attendees with people-index matches get resolved names."""
        cap = _make_capability()
        cap._connected = True
        handler = _find_attendees_handler(cap)

        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = lambda s: s
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_db.connection.return_value = mock_conn

        meta = json.dumps({"attendees": ["alice@corp.com", "bob@corp.com"]})
        mock_cursor.fetchone.return_value = ("Team Standup", meta)

        resolve_results = {
            "alice@corp.com": [{"email": "alice@corp.com", "name": "Alice Chen"}],
            "bob@corp.com": [{"email": "bob@corp.com", "name": "Bob Smith"}],
        }

        with patch(
            "services.database_service.get_shared_db_service",
            return_value=mock_db,
        ), patch(
            "capabilities.contact_resolver.resolve",
            side_effect=lambda email, limit=5: resolve_results.get(email, []),
        ):
            result = handler("topic1", {"uid": "evt-001"})

        assert result["event_title"] == "Team Standup"
        assert result["count"] == 2
        assert result["attendees"][0]["name"] == "Alice Chen"
        assert result["attendees"][1]["name"] == "Bob Smith"

    def test_no_attendees(self):
        """Event with empty attendee list returns empty result."""
        cap = _make_capability()
        cap._connected = True
        handler = _find_attendees_handler(cap)

        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = lambda s: s
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_db.connection.return_value = mock_conn

        meta = json.dumps({"attendees": []})
        mock_cursor.fetchone.return_value = ("Solo Focus", meta)

        with patch(
            "services.database_service.get_shared_db_service",
            return_value=mock_db,
        ):
            result = handler("topic1", {"uid": "evt-002"})

        assert result["count"] == 0
        assert result["attendees"] == []

    def test_event_not_found(self):
        """Missing event UID returns an error."""
        cap = _make_capability()
        cap._connected = True
        handler = _find_attendees_handler(cap)

        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = lambda s: s
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_db.connection.return_value = mock_conn
        mock_cursor.fetchone.return_value = None

        with patch(
            "services.database_service.get_shared_db_service",
            return_value=mock_db,
        ):
            result = handler("topic1", {"uid": "nonexistent"})

        assert "error" in result
        assert "not found" in result["error"].lower()

    def test_unresolved_email(self):
        """Attendee email with no people-index match gets empty name."""
        cap = _make_capability()
        cap._connected = True
        handler = _find_attendees_handler(cap)

        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = lambda s: s
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_db.connection.return_value = mock_conn

        meta = json.dumps({"attendees": ["unknown@external.com"]})
        mock_cursor.fetchone.return_value = ("Client Call", meta)

        with patch(
            "services.database_service.get_shared_db_service",
            return_value=mock_db,
        ), patch(
            "capabilities.contact_resolver.resolve",
            return_value=[],
        ):
            result = handler("topic1", {"uid": "evt-003"})

        assert result["count"] == 1
        assert result["attendees"][0]["email"] == "unknown@external.com"
        assert result["attendees"][0]["name"] == ""

    def test_not_connected_returns_error(self):
        """Handler returns error when CalDAV is not connected."""
        cap = _make_capability()
        cap._connected = False
        handler = _find_attendees_handler(cap)

        result = handler("topic1", {"uid": "evt-001"})
        assert "error" in result

    def test_missing_uid_returns_error(self):
        """Handler returns error when uid param is missing."""
        cap = _make_capability()
        cap._connected = True
        handler = _find_attendees_handler(cap)

        result = handler("topic1", {})
        assert "error" in result
        assert "uid" in result["error"].lower()
