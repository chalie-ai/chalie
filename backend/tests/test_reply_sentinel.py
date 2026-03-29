"""Tests for reply sentinel (capabilities.reply_sentinel)."""

import sqlite3
from contextlib import contextmanager
from datetime import timedelta
from unittest.mock import MagicMock, patch

import pytest

from services.time_utils import utc_now


def _make_db():
    """Create an in-memory SQLite connection with the hook_dedup table."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS hook_dedup ("
        "  key TEXT PRIMARY KEY,"
        "  expires_at TEXT NOT NULL"
        ")"
    )
    return conn


@contextmanager
def _fake_connection(conn):
    yield conn


def _mock_db(conn):
    mock = MagicMock()
    mock.connection.side_effect = lambda: _fake_connection(conn)
    return mock


@pytest.fixture(autouse=True)
def _reset_ensured():
    import capabilities.hook_dedup as mod
    mod._ENSURED = True
    yield
    mod._ENSURED = False


def _patches(conn):
    """Return combined patches for DB + MemoryStore."""
    mock_store = MagicMock()
    mock_store.get.return_value = None
    mock_mcs = MagicMock()
    mock_mcs.create_connection.return_value = mock_store
    return (
        patch("services.database_service.get_shared_db_service",
              return_value=_mock_db(conn)),
        patch("services.memory_client.MemoryClientService", mock_mcs),
    )


@pytest.mark.unit
class TestReplySentinel:

    def test_mark_and_check(self):
        """mark_replied records a Message-ID; is_replied finds it."""
        conn = _make_db()
        p1, p2 = _patches(conn)
        with p1, p2:
            from capabilities.reply_sentinel import is_replied, mark_replied
            msg_id = "<abc123@example.com>"
            assert not is_replied(msg_id)
            mark_replied(msg_id)
            assert is_replied(msg_id)
        conn.close()

    def test_filter_replied_removes_matched(self):
        """filter_replied excludes emails whose message_id was marked."""
        conn = _make_db()
        p1, p2 = _patches(conn)
        with p1, p2:
            from capabilities.reply_sentinel import filter_replied, mark_replied
            mark_replied("<replied@example.com>")
            emails = [
                {"uid": 1, "message_id": "<replied@example.com>",
                 "subject": "Old"},
                {"uid": 2, "message_id": "<fresh@example.com>",
                 "subject": "New"},
                {"uid": 3, "subject": "No MsgID"},
            ]
            result = filter_replied(emails)
            assert len(result) == 2
            assert result[0]["uid"] == 2
            assert result[1]["uid"] == 3
        conn.close()

    def test_filter_replied_passthrough(self):
        """filter_replied passes through empty lists and unmatched emails."""
        conn = _make_db()
        p1, p2 = _patches(conn)
        with p1, p2:
            from capabilities.reply_sentinel import filter_replied
            assert filter_replied([]) == []
            emails = [
                {"uid": 1, "message_id": "<a@b.com>", "subject": "A"},
                {"uid": 2, "message_id": "<c@d.com>", "subject": "B"},
            ]
            assert len(filter_replied(emails)) == 2
        conn.close()

    def test_expired_sentinel_not_matched(self):
        """An expired sentinel entry should not filter emails."""
        conn = _make_db()
        expired = (utc_now() - timedelta(hours=1)).isoformat()
        conn.execute(
            "INSERT INTO hook_dedup (key, expires_at) VALUES (?, ?)",
            ("replied_to:<old@example.com>", expired),
        )
        p1, p2 = _patches(conn)
        with p1, p2:
            from capabilities.reply_sentinel import is_replied
            assert not is_replied("<old@example.com>")
        conn.close()
