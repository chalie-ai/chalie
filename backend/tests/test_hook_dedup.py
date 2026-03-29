"""Tests for persistent hook dedup (capabilities.hook_dedup)."""

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
    """Context manager that yields the raw connection."""
    yield conn


def _mock_db(conn):
    """Return a mock DB service whose .connection() yields *conn* each time."""
    mock = MagicMock()
    mock.connection.side_effect = lambda: _fake_connection(conn)
    return mock


@pytest.fixture(autouse=True)
def _reset_ensured():
    """Reset the module-level _ENSURED flag between tests."""
    import capabilities.hook_dedup as mod
    mod._ENSURED = True  # skip CREATE TABLE — we manage the table ourselves
    yield
    mod._ENSURED = False


@pytest.mark.unit
class TestHookDedup:

    def test_mark_and_check(self):
        """mark_fired persists to SQLite; is_fired reads it back."""
        conn = _make_db()
        mock_store = MagicMock()
        mock_store.get.return_value = None
        mock_mcs = MagicMock()
        mock_mcs.create_connection.return_value = mock_store

        with patch("services.database_service.get_shared_db_service",
                   return_value=_mock_db(conn)), \
             patch("services.memory_client.MemoryClientService", mock_mcs):
            from capabilities.hook_dedup import is_fired, mark_fired
            assert not is_fired("morning_brief:2026-03-29")
            mark_fired("morning_brief:2026-03-29", 86400)
            assert is_fired("morning_brief:2026-03-29")
        conn.close()

    def test_expired_flag_not_fired(self):
        """An expired flag should return False from is_fired."""
        conn = _make_db()
        expired = (utc_now() - timedelta(hours=1)).isoformat()
        conn.execute(
            "INSERT INTO hook_dedup (key, expires_at) VALUES (?, ?)",
            ("old_flag", expired),
        )
        mock_store = MagicMock()
        mock_store.get.return_value = None
        mock_mcs = MagicMock()
        mock_mcs.create_connection.return_value = mock_store

        with patch("services.database_service.get_shared_db_service",
                   return_value=_mock_db(conn)), \
             patch("services.memory_client.MemoryClientService", mock_mcs):
            from capabilities.hook_dedup import is_fired
            assert not is_fired("old_flag")
        conn.close()

    def test_memstore_cache_hit(self):
        """If MemoryStore has the key, skip SQLite entirely."""
        mock_store = MagicMock()
        mock_store.get.return_value = "1"
        mock_mcs = MagicMock()
        mock_mcs.create_connection.return_value = mock_store

        with patch("services.memory_client.MemoryClientService", mock_mcs):
            from capabilities.hook_dedup import is_fired
            assert is_fired("cached_key")

    def test_mark_cleans_expired(self):
        """mark_fired should delete expired rows."""
        conn = _make_db()
        expired = (utc_now() - timedelta(hours=1)).isoformat()
        conn.execute(
            "INSERT INTO hook_dedup (key, expires_at) VALUES (?, ?)",
            ("stale", expired),
        )
        mock_store = MagicMock()
        mock_store.get.return_value = None
        mock_mcs = MagicMock()
        mock_mcs.create_connection.return_value = mock_store

        with patch("services.database_service.get_shared_db_service",
                   return_value=_mock_db(conn)), \
             patch("services.memory_client.MemoryClientService", mock_mcs):
            from capabilities.hook_dedup import mark_fired
            mark_fired("new_key", 3600)

        row = conn.execute(
            "SELECT 1 FROM hook_dedup WHERE key = 'stale'"
        ).fetchone()
        assert row is None
        conn.close()

    def test_survives_memstore_failure(self):
        """is_fired falls through to SQLite when MemoryStore raises."""
        conn = _make_db()
        future = (utc_now() + timedelta(hours=1)).isoformat()
        conn.execute(
            "INSERT INTO hook_dedup (key, expires_at) VALUES (?, ?)",
            ("persisted", future),
        )
        broken_mcs = MagicMock()
        broken_mcs.create_connection.side_effect = RuntimeError("down")

        with patch("services.database_service.get_shared_db_service",
                   return_value=_mock_db(conn)), \
             patch("services.memory_client.MemoryClientService", broken_mcs):
            from capabilities.hook_dedup import is_fired
            assert is_fired("persisted")
        conn.close()
