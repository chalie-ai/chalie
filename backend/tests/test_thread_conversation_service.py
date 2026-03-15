"""Tests for ThreadConversationService — CRUD operations, TTL behavior, SQLite persistence."""

import json
import sqlite3
import pytest
from unittest.mock import patch, MagicMock

from services.thread_conversation_service import ThreadConversationService


@pytest.fixture
def conv_service(mock_store):
    """ThreadConversationService with fake MemoryStore."""
    with patch('services.thread_conversation_service.MemoryClientService.create_connection', return_value=mock_store):
        yield ThreadConversationService()


@pytest.fixture
def conv_service_with_db(mock_store):
    """ThreadConversationService with fake MemoryStore and real in-memory SQLite."""
    db = _create_in_memory_db()
    with patch('services.thread_conversation_service.MemoryClientService.create_connection', return_value=mock_store):
        svc = ThreadConversationService()
        svc._db_service = db
        yield svc, db, mock_store


def _create_in_memory_db():
    """Create an in-memory DatabaseService with the thread_exchanges table."""
    from services.database_service import DatabaseService
    db = DatabaseService.__new__(DatabaseService)
    db.db_path = ":memory:"

    # Force a shared connection for the in-memory DB (thread-local won't work
    # because :memory: creates a new DB per connection).
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE thread_exchanges (
            id TEXT PRIMARY KEY,
            thread_id TEXT NOT NULL,
            topic TEXT NOT NULL DEFAULT '',
            prompt_message TEXT NOT NULL DEFAULT '',
            prompt_time TEXT NOT NULL,
            response_message TEXT,
            response_time TEXT,
            response_error TEXT,
            generation_time_ms REAL,
            steps TEXT DEFAULT '[]',
            memory_chunk TEXT DEFAULT '{}',
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()

    # Monkey-patch the DB service to always return this connection
    import contextlib

    @contextlib.contextmanager
    def fake_connection():
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    db.connection = fake_connection

    from services.database_service import DictCursor

    def fake_execute(sql, params=None):
        with fake_connection() as c:
            cur = c.cursor()
            try:
                if params is None:
                    cur.execute(sql)
                else:
                    cur.execute(sql, params)
            finally:
                cur.close()

    def fake_fetch_all(sql, params=None):
        with fake_connection() as c:
            cur = DictCursor(c.cursor())
            try:
                cur.execute(sql, params)
                return cur.fetchall()
            finally:
                cur.close()

    db.execute = fake_execute
    db.fetch_all = fake_fetch_all
    return db


THREAD_ID = "telegram:user1:chan1:1"


class TestAddExchange:
    def test_adds_exchange_with_prompt(self, conv_service, mock_store):
        eid = conv_service.add_exchange(THREAD_ID, "test-topic", {
            "message": "Hello there",
            "classification_time": 0.05,
        })

        assert eid  # non-empty UUID
        history = conv_service.get_conversation_history(THREAD_ID)
        assert len(history) == 1
        assert history[0]["prompt"]["message"] == "Hello there"
        assert history[0]["topic"] == "test-topic"

    def test_exchange_has_no_response_initially(self, conv_service):
        conv_service.add_exchange(THREAD_ID, "topic", {"message": "Hi"})
        history = conv_service.get_conversation_history(THREAD_ID)
        assert history[0]["response"] is None

    def test_exchange_count_increments(self, conv_service):
        conv_service.add_exchange(THREAD_ID, "topic", {"message": "1"})
        conv_service.add_exchange(THREAD_ID, "topic", {"message": "2"})
        assert conv_service.get_exchange_count(THREAD_ID) == 2


class TestAddResponse:
    def test_adds_response_to_latest_exchange(self, conv_service):
        conv_service.add_exchange(THREAD_ID, "topic", {"message": "Hi"})
        conv_service.add_response(THREAD_ID, "Hello back!", 1.5)

        history = conv_service.get_conversation_history(THREAD_ID)
        assert history[0]["response"]["message"] == "Hello back!"
        assert history[0]["response"]["generation_time"] == 1.5

    def test_response_error(self, conv_service):
        conv_service.add_exchange(THREAD_ID, "topic", {"message": "Hi"})
        conv_service.add_response_error(THREAD_ID, "LLM timeout")

        history = conv_service.get_conversation_history(THREAD_ID)
        assert "error" in history[0]["response"]
        assert history[0]["response"]["error"] == "LLM timeout"


class TestAddSteps:
    def test_adds_steps_to_latest_exchange(self, conv_service):
        conv_service.add_exchange(THREAD_ID, "topic", {"message": "Check weather"})
        conv_service.add_steps_to_exchange(THREAD_ID, [
            {"type": "recall", "description": "Look up weather data"},
        ])

        history = conv_service.get_conversation_history(THREAD_ID)
        assert len(history[0]["steps"]) == 1
        assert history[0]["steps"][0]["status"] == "pending"


class TestAddMemoryChunk:
    def test_adds_memory_chunk_by_exchange_id(self, conv_service):
        eid = conv_service.add_exchange(THREAD_ID, "topic", {"message": "Hi"})
        conv_service.add_memory_chunk(THREAD_ID, eid, {"gists": [{"content": "Greeting"}]})

        history = conv_service.get_conversation_history(THREAD_ID)
        assert history[0]["memory_chunk"]["gists"][0]["content"] == "Greeting"


class TestGetActiveSteps:
    def test_returns_only_active_steps(self, conv_service):
        conv_service.add_exchange(THREAD_ID, "topic", {"message": "Do stuff"})
        conv_service.add_steps_to_exchange(THREAD_ID, [
            {"type": "task", "description": "Step 1"},
            {"type": "task", "description": "Step 2"},
        ])

        # Mark step 1 as completed by updating the exchange directly
        history = conv_service.get_conversation_history(THREAD_ID)
        exchange = history[0]
        exchange["steps"][0]["status"] = "completed"
        conv_service.store.lset(conv_service._conv_key(THREAD_ID), 0, json.dumps(exchange))

        active = conv_service.get_active_steps(THREAD_ID)
        assert len(active) == 1
        assert active[0]["description"] == "Step 2"


class TestGetLatestExchangeId:
    def test_returns_latest_id(self, conv_service):
        eid1 = conv_service.add_exchange(THREAD_ID, "topic", {"message": "First"})
        eid2 = conv_service.add_exchange(THREAD_ID, "topic", {"message": "Second"})

        assert conv_service.get_latest_exchange_id(THREAD_ID) == eid2

    def test_returns_unknown_for_empty(self, conv_service):
        assert conv_service.get_latest_exchange_id(THREAD_ID) == "unknown"


class TestRemoveExchanges:
    def test_removes_specific_exchanges(self, conv_service):
        eid1 = conv_service.add_exchange(THREAD_ID, "topic", {"message": "Keep"})
        eid2 = conv_service.add_exchange(THREAD_ID, "topic", {"message": "Remove"})

        conv_service.remove_exchanges(THREAD_ID, [eid2])

        history = conv_service.get_conversation_history(THREAD_ID)
        assert len(history) == 1
        assert history[0]["prompt"]["message"] == "Keep"


class TestSQLitePersistence:
    """Tests for write-through SQLite persistence and fallback loading."""

    def test_exchange_persisted_to_sqlite(self, conv_service_with_db):
        svc, db, store = conv_service_with_db

        eid = svc.add_exchange(THREAD_ID, "greetings", {"message": "Hello"})

        rows = db.fetch_all(
            "SELECT * FROM thread_exchanges WHERE id = ?", (eid,)
        )
        assert len(rows) == 1
        assert rows[0]["thread_id"] == THREAD_ID
        assert rows[0]["topic"] == "greetings"
        assert rows[0]["prompt_message"] == "Hello"

    def test_response_persisted_to_sqlite(self, conv_service_with_db):
        svc, db, store = conv_service_with_db

        eid = svc.add_exchange(THREAD_ID, "topic", {"message": "Hi"})
        svc.add_response(THREAD_ID, "Hello back!", 1.5)

        rows = db.fetch_all(
            "SELECT * FROM thread_exchanges WHERE id = ?", (eid,)
        )
        assert rows[0]["response_message"] == "Hello back!"
        assert rows[0]["generation_time_ms"] == 1.5

    def test_error_persisted_to_sqlite(self, conv_service_with_db):
        svc, db, store = conv_service_with_db

        eid = svc.add_exchange(THREAD_ID, "topic", {"message": "Hi"})
        svc.add_response_error(THREAD_ID, "LLM timeout")

        rows = db.fetch_all(
            "SELECT * FROM thread_exchanges WHERE id = ?", (eid,)
        )
        assert rows[0]["response_error"] == "LLM timeout"
        assert rows[0]["response_message"] is None

    def test_steps_persisted_to_sqlite(self, conv_service_with_db):
        svc, db, store = conv_service_with_db

        eid = svc.add_exchange(THREAD_ID, "topic", {"message": "Do stuff"})
        svc.add_steps_to_exchange(THREAD_ID, [
            {"type": "recall", "description": "Look up data"},
        ])

        rows = db.fetch_all(
            "SELECT steps FROM thread_exchanges WHERE id = ?", (eid,)
        )
        steps = json.loads(rows[0]["steps"])
        assert len(steps) == 1
        assert steps[0]["description"] == "Look up data"

    def test_memory_chunk_persisted_to_sqlite(self, conv_service_with_db):
        svc, db, store = conv_service_with_db

        eid = svc.add_exchange(THREAD_ID, "topic", {"message": "Hi"})
        svc.add_memory_chunk(THREAD_ID, eid, {"gists": [{"content": "Greeting"}]})

        rows = db.fetch_all(
            "SELECT memory_chunk FROM thread_exchanges WHERE id = ?", (eid,)
        )
        chunk = json.loads(rows[0]["memory_chunk"])
        assert chunk["gists"][0]["content"] == "Greeting"

    def test_load_from_sqlite_on_empty_memorystore(self, conv_service_with_db):
        svc, db, store = conv_service_with_db

        # Add exchange with response (written to both MemoryStore and SQLite)
        eid = svc.add_exchange(THREAD_ID, "topic", {"message": "Hello"})
        svc.add_response(THREAD_ID, "Hi there!", 0.8)

        # Clear MemoryStore (simulates server restart)
        store.delete(svc._conv_key(THREAD_ID))
        store.delete(svc._index_key(THREAD_ID))

        # get_conversation_history should fall back to SQLite
        history = svc.get_conversation_history(THREAD_ID)
        assert len(history) == 1
        assert history[0]["prompt"]["message"] == "Hello"
        assert history[0]["response"]["message"] == "Hi there!"

    def test_paginated_history_falls_back_to_sqlite(self, conv_service_with_db):
        svc, db, store = conv_service_with_db

        for i in range(5):
            svc.add_exchange(THREAD_ID, "topic", {"message": f"Msg {i}"})
            svc.add_response(THREAD_ID, f"Reply {i}", 0.5)

        # Clear MemoryStore
        store.delete(svc._conv_key(THREAD_ID))
        store.delete(svc._index_key(THREAD_ID))

        # Paginated history should load from SQLite
        page = svc.get_paginated_history(THREAD_ID, limit=3, offset=0)
        assert page["total"] == 5
        assert len(page["exchanges"]) == 3
        assert page["has_more"] is True

    def test_sqlite_failure_does_not_block_memorystore(self, conv_service_with_db):
        svc, db, store = conv_service_with_db

        # Make SQLite fail
        svc._db_service = MagicMock()
        svc._db_service.execute.side_effect = Exception("DB down")

        # Should still work via MemoryStore
        eid = svc.add_exchange(THREAD_ID, "topic", {"message": "Still works"})
        assert eid
        history = svc.get_conversation_history(THREAD_ID)
        assert len(history) == 1
        assert history[0]["prompt"]["message"] == "Still works"

    def test_purge_keeps_max_exchanges(self, conv_service_with_db):
        svc, db, store = conv_service_with_db

        # Override max for test speed
        import services.thread_conversation_service as mod
        orig_max = mod.MAX_SQLITE_EXCHANGES
        orig_every = mod._PURGE_EVERY
        mod.MAX_SQLITE_EXCHANGES = 5
        mod._PURGE_EVERY = 1  # Purge check on every insert
        try:
            for i in range(8):
                svc.add_exchange(THREAD_ID, "topic", {"message": f"Msg {i}"})

            rows = db.fetch_all("SELECT COUNT(*) AS cnt FROM thread_exchanges")
            assert rows[0]["cnt"] <= 5
        finally:
            mod.MAX_SQLITE_EXCHANGES = orig_max
            mod._PURGE_EVERY = orig_every
