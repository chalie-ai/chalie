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
        conv_service.add_exchange(THREAD_ID, "topic", {"message": "First"})
        eid2 = conv_service.add_exchange(THREAD_ID, "topic", {"message": "Second"})

        assert conv_service.get_latest_exchange_id(THREAD_ID) == eid2

    def test_returns_unknown_for_empty(self, conv_service):
        assert conv_service.get_latest_exchange_id(THREAD_ID) == "unknown"


class TestRemoveExchanges:
    def test_removes_specific_exchanges(self, conv_service):
        conv_service.add_exchange(THREAD_ID, "topic", {"message": "Keep"})
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
        svc.add_exchange(THREAD_ID, "topic", {"message": "Hello"})
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

    def test_durable_history_returns_exchanges_after_restart(self, conv_service_with_db):
        """get_paginated_history_durable returns exchanges from SQLite when MemoryStore is empty."""
        svc, db, store = conv_service_with_db

        # Populate 5 exchanges with responses
        for i in range(5):
            svc.add_exchange(THREAD_ID, "greetings", {"message": f"Hello {i}"})
            svc.add_response(THREAD_ID, f"Hi {i}!", 0.5)

        # Clear MemoryStore (simulates server restart)
        store.delete(svc._conv_key(THREAD_ID))
        store.delete(svc._index_key(THREAD_ID))

        # get_paginated_history_durable reads directly from SQLite
        page = svc.get_paginated_history_durable(THREAD_ID, limit=12, offset=0)
        assert page["total"] == 5
        assert len(page["exchanges"]) == 5
        assert page["has_more"] is False
        # Verify chronological order (oldest first)
        assert page["exchanges"][0]["prompt"]["message"] == "Hello 0"
        assert page["exchanges"][4]["prompt"]["message"] == "Hello 4"
        # Verify responses are present
        assert page["exchanges"][0]["response"]["message"] == "Hi 0!"

    def test_durable_history_pagination(self, conv_service_with_db):
        """get_paginated_history_durable paginates correctly from the end."""
        svc, db, store = conv_service_with_db

        for i in range(10):
            svc.add_exchange(THREAD_ID, "topic", {"message": f"Msg {i}"})
            svc.add_response(THREAD_ID, f"Reply {i}", 0.3)

        store.delete(svc._conv_key(THREAD_ID))
        store.delete(svc._index_key(THREAD_ID))

        # Page 1: most recent 3
        p1 = svc.get_paginated_history_durable(THREAD_ID, limit=3, offset=0)
        assert p1["total"] == 10
        assert len(p1["exchanges"]) == 3
        assert p1["has_more"] is True
        # With ORDER BY rowid DESC OFFSET 0 LIMIT 3, we get the 3 most recent
        assert p1["exchanges"][0]["prompt"]["message"] == "Msg 7"
        assert p1["exchanges"][2]["prompt"]["message"] == "Msg 9"

        # Page 2: next 3
        p2 = svc.get_paginated_history_durable(THREAD_ID, limit=3, offset=3)
        assert len(p2["exchanges"]) == 3
        assert p2["has_more"] is True
        assert p2["exchanges"][0]["prompt"]["message"] == "Msg 4"
        assert p2["exchanges"][2]["prompt"]["message"] == "Msg 6"

    def test_durable_history_spans_multiple_threads(self, conv_service_with_db):
        """get_paginated_history_durable returns exchanges across all threads for the same channel."""
        svc, db, store = conv_service_with_db

        # Thread 1 — old conversation
        thread1 = "telegram:user1:chan1:1"
        for i in range(3):
            svc.add_exchange(thread1, "topic", {"message": f"Thread1 Msg {i}"})
            svc.add_response(thread1, f"Thread1 Reply {i}", 0.3)

        # Thread 2 — new conversation (same channel prefix)
        thread2 = "telegram:user1:chan1:2"
        for i in range(2):
            svc.add_exchange(thread2, "topic", {"message": f"Thread2 Msg {i}"})
            svc.add_response(thread2, f"Thread2 Reply {i}", 0.3)

        store.delete(svc._conv_key(thread1))
        store.delete(svc._conv_key(thread2))

        # Query from thread2 should return all 5 exchanges across both threads
        page = svc.get_paginated_history_durable(thread2, limit=12, offset=0)
        assert page["total"] == 5
        assert len(page["exchanges"]) == 5
        # Chronological: thread1 first, then thread2
        assert page["exchanges"][0]["prompt"]["message"] == "Thread1 Msg 0"
        assert page["exchanges"][3]["prompt"]["message"] == "Thread2 Msg 0"


class TestRestartScenario:
    """Full restart scenario — thread + exchanges in SQLite, empty MemoryStore."""

    def test_get_most_recent_thread_id_finds_active_thread(self, conv_service_with_db):
        """After restart, get_most_recent_thread_id finds the active thread from SQLite."""
        svc, db, store = conv_service_with_db

        # Create threads table in the same in-memory DB
        with db.connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS threads (
                    thread_id TEXT PRIMARY KEY,
                    channel_id TEXT NOT NULL,
                    platform TEXT NOT NULL DEFAULT 'unknown',
                    state TEXT NOT NULL DEFAULT 'active',
                    current_topic TEXT,
                    topic_history TEXT DEFAULT '[]',
                    exchange_count INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now')),
                    last_activity TEXT DEFAULT (datetime('now')),
                    expired_at TEXT,
                    summary TEXT
                )
            """)
            conn.execute("""
                INSERT INTO threads (thread_id, channel_id, platform, state, last_activity)
                VALUES (?, 'default', 'unknown', 'active', datetime('now'))
            """, (THREAD_ID,))

        # Add exchanges
        svc.add_exchange(THREAD_ID, "test", {"message": "Hello"})
        svc.add_response(THREAD_ID, "Hi!", 0.5)

        # Clear MemoryStore
        store.delete(svc._conv_key(THREAD_ID))
        store.delete(svc._index_key(THREAD_ID))

        # Patch get_shared_db_service where it's imported (locally inside the method)
        with patch('services.database_service.get_shared_db_service', return_value=db):
            thread_id, from_expired = svc.get_most_recent_thread_id()
            assert thread_id == THREAD_ID
            assert from_expired is False

    def test_full_api_restart_flow(self, conv_service_with_db):
        """Full API test: populate data, wipe MemoryStore, GET /conversation/recent returns history."""
        svc, db, store = conv_service_with_db

        # Create threads table
        with db.connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS threads (
                    thread_id TEXT PRIMARY KEY,
                    channel_id TEXT NOT NULL,
                    platform TEXT NOT NULL DEFAULT 'unknown',
                    state TEXT NOT NULL DEFAULT 'active',
                    current_topic TEXT,
                    topic_history TEXT DEFAULT '[]',
                    exchange_count INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now')),
                    last_activity TEXT DEFAULT (datetime('now')),
                    expired_at TEXT,
                    summary TEXT
                )
            """)
            conn.execute("""
                INSERT INTO threads (thread_id, channel_id, platform, state, last_activity)
                VALUES (?, 'default', 'unknown', 'active', datetime('now'))
            """, (THREAD_ID,))

        # Add 3 exchanges with responses
        for i in range(3):
            svc.add_exchange(THREAD_ID, "greetings", {"message": f"Hello {i}"})
            svc.add_response(THREAD_ID, f"Hi {i}!", 0.5)

        # === SIMULATE RESTART: wipe MemoryStore ===
        store.delete(svc._conv_key(THREAD_ID))
        store.delete(svc._index_key(THREAD_ID))
        store.delete("active_thread:default")

        # Create a fresh MemoryStore (simulates complete restart)
        from services.memory_store import MemoryStore
        fresh_store = MemoryStore()

        # Build Flask test app
        from flask import Flask
        from api.conversation import conversation_bp
        app = Flask(__name__)
        app.register_blueprint(conversation_bp)
        app.config['TESTING'] = True
        client = app.test_client()

        # Mock ThreadService to use fresh (empty) store
        mock_ts = MagicMock()
        mock_ts.get_active_thread_id.return_value = None  # MemoryStore is empty

        # Patch at the module where it's looked up
        with patch('services.auth_session_service.validate_session', return_value=True), \
             patch('services.thread_service.get_thread_service', return_value=mock_ts), \
             patch('services.database_service.get_shared_db_service', return_value=db):

            # Create a new TCS that will use our in-memory DB
            fresh_tcs = ThreadConversationService()
            fresh_tcs._db_service = db

            with patch('services.thread_conversation_service.ThreadConversationService', return_value=fresh_tcs), \
                 patch('services.thread_conversation_service.MemoryClientService.create_connection', return_value=fresh_store):
                response = client.get('/conversation/recent?limit=12&offset=0')

        assert response.status_code == 200
        data = response.get_json()

        # THIS is the critical assertion: after restart, we must see our history
        assert data["thread_id"] == THREAD_ID
        assert len(data["exchanges"]) == 3
        assert data["total"] == 3
        assert data["exchanges"][0]["prompt"] == "Hello 0"
        assert data["exchanges"][2]["prompt"] == "Hello 2"

    def test_fallback_to_thread_exchanges_when_threads_table_empty(self, conv_service_with_db):
        """If threads table has no match, fall back to thread_exchanges for thread_id."""
        svc, db, store = conv_service_with_db

        # Create threads table but leave it EMPTY
        with db.connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS threads (
                    thread_id TEXT PRIMARY KEY,
                    channel_id TEXT NOT NULL,
                    platform TEXT NOT NULL DEFAULT 'unknown',
                    state TEXT NOT NULL DEFAULT 'active',
                    current_topic TEXT,
                    topic_history TEXT DEFAULT '[]',
                    exchange_count INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now')),
                    last_activity TEXT DEFAULT (datetime('now')),
                    expired_at TEXT,
                    summary TEXT
                )
            """)

        # But thread_exchanges HAS data (thread persist failed, exchange persist succeeded)
        svc.add_exchange(THREAD_ID, "test", {"message": "Orphaned exchange"})
        svc.add_response(THREAD_ID, "Still here!", 0.5)

        store.delete(svc._conv_key(THREAD_ID))
        store.delete(svc._index_key(THREAD_ID))

        with patch('services.database_service.get_shared_db_service', return_value=db):
            thread_id, from_expired = svc.get_most_recent_thread_id()
            assert thread_id == THREAD_ID
            # Recovered from exchanges → treated as expired
            assert from_expired is True

    def test_expired_thread_still_returns_history(self, conv_service_with_db):
        """Even if the thread is expired, history is still returned."""
        svc, db, store = conv_service_with_db

        with db.connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS threads (
                    thread_id TEXT PRIMARY KEY,
                    channel_id TEXT NOT NULL,
                    platform TEXT NOT NULL DEFAULT 'unknown',
                    state TEXT NOT NULL DEFAULT 'active',
                    current_topic TEXT,
                    topic_history TEXT DEFAULT '[]',
                    exchange_count INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now')),
                    last_activity TEXT DEFAULT (datetime('now')),
                    expired_at TEXT,
                    summary TEXT
                )
            """)
            # Insert as expired
            conn.execute("""
                INSERT INTO threads (thread_id, channel_id, platform, state, expired_at)
                VALUES (?, 'default', 'unknown', 'expired', datetime('now'))
            """, (THREAD_ID,))

        svc.add_exchange(THREAD_ID, "topic", {"message": "Before expiry"})
        svc.add_response(THREAD_ID, "Reply before expiry", 0.5)

        store.delete(svc._conv_key(THREAD_ID))
        store.delete(svc._index_key(THREAD_ID))

        with patch('services.database_service.get_shared_db_service', return_value=db):
            thread_id, from_expired = svc.get_most_recent_thread_id()
            assert thread_id == THREAD_ID
            assert from_expired is True

            page = svc.get_paginated_history_durable(thread_id, limit=12, offset=0)
            assert page["total"] == 1
            assert page["exchanges"][0]["prompt"]["message"] == "Before expiry"
