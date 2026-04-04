"""Tests for InterfaceRegistryService — interface lifecycle management."""

import sqlite3
import pytest
from unittest.mock import patch, MagicMock
from datetime import timedelta

from services.time_utils import utc_now


def _create_in_memory_db():
    """Return a minimal DatabaseService-like object backed by an in-memory SQLite."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row

    # wrapper_tokens (migration 011)
    conn.execute("""
        CREATE TABLE wrapper_tokens (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            token_hash TEXT NOT NULL UNIQUE,
            wrapper_id TEXT NOT NULL UNIQUE,
            capabilities TEXT NOT NULL DEFAULT '{}',
            permissions TEXT NOT NULL DEFAULT '{}',
            metadata TEXT NOT NULL DEFAULT '{}',
            last_seen_at TEXT,
            created_at TEXT NOT NULL,
            revoked_at TEXT
        )
    """)

    # interfaces (migration 012)
    conn.execute("""
        CREATE TABLE interfaces (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            host TEXT NOT NULL,
            port INTEGER NOT NULL,
            signal_token_hash TEXT,
            status TEXT NOT NULL DEFAULT 'paired',
            capabilities_hash TEXT,
            last_seen_at TEXT,
            paired_at TEXT NOT NULL,
            metadata TEXT NOT NULL DEFAULT '{}'
        )
    """)
    conn.execute("""
        CREATE TABLE interface_tools (
            interface_id TEXT NOT NULL REFERENCES interfaces(id) ON DELETE CASCADE,
            tool_name TEXT NOT NULL,
            manifest_json TEXT NOT NULL,
            registered_at TEXT NOT NULL,
            PRIMARY KEY (interface_id, tool_name)
        )
    """)
    conn.execute("""
        CREATE TABLE interface_pairing_keys (
            key_hash TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            used_at TEXT
        )
    """)
    conn.commit()

    class _FakeDB:
        """Minimal DatabaseService shim wrapping a real in-memory sqlite3 connection."""

        def __init__(self, _conn):
            self._conn = _conn

        class _Ctx:
            def __init__(self, conn):
                self._conn = conn

            def __enter__(self):
                return self._conn

            def __exit__(self, exc_type, _exc_val, _exc_tb):
                if exc_type:
                    self._conn.rollback()
                else:
                    self._conn.commit()
                return False

        def connection(self):
            return self._Ctx(self._conn)

    return _FakeDB(conn)


@pytest.fixture
def db():
    """Isolated in-memory database for each test."""
    return _create_in_memory_db()


@pytest.fixture
def svc(db):
    """InterfaceRegistryService with in-memory DB."""
    from services.interface_registry_service import InterfaceRegistryService
    return InterfaceRegistryService(db=db)


@pytest.fixture
def mock_interface_server():
    """Mock HTTP responses for a healthy interface."""
    health_response = MagicMock()
    health_response.status_code = 200
    health_response.json.return_value = {"status": "ok", "name": "Test Calendar", "version": "1.0.0"}
    health_response.raise_for_status = MagicMock()

    capabilities_response = MagicMock()
    capabilities_response.status_code = 200
    capabilities_response.json.return_value = [
        {
            "name": "create_event",
            "description": "Create a calendar event",
            "parameters": [
                {"name": "title", "type": "string", "required": True},
                {"name": "start", "type": "string", "required": True},
            ],
        },
        {
            "name": "query_events",
            "description": "Query calendar events",
            "parameters": [
                {"name": "start", "type": "string", "required": True},
                {"name": "end", "type": "string", "required": True},
            ],
        },
    ]
    capabilities_response.raise_for_status = MagicMock()

    def mock_get(url, **kwargs):
        if "/health" in url:
            return health_response
        if "/capabilities" in url:
            return capabilities_response
        raise Exception(f"Unexpected GET: {url}")

    return mock_get


class TestGeneratePairingKey:
    @pytest.mark.unit
    def test_generates_key_with_expiry(self, svc):
        with patch('runtime_config.get', side_effect=lambda k, d=None: {"host": "192.168.1.10", "port": 8081}.get(k, d)):
            result = svc.generate_pairing_key()

        assert "pairing_key" in result
        assert len(result["pairing_key"]) > 10
        assert result["expires_in_seconds"] == 600

    @pytest.mark.unit
    def test_key_stored_as_hash(self, svc, db):
        with patch('runtime_config.get', side_effect=lambda k, d=None: {"host": "localhost", "port": 8081}.get(k, d)):
            result = svc.generate_pairing_key()

        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM interface_pairing_keys")
            count = cursor.fetchone()[0]

        assert count == 1
        # Raw key should NOT be stored
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT key_hash FROM interface_pairing_keys")
            stored_hash = cursor.fetchone()[0]

        assert stored_hash != result["pairing_key"]


class TestPairing:
    @pytest.mark.unit
    def test_successful_pairing(self, svc, db, mock_interface_server):
        # Generate key
        with patch('runtime_config.get', side_effect=lambda k, d=None: {"host": "localhost", "port": 8081}.get(k, d)):
            key_result = svc.generate_pairing_key()

        # Pair
        with patch('services.interface_registry_service.requests') as mock_requests, \
             patch('services.tool_registry_service.ToolRegistryService') as MockRegistry:
            mock_requests.get = mock_interface_server
            mock_registry_instance = MagicMock()
            MockRegistry.return_value = mock_registry_instance

            result = svc.pair(
                pairing_key=key_result["pairing_key"],
                name="Test Calendar",
                host="192.168.1.50",
                port=3100,
            )

        assert result["status"] == "online"
        assert result["interface_id"]
        assert len(result["capabilities"]) == 2
        assert "create_event" in result["capabilities"]

    @pytest.mark.unit
    def test_invalid_pairing_key(self, svc):
        with pytest.raises(ValueError, match="Invalid pairing key"):
            svc.pair("bogus-key", "Test", "localhost", 3100)

    @pytest.mark.unit
    def test_expired_pairing_key(self, svc, db):
        from services.interface_registry_service import _hash

        # Insert an expired key directly
        now = utc_now()
        expired = now - timedelta(hours=1)
        key_hash = _hash("expired-key")

        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO interface_pairing_keys (key_hash, created_at, expires_at) VALUES (?, ?, ?)",
                (key_hash, expired.isoformat(), expired.isoformat()),
            )
            cursor.close()

        with pytest.raises(ValueError, match="expired"):
            svc.pair("expired-key", "Test", "localhost", 3100)

    @pytest.mark.unit
    def test_reused_pairing_key(self, svc, db, mock_interface_server):
        with patch('runtime_config.get', side_effect=lambda k, d=None: {"host": "localhost", "port": 8081}.get(k, d)):
            key_result = svc.generate_pairing_key()

        # First use succeeds
        with patch('services.interface_registry_service.requests') as mock_requests, \
             patch('services.tool_registry_service.ToolRegistryService'):
            mock_requests.get = mock_interface_server
            svc.pair(key_result["pairing_key"], "Test", "localhost", 3100)

        # Second use fails
        with pytest.raises(ValueError, match="already used"):
            svc.pair(key_result["pairing_key"], "Test2", "localhost", 3101)

    @pytest.mark.unit
    def test_unreachable_interface(self, svc, db):
        with patch('runtime_config.get', side_effect=lambda k, d=None: {"host": "localhost", "port": 8081}.get(k, d)):
            key_result = svc.generate_pairing_key()

        with patch('services.interface_registry_service.requests') as mock_requests:
            mock_requests.get.side_effect = Exception("Connection refused")

            with pytest.raises(ValueError, match="not reachable"):
                svc.pair(key_result["pairing_key"], "Test", "localhost", 3100)


class TestHealthCheck:
    @pytest.mark.unit
    def test_healthy_interface(self, svc, db, mock_interface_server):
        # Setup: pair an interface
        with patch('runtime_config.get', side_effect=lambda k, d=None: {"host": "localhost", "port": 8081}.get(k, d)):
            key_result = svc.generate_pairing_key()

        with patch('services.interface_registry_service.requests') as mock_requests, \
             patch('services.tool_registry_service.ToolRegistryService'):
            mock_requests.get = mock_interface_server
            pair_result = svc.pair(key_result["pairing_key"], "Test", "localhost", 3100)

        # Health check
        with patch('services.interface_registry_service.requests') as mock_requests:
            mock_requests.get = mock_interface_server
            result = svc.health_check(pair_result["interface_id"])

        assert result is True

    @pytest.mark.unit
    def test_offline_after_3_failures(self, svc, db, mock_interface_server):
        # Setup: pair an interface
        with patch('runtime_config.get', side_effect=lambda k, d=None: {"host": "localhost", "port": 8081}.get(k, d)):
            key_result = svc.generate_pairing_key()

        with patch('services.interface_registry_service.requests') as mock_requests, \
             patch('services.tool_registry_service.ToolRegistryService'):
            mock_requests.get = mock_interface_server
            pair_result = svc.pair(key_result["pairing_key"], "Test", "localhost", 3100)

        interface_id = pair_result["interface_id"]

        # 3 failures
        with patch('services.interface_registry_service.requests') as mock_requests, \
             patch('services.tool_registry_service.ToolRegistryService'):
            mock_requests.get.side_effect = Exception("timeout")
            svc.health_check(interface_id)
            svc.health_check(interface_id)
            svc.health_check(interface_id)

        iface = svc.get_interface(interface_id)
        assert iface["status"] == "offline"

    @pytest.mark.unit
    def test_recovery_from_offline(self, svc, db, mock_interface_server):
        # Setup: pair and force offline
        with patch('runtime_config.get', side_effect=lambda k, d=None: {"host": "localhost", "port": 8081}.get(k, d)):
            key_result = svc.generate_pairing_key()

        with patch('services.interface_registry_service.requests') as mock_requests, \
             patch('services.tool_registry_service.ToolRegistryService'):
            mock_requests.get = mock_interface_server
            pair_result = svc.pair(key_result["pairing_key"], "Test", "localhost", 3100)

        interface_id = pair_result["interface_id"]

        # Force offline
        with patch('services.interface_registry_service.requests') as mock_requests, \
             patch('services.tool_registry_service.ToolRegistryService'):
            mock_requests.get.side_effect = Exception("timeout")
            for _ in range(3):
                svc.health_check(interface_id)

        assert svc.get_interface(interface_id)["status"] == "offline"

        # Recovery
        with patch('services.interface_registry_service.requests') as mock_requests, \
             patch('services.tool_registry_service.ToolRegistryService'):
            mock_requests.get = mock_interface_server
            result = svc.health_check(interface_id)

        assert result is True
        assert svc.get_interface(interface_id)["status"] == "online"


class TestExecuteCapability:
    @pytest.mark.unit
    def test_successful_execution(self, svc, db, mock_interface_server):
        # Setup: pair
        with patch('runtime_config.get', side_effect=lambda k, d=None: {"host": "localhost", "port": 8081}.get(k, d)):
            key_result = svc.generate_pairing_key()

        with patch('services.interface_registry_service.requests') as mock_requests, \
             patch('services.tool_registry_service.ToolRegistryService'):
            mock_requests.get = mock_interface_server
            pair_result = svc.pair(key_result["pairing_key"], "Test", "localhost", 3100)

        # Execute
        execute_response = MagicMock()
        execute_response.status_code = 200
        execute_response.json.return_value = {
            "text": "Event created: Meeting with Bob",
            "data": {"event_id": "abc123"},
            "error": None,
        }
        execute_response.raise_for_status = MagicMock()

        with patch('services.interface_registry_service.requests') as mock_requests:
            mock_requests.post.return_value = execute_response
            result = svc.execute_capability(
                pair_result["interface_id"],
                "create_event",
                {"title": "Meeting with Bob", "start": "2026-03-18T15:00:00"},
            )

        assert result["text"] == "Event created: Meeting with Bob"
        assert result["error"] is None

    @pytest.mark.unit
    def test_execution_on_offline_interface(self, svc, db, mock_interface_server):
        # Setup: pair then force offline
        with patch('runtime_config.get', side_effect=lambda k, d=None: {"host": "localhost", "port": 8081}.get(k, d)):
            key_result = svc.generate_pairing_key()

        with patch('services.interface_registry_service.requests') as mock_requests, \
             patch('services.tool_registry_service.ToolRegistryService'):
            mock_requests.get = mock_interface_server
            pair_result = svc.pair(key_result["pairing_key"], "Test", "localhost", 3100)

        svc._set_status(pair_result["interface_id"], "offline")

        result = svc.execute_capability(pair_result["interface_id"], "create_event", {})
        assert "offline" in result["error"]


class TestRemove:
    @pytest.mark.unit
    def test_remove_interface(self, svc, db, mock_interface_server):
        # Setup: pair
        with patch('runtime_config.get', side_effect=lambda k, d=None: {"host": "localhost", "port": 8081}.get(k, d)):
            key_result = svc.generate_pairing_key()

        with patch('services.interface_registry_service.requests') as mock_requests, \
             patch('services.tool_registry_service.ToolRegistryService'):
            mock_requests.get = mock_interface_server
            pair_result = svc.pair(key_result["pairing_key"], "Test", "localhost", 3100)

        # Remove
        with patch('services.tool_registry_service.ToolRegistryService'), \
             patch('services.wrapper_auth_service.WrapperAuthService'):
            svc.remove(pair_result["interface_id"])

        assert svc.get_interface(pair_result["interface_id"]) is None
        assert len(svc.list_interfaces()) == 0


class TestListAndGet:
    @pytest.mark.unit
    def test_list_empty(self, svc):
        assert svc.list_interfaces() == []

    @pytest.mark.unit
    def test_get_nonexistent(self, svc):
        assert svc.get_interface("nonexistent-id") is None
