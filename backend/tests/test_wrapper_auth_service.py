"""
Unit tests for WrapperAuthService.

All tests use an in-memory SQLite database so they are fully isolated from the
production database.  The wrapper_tokens table is created from the migration
SQL inline to avoid a dependency on the migration runner.
"""

import hashlib
import json
import sqlite3

import pytest
from unittest.mock import MagicMock

from services.wrapper_auth_service import WrapperAuthService, _hash_token


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _create_in_memory_db():
    """Return a minimal DatabaseService-like object backed by an in-memory SQLite.

    Implements only the interface used by WrapperAuthService:
      - ``connection()`` context manager
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

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
    """WrapperAuthService backed by in-memory SQLite."""
    return WrapperAuthService(db)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_flask_request(authorization_header: str = "") -> MagicMock:
    """Build a mock Flask request with a given Authorization header."""
    req = MagicMock()
    # Use a MagicMock for headers so .get() can be freely overridden
    headers_mock = MagicMock()
    headers_mock.get = lambda key, default="": (
        authorization_header if key == "Authorization" else default
    )
    req.headers = headers_mock
    return req


# ---------------------------------------------------------------------------
# _hash_token
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestHashToken:
    def test_produces_sha256_hex(self):
        token = "hello"
        expected = hashlib.sha256(b"hello").hexdigest()
        assert _hash_token(token) == expected

    def test_length_is_64(self):
        assert len(_hash_token("anything")) == 64

    def test_deterministic(self):
        assert _hash_token("abc") == _hash_token("abc")

    def test_different_inputs_differ(self):
        assert _hash_token("aaa") != _hash_token("bbb")


# ---------------------------------------------------------------------------
# create_token
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCreateToken:
    def test_returns_raw_token_and_wrapper_id(self, svc):
        raw, wid = svc.create_token("Test Wrapper")
        assert isinstance(raw, str)
        assert len(raw) > 20
        assert wid.startswith("wrp_")

    def test_wrapper_id_is_unique(self, svc):
        _, wid1 = svc.create_token("W1")
        _, wid2 = svc.create_token("W2")
        assert wid1 != wid2

    def test_raw_token_is_unique(self, svc):
        raw1, _ = svc.create_token("W1")
        raw2, _ = svc.create_token("W2")
        assert raw1 != raw2

    def test_hash_is_stored_not_raw_token(self, svc, db):
        raw, wid = svc.create_token("W1")
        with db.connection() as conn:
            row = conn.execute(
                "SELECT token_hash FROM wrapper_tokens WHERE wrapper_id = ?", (wid,)
            ).fetchone()
        assert row is not None
        assert row[0] == _hash_token(raw)
        assert row[0] != raw

    def test_capabilities_stored(self, svc, db):
        caps = {"signals": ["context_change"], "intents": ["read_memory"]}
        _, wid = svc.create_token("W1", capabilities=caps)
        with db.connection() as conn:
            row = conn.execute(
                "SELECT capabilities FROM wrapper_tokens WHERE wrapper_id = ?", (wid,)
            ).fetchone()
        assert json.loads(row[0]) == caps

    def test_permissions_stored(self, svc, db):
        perms = {"query": ["memory"], "update": [], "broadcast": True}
        _, wid = svc.create_token("W1", permissions=perms)
        with db.connection() as conn:
            row = conn.execute(
                "SELECT permissions FROM wrapper_tokens WHERE wrapper_id = ?", (wid,)
            ).fetchone()
        assert json.loads(row[0]) == perms

    def test_defaults_to_empty_dicts(self, svc, db):
        _, wid = svc.create_token("W1")
        with db.connection() as conn:
            row = conn.execute(
                "SELECT capabilities, permissions, metadata FROM wrapper_tokens WHERE wrapper_id = ?",
                (wid,),
            ).fetchone()
        assert json.loads(row[0]) == {}
        assert json.loads(row[1]) == {}
        assert json.loads(row[2]) == {}

    def test_created_at_is_utc_isoformat(self, svc, db):
        _, wid = svc.create_token("W1")
        with db.connection() as conn:
            row = conn.execute(
                "SELECT created_at FROM wrapper_tokens WHERE wrapper_id = ?", (wid,)
            ).fetchone()
        # ISO 8601 with UTC offset
        assert "+00:00" in row[0] or "Z" in row[0]

    def test_revoked_at_is_null_on_creation(self, svc, db):
        _, wid = svc.create_token("W1")
        with db.connection() as conn:
            row = conn.execute(
                "SELECT revoked_at FROM wrapper_tokens WHERE wrapper_id = ?", (wid,)
            ).fetchone()
        assert row[0] is None


# ---------------------------------------------------------------------------
# validate_bearer
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestValidateBearer:
    def test_valid_token_returns_wrapper_id(self, svc):
        raw, wid = svc.create_token("W1")
        req = _make_flask_request(f"Bearer {raw}")
        result = svc.validate_bearer(req)
        assert result == wid

    def test_invalid_token_returns_none(self, svc):
        svc.create_token("W1")
        req = _make_flask_request("Bearer totally-wrong-token")
        assert svc.validate_bearer(req) is None

    def test_no_auth_header_returns_none(self, svc):
        req = _make_flask_request("")
        assert svc.validate_bearer(req) is None

    def test_non_bearer_scheme_returns_none(self, svc):
        raw, _ = svc.create_token("W1")
        req = _make_flask_request(f"Basic {raw}")
        assert svc.validate_bearer(req) is None

    def test_revoked_token_returns_none(self, svc):
        raw, wid = svc.create_token("W1")
        svc.revoke(wid)
        req = _make_flask_request(f"Bearer {raw}")
        assert svc.validate_bearer(req) is None

    def test_slides_last_seen_at(self, svc, db):
        raw, wid = svc.create_token("W1")
        req = _make_flask_request(f"Bearer {raw}")
        svc.validate_bearer(req)
        with db.connection() as conn:
            row = conn.execute(
                "SELECT last_seen_at FROM wrapper_tokens WHERE wrapper_id = ?", (wid,)
            ).fetchone()
        assert row[0] is not None

    def test_empty_bearer_value_returns_none(self, svc):
        req = _make_flask_request("Bearer ")
        assert svc.validate_bearer(req) is None


# ---------------------------------------------------------------------------
# check_permission
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCheckPermission:
    def test_query_permission_granted(self, svc):
        perms = {"query": ["memory", "threads"], "update": [], "broadcast": False}
        _, wid = svc.create_token("W1", permissions=perms)
        assert svc.check_permission(wid, "query", "memory") is True
        assert svc.check_permission(wid, "query", "threads") is True

    def test_query_permission_denied_for_unlisted_resource(self, svc):
        perms = {"query": ["memory"], "update": [], "broadcast": False}
        _, wid = svc.create_token("W1", permissions=perms)
        assert svc.check_permission(wid, "query", "tools") is False

    def test_update_permission_granted(self, svc):
        perms = {"query": [], "update": ["memory"], "broadcast": False}
        _, wid = svc.create_token("W1", permissions=perms)
        assert svc.check_permission(wid, "update", "memory") is True

    def test_broadcast_true(self, svc):
        perms = {"query": [], "update": [], "broadcast": True}
        _, wid = svc.create_token("W1", permissions=perms)
        assert svc.check_permission(wid, "broadcast", "") is True

    def test_broadcast_false(self, svc):
        perms = {"query": [], "update": [], "broadcast": False}
        _, wid = svc.create_token("W1", permissions=perms)
        assert svc.check_permission(wid, "broadcast", "") is False

    def test_missing_wrapper_returns_false(self, svc):
        assert svc.check_permission("nonexistent", "query", "memory") is False

    def test_revoked_wrapper_returns_false(self, svc):
        _, wid = svc.create_token("W1")
        svc.revoke(wid)
        assert svc.check_permission(wid, "query", "memory") is False

    def test_empty_permissions_denies_all(self, svc):
        _, wid = svc.create_token("W1", permissions={})
        assert svc.check_permission(wid, "query", "memory") is False
        assert svc.check_permission(wid, "update", "memory") is False
        assert svc.check_permission(wid, "broadcast", "") is False


# ---------------------------------------------------------------------------
# revoke
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRevoke:
    def test_revoke_returns_true_on_success(self, svc):
        _, wid = svc.create_token("W1")
        assert svc.revoke(wid) is True

    def test_revoke_sets_revoked_at(self, svc, db):
        _, wid = svc.create_token("W1")
        svc.revoke(wid)
        with db.connection() as conn:
            row = conn.execute(
                "SELECT revoked_at FROM wrapper_tokens WHERE wrapper_id = ?", (wid,)
            ).fetchone()
        assert row[0] is not None

    def test_revoke_nonexistent_returns_false(self, svc):
        assert svc.revoke("wrp_doesnotexist") is False

    def test_revoke_already_revoked_returns_false(self, svc):
        _, wid = svc.create_token("W1")
        svc.revoke(wid)
        assert svc.revoke(wid) is False

    def test_revoked_wrapper_hidden_from_get(self, svc):
        _, wid = svc.create_token("W1")
        svc.revoke(wid)
        assert svc.get_wrapper(wid) is None


# ---------------------------------------------------------------------------
# list_wrappers / get_wrapper
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestListAndGet:
    def test_list_returns_all_active(self, svc):
        _, wid1 = svc.create_token("W1")
        _, wid2 = svc.create_token("W2")
        result = svc.list_wrappers()
        ids = [w["wrapper_id"] for w in result]
        assert wid1 in ids
        assert wid2 in ids

    def test_list_excludes_revoked(self, svc):
        _, wid1 = svc.create_token("W1")
        _, wid2 = svc.create_token("W2")
        svc.revoke(wid1)
        result = svc.list_wrappers()
        ids = [w["wrapper_id"] for w in result]
        assert wid1 not in ids
        assert wid2 in ids

    def test_list_empty_when_none(self, svc):
        assert svc.list_wrappers() == []

    def test_get_returns_wrapper(self, svc):
        caps = {"signals": ["x"]}
        _, wid = svc.create_token("MyWrapper", capabilities=caps)
        result = svc.get_wrapper(wid)
        assert result is not None
        assert result["name"] == "MyWrapper"
        assert result["capabilities"] == caps

    def test_get_returns_none_for_missing(self, svc):
        assert svc.get_wrapper("wrp_nope") is None

    def test_get_returns_none_for_revoked(self, svc):
        _, wid = svc.create_token("W1")
        svc.revoke(wid)
        assert svc.get_wrapper(wid) is None

    def test_wrapper_dict_has_expected_keys(self, svc):
        _, wid = svc.create_token("W1")
        result = svc.get_wrapper(wid)
        for key in ("id", "wrapper_id", "name", "capabilities", "permissions",
                    "metadata", "last_seen_at", "created_at"):
            assert key in result


# ---------------------------------------------------------------------------
# update_capabilities
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestUpdateCapabilities:
    def test_update_returns_true(self, svc):
        _, wid = svc.create_token("W1")
        assert svc.update_capabilities(wid, {"signals": ["x"]}) is True

    def test_update_persists_new_capabilities(self, svc):
        _, wid = svc.create_token("W1", capabilities={"signals": ["old"]})
        new_caps = {"signals": ["new1", "new2"], "intents": ["write"]}
        svc.update_capabilities(wid, new_caps)
        result = svc.get_wrapper(wid)
        assert result["capabilities"] == new_caps

    def test_update_nonexistent_returns_false(self, svc):
        assert svc.update_capabilities("wrp_nope", {}) is False

    def test_update_revoked_returns_false(self, svc):
        _, wid = svc.create_token("W1")
        svc.revoke(wid)
        assert svc.update_capabilities(wid, {"signals": []}) is False
