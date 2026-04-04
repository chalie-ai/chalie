"""
Tests for backend/api/user_auth.py — authentication blueprint.

Covers /auth/status, /auth/register, /auth/login, and /auth/logout.
These tests do NOT bypass auth — they test the auth system itself.

Uses a self-contained minimal-schema fixture to avoid the sqlite-vec
/ vec0 extension requirement that prevents the shared ``db`` fixture
from working in environments where that native extension is
unavailable.
"""

import pytest
from unittest.mock import patch, MagicMock
from flask import Flask
from werkzeug.security import generate_password_hash

import services.database_service as _db_mod
from services.database_service import DatabaseService
from api.user_auth import user_auth_bp

# Short aliases for frequently-patched module paths
_P_VALIDATE = "services.auth_session_service.validate_session"
_P_CREATE = "services.auth_session_service.create_session"
_P_DESTROY = "services.auth_session_service.destroy_session"
_P_VAULT = "services.vault_service.get_vault_service"
_P_RECONNECT = "api.user_auth._reconnect_capabilities"


# ── Minimal schema — only tables touched by auth endpoints ──

_AUTH_TEST_SCHEMA = """
CREATE TABLE IF NOT EXISTS master_account (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT    NOT NULL UNIQUE,
    password_hash TEXT    NOT NULL,
    created_at    TEXT    DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS providers (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    name      TEXT,
    platform  TEXT,
    model     TEXT,
    is_active INTEGER NOT NULL DEFAULT 0,
    api_key   BLOB
);

CREATE TABLE IF NOT EXISTS vault_config (
    id             INTEGER PRIMARY KEY CHECK (id = 1),
    kdf_salt       BLOB    NOT NULL,
    kdf_algorithm  TEXT    NOT NULL DEFAULT 'pbkdf2_sha256',
    kdf_iterations INTEGER NOT NULL DEFAULT 600000,
    wrapped_dek    BLOB    NOT NULL,
    dek_nonce      BLOB    NOT NULL,
    created_at     TEXT,
    updated_at     TEXT
);
"""


# ── Per-test database fixture ────────────────────────────────

@pytest.fixture()
def auth_db(tmp_path):
    """Provide a fresh, minimal SQLite database for auth tests.

    Creates an in-process SQLite database containing only the tables
    needed by the user_auth blueprint, then patches
    ``get_shared_db_service`` so every service in the call chain
    uses this isolated database.

    Yields:
        sqlite3.Connection: Raw connection for seeding / inspecting.
    """
    db_path = str(tmp_path / "auth_test.db")
    db_service = DatabaseService(db_path)

    with db_service.connection() as conn:
        conn.executescript(_AUTH_TEST_SCHEMA)

    _db_mod._local.conn = None
    _db_mod._local.db_path = None

    original = _db_mod._shared_db_service
    _db_mod._shared_db_service = db_service

    raw_conn = db_service._get_connection()
    try:
        yield raw_conn
    finally:
        db_service.close_pool()
        _db_mod._shared_db_service = original
        _db_mod._local.conn = None
        _db_mod._local.db_path = None


def _make_vault_mock(
    *,
    unlocked: bool = True,
    state: str | None = None,
    initialize_raises=None,
    unlock_returns=True,
):
    """Build a MagicMock that mimics VaultService.

    Args:
        unlocked:          Value returned by ``is_unlocked()``.
        state:             Explicit vault state override. When ``None``,
                           derived from *unlocked* (``"unlocked"`` / ``"uninitialized"``).
        initialize_raises: If set, ``initialize()`` raises this.
        unlock_returns:    Return value of ``unlock()``.
    """
    mock_vault = MagicMock()
    mock_vault.is_unlocked.return_value = unlocked
    mock_vault.unlock.return_value = unlock_returns
    mock_vault.lock.return_value = None
    if state is None:
        state = "unlocked" if unlocked else "uninitialized"
    mock_vault.get_state.return_value = state

    if initialize_raises is not None:
        mock_vault.initialize.side_effect = initialize_raises
    else:
        mock_vault.initialize.return_value = None

    return mock_vault


def _seed_account(db, password="testpassword"):
    """Insert a master_account row and return the password."""
    db.execute(
        "INSERT INTO master_account "
        "(username, password_hash) VALUES (?, ?)",
        ("admin", generate_password_hash(password)),
    )
    db.commit()
    return password


@pytest.mark.unit
class TestAuthAPI:
    """Test user authentication API endpoints."""

    @pytest.fixture
    def client(self, auth_db):
        """Create Flask test client with user_auth blueprint."""
        app = Flask(__name__)
        app.secret_key = 'test-secret-key'
        app.register_blueprint(user_auth_bp)
        app.config['TESTING'] = True
        return app.test_client()

    # ── GET /auth/status ─────────────────────────────────────

    def test_status_returns_expected_keys(self, client, auth_db):
        """GET /auth/status returns the four expected keys."""
        _seed_account(auth_db)

        mv = _make_vault_mock(unlocked=False)
        with patch(_P_VALIDATE, return_value=False), \
             patch(_P_VAULT, return_value=mv):
            resp = client.get('/auth/status')

            assert resp.status_code == 200
            data = resp.get_json()
            assert data["has_master_account"] is True
            assert data["has_providers"] is False
            assert data["has_session"] is False
            assert "vault_state" in data

    def test_status_with_valid_session(self, client, auth_db):
        """GET /auth/status with valid session returns True."""
        _seed_account(auth_db)
        auth_db.execute(
            "INSERT INTO providers "
            "(name, platform, model, is_active) "
            "VALUES (?, ?, ?, ?)",
            ("test-provider", "openai", "gpt-4", 1),
        )
        auth_db.commit()

        mv = _make_vault_mock(unlocked=True)
        with patch(_P_VALIDATE, return_value=True), \
             patch(_P_VAULT, return_value=mv):
            resp = client.get('/auth/status')

            assert resp.status_code == 200
            data = resp.get_json()
            assert data["has_master_account"] is True
            assert data["has_providers"] is True
            assert data["has_session"] is True
            assert data["vault_state"] == "unlocked"

    def test_status_vault_state_unlocked_when_vault_open(
        self, client, auth_db,
    ):
        """GET /auth/status reports 'unlocked' when open."""
        mv = _make_vault_mock(unlocked=True)
        with patch(_P_VALIDATE, return_value=False), \
             patch(_P_VAULT, return_value=mv):
            resp = client.get('/auth/status')
            assert resp.status_code == 200
            assert resp.get_json()["vault_state"] == "unlocked"

    def test_status_session_forced_false_when_vault_locked(
        self, client, auth_db,
    ):
        """Session valid + vault locked → has_session must be False.

        After a server restart the session cookie survives but the
        in-memory DEK is gone.  The status endpoint must force a
        re-login so the vault can be unlocked with the password.
        """
        _seed_account(auth_db)
        mv = _make_vault_mock(unlocked=False, state="locked")
        with patch(_P_VALIDATE, return_value=True), \
             patch(_P_VAULT, return_value=mv):
            resp = client.get('/auth/status')

            assert resp.status_code == 200
            data = resp.get_json()
            assert data["has_session"] is False
            assert data["vault_state"] == "locked"

    def test_status_vault_state_uninitialized_when_no_config_row(
        self, client, auth_db,
    ):
        """GET /auth/status reports 'uninitialized' with no row."""
        mv = _make_vault_mock(unlocked=False)
        with patch(_P_VALIDATE, return_value=False), \
             patch(_P_VAULT, return_value=mv):
            resp = client.get('/auth/status')
            assert resp.status_code == 200
            j = resp.get_json()
            assert j["vault_state"] == "uninitialized"

    # ── POST /auth/register ──────────────────────────────────

    def test_register_short_password_returns_400(
        self, client, auth_db,
    ):
        """POST /auth/register with short password returns 400."""
        resp = client.post(
            '/auth/register',
            json={"username": "admin", "password": "short"},
            content_type='application/json',
        )

        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data
        err = data["error"].lower()
        assert "8 characters" in err or "password" in err

    def test_register_missing_username_returns_400(
        self, client, auth_db,
    ):
        """POST /auth/register with no username returns 400."""
        resp = client.post(
            '/auth/register',
            json={"password": "securepassword123"},
            content_type='application/json',
        )

        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data
        assert "username" in data["error"].lower()

    def test_register_success_returns_201(self, client, auth_db):
        """POST /auth/register creates account + vault, 201."""
        mv = _make_vault_mock(unlocked=True)
        with patch(_P_CREATE) as mock_session, \
             patch(_P_VAULT, return_value=mv):
            resp = client.post(
                '/auth/register',
                json={
                    "username": "admin",
                    "password": "securepassword123",
                },
                content_type='application/json',
            )

            assert resp.status_code == 201
            data = resp.get_json()
            assert data["ok"] is True
            assert data.get("vault_state") == "unlocked"

            row = auth_db.execute(
                "SELECT COUNT(*) FROM master_account"
            ).fetchone()
            assert row[0] == 1

            mock_session.assert_called_once()
            mv.initialize.assert_called_once_with(
                "securepassword123",
            )
            mv.unlock.assert_called_once_with(
                "securepassword123",
            )

    def test_register_vault_failure_returns_500(
        self, client, auth_db,
    ):
        """POST /auth/register returns 500 on vault failure."""
        mv = _make_vault_mock(
            initialize_raises=RuntimeError("disk full"),
        )
        with patch(_P_CREATE), \
             patch(_P_VAULT, return_value=mv):
            resp = client.post(
                '/auth/register',
                json={
                    "username": "admin",
                    "password": "securepassword123",
                },
                content_type='application/json',
            )

            assert resp.status_code == 500
            assert "error" in resp.get_json()

    def test_register_duplicate_returns_409(
        self, client, auth_db,
    ):
        """POST /auth/register when account exists returns 409."""
        _seed_account(auth_db, "existingpassword")

        resp = client.post(
            '/auth/register',
            json={
                "username": "admin",
                "password": "securepassword123",
            },
            content_type='application/json',
        )

        assert resp.status_code == 409
        data = resp.get_json()
        assert "error" in data
        assert "already exists" in data["error"].lower()

    # ── POST /auth/login ─────────────────────────────────────

    def test_login_missing_credentials_returns_400(
        self, client, auth_db,
    ):
        """POST /auth/login with missing credentials → 400."""
        resp = client.post(
            '/auth/login',
            json={"username": "admin"},
            content_type='application/json',
        )

        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_login_invalid_credentials_returns_401(
        self, client, auth_db,
    ):
        """POST /auth/login with invalid credentials → 401."""
        resp = client.post(
            '/auth/login',
            json={
                "username": "admin",
                "password": "wrongpassword",
            },
            content_type='application/json',
        )

        assert resp.status_code == 401
        data = resp.get_json()
        assert "error" in data
        assert "invalid" in data["error"].lower()

    def test_login_success_returns_200(self, client, auth_db):
        """POST /auth/login with valid creds unlocks vault."""
        pw = _seed_account(auth_db, "securepassword123")

        mv = _make_vault_mock(unlocked=True, unlock_returns=True)
        with patch(_P_CREATE) as mock_session, \
             patch(_P_VAULT, return_value=mv), \
             patch(_P_RECONNECT) as mock_rc:
            resp = client.post(
                '/auth/login',
                json={"username": "admin", "password": pw},
                content_type='application/json',
            )

            assert resp.status_code == 200
            data = resp.get_json()
            assert data["ok"] is True
            assert data.get("vault_state") == "unlocked"

            mock_session.assert_called_once()
            mv.unlock.assert_called_once_with(pw)
            mock_rc.assert_called_once()

    def test_login_vault_unlock_false_returns_401(
        self, client, auth_db,
    ):
        """POST /auth/login → 401 when unlock() returns False."""
        pw = _seed_account(auth_db, "securepassword123")

        # unlock=False signals vault inconsistency
        mv = _make_vault_mock(unlock_returns=False)
        with patch(_P_VAULT, return_value=mv):
            resp = client.post(
                '/auth/login',
                json={"username": "admin", "password": pw},
                content_type='application/json',
            )

            assert resp.status_code == 401
            data = resp.get_json()
            assert "error" in data
            assert "invalid" in data["error"].lower()

    def test_login_vault_uninitialized_auto_initializes(
        self, client, auth_db,
    ):
        """Login auto-initializes vault for upgrading users.

        When ``get_state()`` returns ``"uninitialized"``, login
        calls ``initialize(password)`` before ``unlock(password)``
        so existing users get a working vault on first login.
        """
        pw = _seed_account(auth_db, "securepassword123")

        mv = _make_vault_mock(
            unlocked=False, unlock_returns=True,
        )
        with patch(_P_CREATE), \
             patch(_P_VAULT, return_value=mv), \
             patch(_P_RECONNECT):
            resp = client.post(
                '/auth/login',
                json={"username": "admin", "password": pw},
                content_type='application/json',
            )

            assert resp.status_code == 200
            data = resp.get_json()
            assert data["ok"] is True
            assert data.get("vault_state") == "unlocked"
            mv.initialize.assert_called_once_with(pw)
            mv.unlock.assert_called_once_with(pw)

    # ── POST /auth/logout ────────────────────────────────────

    def test_logout_returns_ok(self, client, auth_db):
        """POST /auth/logout seals vault and destroys session."""
        mv = _make_vault_mock()
        with patch(_P_DESTROY) as mock_destroy, \
             patch(_P_VAULT, return_value=mv):
            resp = client.post('/auth/logout')

            assert resp.status_code == 200
            assert resp.get_json()["ok"] is True
            mock_destroy.assert_called_once()
            mv.lock.assert_called_once()

    def test_logout_vault_lock_failure_still_destroys_session(
        self, client, auth_db,
    ):
        """Logout destroys session even when vault.lock() fails.

        Ensures that a vault failure during logout is non-fatal —
        the session cookie is still cleared so the user is logged
        out of the application.
        """
        mv = MagicMock()
        mv.lock.side_effect = Exception("unexpected error")

        with patch(_P_DESTROY) as mock_destroy, \
             patch(_P_VAULT, return_value=mv):
            resp = client.post('/auth/logout')

            assert resp.status_code == 200
            assert resp.get_json()["ok"] is True
            mock_destroy.assert_called_once()


@pytest.mark.unit
class TestSessionPersistence:
    """Sessions must survive MemoryStore wipes (container restart)."""

    @pytest.fixture
    def session_db(self, tmp_path):
        """Provide a SQLite DB with the auth_sessions table."""
        db_path = str(tmp_path / "session_test.db")
        db_service = DatabaseService(db_path)
        with db_service.connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS auth_sessions (
                    token TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                )
            """)

        _db_mod._local.conn = None
        _db_mod._local.db_path = None

        original = _db_mod._shared_db_service
        _db_mod._shared_db_service = db_service
        try:
            yield db_service
        finally:
            db_service.close_pool()
            _db_mod._shared_db_service = original
            _db_mod._local.conn = None
            _db_mod._local.db_path = None

    def test_create_session_persists_to_sqlite(self, session_db):
        """create_session writes token to both MemoryStore and SQLite."""
        from services.auth_session_service import create_session
        from services.memory_store import MemoryStore

        store = MemoryStore()
        response = MagicMock()

        with patch('services.memory_client.MemoryClientService.create_connection', return_value=store):
            token = create_session(response)

        # MemoryStore has it
        assert store.exists(f"auth_session:{token}")

        # SQLite has it
        rows = session_db.fetch_all(
            "SELECT token FROM auth_sessions WHERE token = ?", (token,)
        )
        assert len(rows) == 1
        assert rows[0]["token"] == token

    def test_validate_session_survives_memorystore_wipe(self, session_db):
        """After MemoryStore is wiped, validate_session falls back to SQLite."""
        from services.auth_session_service import create_session, validate_session
        from services.memory_store import MemoryStore

        # Phase 1: create session with one store
        store1 = MemoryStore()
        response = MagicMock()
        with patch('services.memory_client.MemoryClientService.create_connection', return_value=store1):
            token = create_session(response)

        # Phase 2: simulate restart — fresh empty MemoryStore
        store2 = MemoryStore()

        # Build a request mock with the session cookie
        request_mock = MagicMock()
        request_mock.cookies.get.return_value = token

        with patch('services.memory_client.MemoryClientService.create_connection', return_value=store2):
            # First validate: MemoryStore misses, SQLite hits, rehydrates
            assert validate_session(request_mock) is True

            # MemoryStore should now have the session (rehydrated)
            assert store2.exists(f"auth_session:{token}")

            # Second validate: MemoryStore hits directly (fast path)
            assert validate_session(request_mock) is True

    def test_destroy_session_removes_from_sqlite(self, session_db):
        """destroy_session removes from both MemoryStore and SQLite."""
        from services.auth_session_service import create_session, destroy_session
        from services.memory_store import MemoryStore

        store = MemoryStore()
        create_response = MagicMock()
        with patch('services.memory_client.MemoryClientService.create_connection', return_value=store):
            token = create_session(create_response)

        # Confirm it exists
        rows = session_db.fetch_all("SELECT token FROM auth_sessions WHERE token = ?", (token,))
        assert len(rows) == 1

        # Destroy
        request_mock = MagicMock()
        request_mock.cookies.get.return_value = token
        destroy_response = MagicMock()
        with patch('services.memory_client.MemoryClientService.create_connection', return_value=store):
            destroy_session(request_mock, destroy_response)

        # Gone from SQLite
        rows = session_db.fetch_all("SELECT token FROM auth_sessions WHERE token = ?", (token,))
        assert len(rows) == 0
