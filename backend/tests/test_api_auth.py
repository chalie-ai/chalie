"""
Tests for backend/api/user_auth.py — authentication blueprint.

Covers /auth/status, /auth/register, /auth/login, and /auth/logout.
These tests do NOT bypass auth — they test the auth system itself.

Uses a self-contained minimal-schema fixture to avoid the sqlite-vec / vec0
extension requirement that prevents the shared ``db`` fixture from working in
environments where that native extension is unavailable.
"""

import pytest
from unittest.mock import patch, MagicMock
from flask import Flask
from werkzeug.security import generate_password_hash

import services.database_service as _db_mod
from services.database_service import DatabaseService
from api.user_auth import user_auth_bp


# ── Minimal schema — only tables touched by auth endpoints ────────────────────

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


# ── Per-test database fixture ──────────────────────────────────────────────────

@pytest.fixture()
def auth_db(tmp_path):
    """Provide a fresh, minimal SQLite database for auth endpoint tests.

    Creates an in-process SQLite database containing only the tables needed
    by the user_auth blueprint, then patches ``get_shared_db_service`` so
    every service in the call chain uses this isolated database.

    Avoids the sqlite-vec / vec0 extension requirement that makes the shared
    ``db`` fixture unavailable in most CI environments.

    Yields:
        sqlite3.Connection: Raw connection for seeding and inspecting test data.
    """
    db_path = str(tmp_path / "auth_test.db")
    db_service = DatabaseService(db_path)

    with db_service.connection() as conn:
        conn.executescript(_AUTH_TEST_SCHEMA)

    # Reset thread-local cache so the new path is used immediately
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


def _make_vault_mock(*, unlocked: bool = True, initialize_raises=None, unlock_returns=True):
    """Build a :class:`~unittest.mock.MagicMock` that mimics :class:`~services.vault_service.VaultService`.

    Args:
        unlocked:           Initial value returned by ``is_unlocked()``.
        initialize_raises:  If not ``None``, ``initialize()`` will raise this exception.
        unlock_returns:     Return value of ``unlock()``.

    Returns:
        A configured :class:`~unittest.mock.MagicMock` for use in ``patch`` calls
        targeting ``services.vault_service.get_vault_service``.
    """
    mock_vault = MagicMock()
    mock_vault.is_unlocked.return_value = unlocked
    mock_vault.unlock.return_value = unlock_returns
    mock_vault.lock.return_value = None
    mock_vault.get_state.return_value = "unlocked" if unlocked else "uninitialized"

    if initialize_raises is not None:
        mock_vault.initialize.side_effect = initialize_raises
    else:
        mock_vault.initialize.return_value = None

    return mock_vault


@pytest.mark.unit
class TestAuthAPI:
    """Test user authentication API endpoints."""

    @pytest.fixture
    def client(self, auth_db):
        """Create Flask test client with user_auth blueprint.

        Requires the ``auth_db`` fixture so that ``get_shared_db_service()``
        returns the test database (patched at module level by ``auth_db``).

        Args:
            auth_db: Minimal per-test SQLite database fixture.

        Yields:
            FlaskClient: Configured test client for the user_auth blueprint.
        """
        app = Flask(__name__)
        app.secret_key = 'test-secret-key'
        app.register_blueprint(user_auth_bp)
        app.config['TESTING'] = True
        return app.test_client()

    # ------------------------------------------------------------------
    # GET /auth/status
    # ------------------------------------------------------------------

    def test_status_returns_expected_keys(self, client, auth_db):
        """GET /auth/status returns has_master_account, has_providers, has_session, vault_state."""
        # Seed one master account, no providers
        auth_db.execute(
            "INSERT INTO master_account (username, password_hash) VALUES (?, ?)",
            ("admin", generate_password_hash("testpassword")),
        )
        auth_db.commit()

        mock_vault = _make_vault_mock(unlocked=False)
        with patch('services.auth_session_service.validate_session', return_value=False), \
             patch('services.vault_service.get_vault_service', return_value=mock_vault):
            response = client.get('/auth/status')

            assert response.status_code == 200
            data = response.get_json()
            assert data["has_master_account"] is True
            assert data["has_providers"] is False
            assert data["has_session"] is False
            assert "vault_state" in data

    def test_status_with_valid_session(self, client, auth_db):
        """GET /auth/status with valid session returns has_session true."""
        # Seed one master account and one active provider
        auth_db.execute(
            "INSERT INTO master_account (username, password_hash) VALUES (?, ?)",
            ("admin", generate_password_hash("testpassword")),
        )
        auth_db.execute(
            "INSERT INTO providers (name, platform, model, is_active) VALUES (?, ?, ?, ?)",
            ("test-provider", "openai", "gpt-4", 1),
        )
        auth_db.commit()

        mock_vault = _make_vault_mock(unlocked=True)
        with patch('services.auth_session_service.validate_session', return_value=True), \
             patch('services.vault_service.get_vault_service', return_value=mock_vault):
            response = client.get('/auth/status')

            assert response.status_code == 200
            data = response.get_json()
            assert data["has_master_account"] is True
            assert data["has_providers"] is True
            assert data["has_session"] is True
            assert data["vault_state"] == "unlocked"

    def test_status_vault_state_unlocked_when_vault_open(self, client, auth_db):
        """GET /auth/status reports vault_state 'unlocked' when vault is open."""
        mock_vault = _make_vault_mock(unlocked=True)
        with patch('services.auth_session_service.validate_session', return_value=False), \
             patch('services.vault_service.get_vault_service', return_value=mock_vault):
            response = client.get('/auth/status')
            assert response.status_code == 200
            assert response.get_json()["vault_state"] == "unlocked"

    def test_status_vault_state_uninitialized_when_no_config_row(self, client, auth_db):
        """GET /auth/status reports 'uninitialized' when vault_config has no row."""
        # vault_config table exists but has no row
        mock_vault = _make_vault_mock(unlocked=False)
        with patch('services.auth_session_service.validate_session', return_value=False), \
             patch('services.vault_service.get_vault_service', return_value=mock_vault):
            response = client.get('/auth/status')
            assert response.status_code == 200
            assert response.get_json()["vault_state"] == "uninitialized"

    # ------------------------------------------------------------------
    # POST /auth/register
    # ------------------------------------------------------------------

    def test_register_short_password_returns_400(self, client, auth_db):
        """POST /auth/register with short password returns 400."""
        response = client.post(
            '/auth/register',
            json={"username": "admin", "password": "short"},
            content_type='application/json',
        )

        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
        assert "8 characters" in data["error"].lower() or "password" in data["error"].lower()

    def test_register_missing_username_returns_400(self, client, auth_db):
        """POST /auth/register with missing username returns 400."""
        response = client.post(
            '/auth/register',
            json={"password": "securepassword123"},
            content_type='application/json',
        )

        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
        assert "username" in data["error"].lower()

    def test_register_success_returns_201(self, client, auth_db):
        """POST /auth/register creates account, initialises vault, returns 201.

        Verifies that:
        - The master account row is inserted into the database.
        - ``VaultService.initialize()`` is called once with the chosen password.
        - ``VaultService.unlock()`` is called once with the chosen password.
        - The response body contains ``ok: True`` and ``vault_state: "unlocked"``.
        - The session cookie creation hook fires exactly once.
        """
        mock_vault = _make_vault_mock(unlocked=True)
        with patch('services.auth_session_service.create_session') as mock_create_session, \
             patch('services.vault_service.get_vault_service', return_value=mock_vault):
            response = client.post(
                '/auth/register',
                json={"username": "admin", "password": "securepassword123"},
                content_type='application/json',
            )

            assert response.status_code == 201
            data = response.get_json()
            assert data["ok"] is True
            assert data.get("vault_state") == "unlocked"

            # Verify the account was actually inserted
            row = auth_db.execute("SELECT COUNT(*) FROM master_account").fetchone()
            assert row[0] == 1

            mock_create_session.assert_called_once()
            mock_vault.initialize.assert_called_once_with("securepassword123")
            mock_vault.unlock.assert_called_once_with("securepassword123")

    def test_register_vault_failure_returns_500(self, client, auth_db):
        """POST /auth/register returns 500 when vault.initialize() raises."""
        mock_vault = _make_vault_mock(initialize_raises=RuntimeError("disk full"))
        with patch('services.auth_session_service.create_session'), \
             patch('services.vault_service.get_vault_service', return_value=mock_vault):
            response = client.post(
                '/auth/register',
                json={"username": "admin", "password": "securepassword123"},
                content_type='application/json',
            )

            assert response.status_code == 500
            data = response.get_json()
            assert "error" in data

    def test_register_duplicate_returns_409(self, client, auth_db):
        """POST /auth/register when account already exists returns 409."""
        # Pre-seed a master account
        auth_db.execute(
            "INSERT INTO master_account (username, password_hash) VALUES (?, ?)",
            ("admin", generate_password_hash("existingpassword")),
        )
        auth_db.commit()

        response = client.post(
            '/auth/register',
            json={"username": "admin", "password": "securepassword123"},
            content_type='application/json',
        )

        assert response.status_code == 409
        data = response.get_json()
        assert "error" in data
        assert "already exists" in data["error"].lower()

    # ------------------------------------------------------------------
    # POST /auth/login
    # ------------------------------------------------------------------

    def test_login_missing_credentials_returns_400(self, client, auth_db):
        """POST /auth/login with missing credentials returns 400."""
        response = client.post(
            '/auth/login',
            json={"username": "admin"},
            content_type='application/json',
        )

        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data

    def test_login_invalid_credentials_returns_401(self, client, auth_db):
        """POST /auth/login with invalid credentials returns 401."""
        # No user in the database — login should fail
        response = client.post(
            '/auth/login',
            json={"username": "admin", "password": "wrongpassword"},
            content_type='application/json',
        )

        assert response.status_code == 401
        data = response.get_json()
        assert "error" in data
        assert "invalid" in data["error"].lower()

    def test_login_success_returns_200(self, client, auth_db):
        """POST /auth/login with valid credentials unlocks vault, reconnects caps, returns 200.

        Verifies that:
        - ``VaultService.unlock()`` is called with the user's password.
        - ``_reconnect_capabilities()`` is called after the vault opens.
        - The response body contains ``ok: True`` and ``vault_state: "unlocked"``.
        - The session cookie creation hook fires exactly once.
        """
        test_password = "securepassword123"
        stored_hash = generate_password_hash(test_password)

        auth_db.execute(
            "INSERT INTO master_account (username, password_hash) VALUES (?, ?)",
            ("admin", stored_hash),
        )
        auth_db.commit()

        mock_vault = _make_vault_mock(unlocked=True, unlock_returns=True)
        with patch('services.auth_session_service.create_session') as mock_create_session, \
             patch('services.vault_service.get_vault_service', return_value=mock_vault), \
             patch('api.user_auth._reconnect_capabilities') as mock_reconnect:
            response = client.post(
                '/auth/login',
                json={"username": "admin", "password": test_password},
                content_type='application/json',
            )

            assert response.status_code == 200
            data = response.get_json()
            assert data["ok"] is True
            assert data.get("vault_state") == "unlocked"

            mock_create_session.assert_called_once()
            mock_vault.unlock.assert_called_once_with(test_password)
            mock_reconnect.assert_called_once()

    def test_login_vault_unlock_false_returns_401(self, client, auth_db):
        """POST /auth/login returns 401 when vault.unlock() returns False (vault inconsistency)."""
        test_password = "securepassword123"
        auth_db.execute(
            "INSERT INTO master_account (username, password_hash) VALUES (?, ?)",
            ("admin", generate_password_hash(test_password)),
        )
        auth_db.commit()

        # Vault returns False despite correct user password — signals vault inconsistency
        mock_vault = _make_vault_mock(unlock_returns=False)
        with patch('services.vault_service.get_vault_service', return_value=mock_vault):
            response = client.post(
                '/auth/login',
                json={"username": "admin", "password": test_password},
                content_type='application/json',
            )

            assert response.status_code == 401
            data = response.get_json()
            assert "error" in data
            assert "invalid" in data["error"].lower()

    def test_login_vault_uninitialized_auto_initializes(self, client, auth_db):
        """POST /auth/login auto-initializes the vault for existing users upgrading.

        When ``vault.get_state()`` returns ``"uninitialized"``, login calls
        ``initialize(password)`` before ``unlock(password)`` so existing users
        get a working vault on first login after the upgrade.
        """
        test_password = "securepassword123"
        auth_db.execute(
            "INSERT INTO master_account (username, password_hash) VALUES (?, ?)",
            ("admin", generate_password_hash(test_password)),
        )
        auth_db.commit()

        mock_vault = _make_vault_mock(unlocked=False, unlock_returns=True)
        # get_state returns "uninitialized" (from unlocked=False in _make_vault_mock)
        with patch('services.auth_session_service.create_session'), \
             patch('services.vault_service.get_vault_service', return_value=mock_vault), \
             patch('api.user_auth._reconnect_capabilities'):
            response = client.post(
                '/auth/login',
                json={"username": "admin", "password": test_password},
                content_type='application/json',
            )

            assert response.status_code == 200
            data = response.get_json()
            assert data["ok"] is True
            assert data.get("vault_state") == "unlocked"
            mock_vault.initialize.assert_called_once_with(test_password)
            mock_vault.unlock.assert_called_once_with(test_password)

    # ------------------------------------------------------------------
    # POST /auth/logout
    # ------------------------------------------------------------------

    def test_logout_returns_ok(self, client, auth_db):
        """POST /auth/logout seals the vault, destroys session, and returns ok."""
        mock_vault = _make_vault_mock()
        with patch('services.auth_session_service.destroy_session') as mock_destroy, \
             patch('services.vault_service.get_vault_service', return_value=mock_vault):
            response = client.post('/auth/logout')

            assert response.status_code == 200
            data = response.get_json()
            assert data["ok"] is True
            mock_destroy.assert_called_once()
            mock_vault.lock.assert_called_once()

    def test_logout_vault_lock_failure_still_destroys_session(self, client, auth_db):
        """POST /auth/logout destroys the session even when vault.lock() raises.

        Ensures that a vault failure during logout is non-fatal — the session
        cookie is still cleared so the user is logged out of the application.
        """
        mock_vault = MagicMock()
        mock_vault.lock.side_effect = Exception("unexpected error")

        with patch('services.auth_session_service.destroy_session') as mock_destroy, \
             patch('services.vault_service.get_vault_service', return_value=mock_vault):
            response = client.post('/auth/logout')

            assert response.status_code == 200
            data = response.get_json()
            assert data["ok"] is True
            mock_destroy.assert_called_once()
