"""
Tests for backend/services/settings_service.py

Covers: get/set settings, sensitive value masking, API key generation.
Uses a minimal SQLite fixture (no vec0 / sqlite-vec) so these tests run
anywhere without the full production schema extension.
"""

import pytest
from unittest.mock import patch

import services.database_service as _db_mod
import services.vault_service as _vault_mod
from services.database_service import DatabaseService
from services.settings_service import SettingsService
from services.vault_service import _vault_state, get_vault_service


# ── Minimal schema: only tables touched by SettingsService / VaultService ─────

_SETTINGS_TEST_SCHEMA = """
CREATE TABLE IF NOT EXISTS vault_config (
    id              INTEGER PRIMARY KEY CHECK (id = 1),
    kdf_salt        BLOB    NOT NULL,
    kdf_algorithm   TEXT    NOT NULL DEFAULT 'pbkdf2_sha256',
    kdf_iterations  INTEGER NOT NULL DEFAULT 600000,
    wrapped_dek     BLOB    NOT NULL,
    dek_nonce       BLOB    NOT NULL,
    created_at      TEXT,
    updated_at      TEXT
);

CREATE TABLE IF NOT EXISTS encryption_keys (
    id          INTEGER PRIMARY KEY CHECK (id = 1),
    key_value   TEXT    NOT NULL,
    created_at  TEXT    DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS settings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    key             TEXT UNIQUE NOT NULL,
    value           TEXT,
    value_type      TEXT DEFAULT 'string',
    description     TEXT,
    is_sensitive    INTEGER NOT NULL DEFAULT 0,
    encrypted_value BLOB,
    created_at      TEXT DEFAULT (datetime('now')),
    updated_at      TEXT DEFAULT (datetime('now'))
);
"""


# ── Per-test database fixture ─────────────────────────────────────────────────

@pytest.fixture()
def settings_db(tmp_path):
    """Provide a fresh, minimal SQLite database and patch the shared DB singleton.

    Creates an in-process SQLite database containing only the tables needed by
    SettingsService and VaultService tests, then patches ``get_shared_db_service``
    so that service factories return instances backed by this test database.

    Yields:
        sqlite3.Connection: A raw connection for seeding / inspecting test data.
    """
    db_path = str(tmp_path / "settings_test.db")

    db_service = DatabaseService(db_path)

    with db_service.connection() as conn:
        conn.executescript(_SETTINGS_TEST_SCHEMA)

    # Reset thread-local connection cache so the new path is used
    _db_mod._local.conn = None
    _db_mod._local.db_path = None

    original_singleton = _db_mod._shared_db_service
    _db_mod._shared_db_service = db_service
    _vault_mod._vault_service_instance = None  # reset cached VaultService

    raw_conn = db_service._get_connection()
    try:
        yield raw_conn
    finally:
        db_service.close_pool()
        _db_mod._shared_db_service = original_singleton
        _vault_mod._vault_service_instance = None  # reset cached VaultService
        _db_mod._local.conn = None
        _db_mod._local.db_path = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_service(settings_db) -> SettingsService:
    """Return a SettingsService wired to the test DB singleton.

    Args:
        settings_db: Raw connection from the ``settings_db`` fixture (acts as
            a dependency signal that the singleton is patched).

    Returns:
        A :class:`~services.settings_service.SettingsService` instance.
    """
    return SettingsService(_db_mod._shared_db_service)


def _unlock_vault(settings_db, password: str = "test-password") -> None:
    """Initialise and unlock the vault so sensitive settings can be encrypted.

    Args:
        settings_db: Raw connection from the ``settings_db`` fixture.
        password:    Master password to use for vault initialisation.
    """
    vault = get_vault_service()
    vault.initialize(password)
    vault.unlock(password)


# ── Test class ────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestSettingsService:
    """Full test coverage for SettingsService.get / set / delete / get_all."""

    @pytest.fixture(autouse=True)
    def _clear_vault_state(self):
        """Reset the module-level _vault_state before and after every test.

        Prevents vault lock/unlock state from one test leaking into the next.
        """
        _vault_state.dek = None
        yield
        _vault_state.dek = None

    # ── get ───────────────────────────────────────────────────────────

    def test_get_returns_value_when_found(self, settings_db):
        """get should return the resolved value when a matching row exists."""
        settings_db.execute(
            "INSERT INTO settings (key, value, is_sensitive, value_type) "
            "VALUES ('theme', 'my-setting-value', 0, 'string')"
        )
        settings_db.commit()

        service = _make_service(settings_db)
        value = service.get('theme')

        assert value == 'my-setting-value'

    def test_get_returns_none_when_not_found(self, settings_db):
        """get should return None when no row matches the key."""
        service = _make_service(settings_db)
        value = service.get('nonexistent_key')

        assert value is None

    def test_get_decrypts_sensitive_value(self, settings_db):
        """get should decrypt a vault-encrypted sensitive setting."""
        import base64

        _unlock_vault(settings_db)
        vault = get_vault_service()

        # Encrypt via vault and store as base64 blob (as set() would)
        encrypted_blob = base64.b64encode(vault.encrypt_str("s3cr3t")).decode()
        settings_db.execute(
            "INSERT INTO settings (key, encrypted_value, is_sensitive, value_type) "
            "VALUES ('my_secret', ?, 1, 'string')",
            (encrypted_blob,)
        )
        settings_db.commit()

        service = _make_service(settings_db)
        value = service.get('my_secret')

        assert value == 's3cr3t'

    # ── set ───────────────────────────────────────────────────────────

    def test_set_creates_new_setting(self, settings_db):
        """set should INSERT when no existing row is found for the key."""
        service = _make_service(settings_db)
        returned = service.set('new_key', 'new_value', 'string', 'A new setting')

        assert returned == 'new_value'

        row = settings_db.execute(
            "SELECT value, value_type, description FROM settings WHERE key = 'new_key'"
        ).fetchone()
        assert row is not None
        assert row[0] == 'new_value'
        assert row[1] == 'string'
        assert row[2] == 'A new setting'

    def test_set_updates_existing_setting(self, settings_db):
        """set should UPDATE when an existing non-sensitive row is found."""
        settings_db.execute(
            "INSERT INTO settings (key, value, is_sensitive, value_type) "
            "VALUES ('existing_key', 'old_value', 0, 'string')"
        )
        settings_db.commit()

        service = _make_service(settings_db)
        returned = service.set('existing_key', 'updated_value')

        assert returned == 'updated_value'

        row = settings_db.execute(
            "SELECT value FROM settings WHERE key = 'existing_key'"
        ).fetchone()
        assert row[0] == 'updated_value'

    def test_set_encrypts_sensitive_value_via_vault(self, settings_db):
        """set should encrypt and store a sensitive setting using VaultService."""
        import base64

        _unlock_vault(settings_db)

        # Seed a sensitive row (simulating what migrations create)
        settings_db.execute(
            "INSERT INTO settings (key, value, is_sensitive, value_type) "
            "VALUES ('db_password', NULL, 1, 'string')"
        )
        settings_db.commit()

        service = _make_service(settings_db)
        returned = service.set('db_password', 'hunter2')

        assert returned == 'hunter2'

        # Encrypted blob should be stored in encrypted_value, NOT value
        row = settings_db.execute(
            "SELECT value, encrypted_value FROM settings WHERE key = 'db_password'"
        ).fetchone()
        assert row[0] is None, "value column should be NULL for sensitive settings"
        assert row[1] is not None, "encrypted_value should be populated"

        # Ensure round-trip decrypt succeeds
        vault = get_vault_service()
        decrypted = vault.decrypt_str(base64.b64decode(row[1]))
        assert decrypted == 'hunter2'

    # ── get_all ──────────────────────────────────────────────────────

    def test_get_all_masks_sensitive_values(self, settings_db):
        """get_all should return '***' for sensitive settings."""
        settings_db.execute(
            "INSERT INTO settings (key, value, encrypted_value, is_sensitive, value_type) "
            "VALUES ('api_key', NULL, 'encrypted-blob', 1, 'string')"
        )
        settings_db.execute(
            "INSERT INTO settings (key, value, is_sensitive, value_type) "
            "VALUES ('theme', 'dark', 0, 'string')"
        )
        settings_db.execute(
            "INSERT INTO settings (key, value, encrypted_value, is_sensitive, value_type) "
            "VALUES ('db_password', NULL, 'another-blob', 1, 'string')"
        )
        settings_db.commit()

        service = _make_service(settings_db)
        settings = service.get_all()

        assert settings['api_key'] == '***'
        assert settings['db_password'] == '***'

    def test_get_all_returns_non_sensitive_values(self, settings_db):
        """get_all should return plain text values for non-sensitive settings."""
        settings_db.execute(
            "INSERT INTO settings (key, value, is_sensitive, value_type) "
            "VALUES ('theme', 'dark', 0, 'string')"
        )
        settings_db.execute(
            "INSERT INTO settings (key, value, is_sensitive, value_type) "
            "VALUES ('language', 'en', 0, 'string')"
        )
        settings_db.commit()

        service = _make_service(settings_db)
        settings = service.get_all()

        assert settings['theme'] == 'dark'
        assert settings['language'] == 'en'
        assert len(settings) == 2

    # ── get_api_key_or_generate ──────────────────────────────────────

    def test_get_api_key_or_generate_returns_existing(self, settings_db):
        """get_api_key_or_generate should return existing key without generating."""
        settings_db.execute(
            "INSERT INTO settings (key, value, is_sensitive, value_type) "
            "VALUES ('api_key', 'existing-api-key-abc123', 0, 'string')"
        )
        settings_db.commit()

        service = _make_service(settings_db)
        key = service.get_api_key_or_generate()

        assert key == 'existing-api-key-abc123'

    def test_get_api_key_or_generate_creates_new_when_none(self, settings_db):
        """get_api_key_or_generate should generate and store a new key when absent."""
        service = _make_service(settings_db)
        with patch('services.settings_service.secrets.token_urlsafe', return_value='generated-key-xyz789'):
            key = service.get_api_key_or_generate()

        assert key == 'generated-key-xyz789'

        # Verify the key was persisted — read back through the service
        stored = service.get('api_key')
        assert stored == 'generated-key-xyz789'
