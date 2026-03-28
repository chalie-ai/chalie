"""
Unit tests for services/vault_service.py — VaultService, VaultLockedError,
_VaultState, and get_vault_service factory.

These tests use a self-contained, minimal SQLite fixture that avoids the
vec0/sqlite-vec extension (not available in all CI environments).  Only the
tables needed by VaultService and its fallback paths are created.
"""

import base64
import hashlib
import sqlite3
import tempfile
import os
import shutil

import pytest

import services.database_service as _db_mod
from services.database_service import DatabaseService
from services.vault_service import (
    VaultService,
    VaultLockedError,
    _vault_state,
    get_vault_service,
)


# ── Minimal schema used by these tests ────────────────────────────────────────
# Only tables touched by VaultService or referenced in its fallback paths.

_VAULT_TEST_SCHEMA = """
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

CREATE TABLE IF NOT EXISTS providers (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT,
    platform        TEXT,
    model           TEXT,
    api_key         BLOB
);
"""


# ── Per-test database fixture ──────────────────────────────────────────────────

@pytest.fixture()
def vault_db(tmp_path):
    """Provide a fresh, minimal SQLite database and patch the shared DB singleton.

    Creates an in-process SQLite database containing only the tables needed by
    VaultService tests, then patches ``get_shared_db_service`` so that
    :func:`get_vault_service` returns a service backed by this test database.

    Yields:
        sqlite3.Connection: A raw connection for seeding/inspecting test data.
    """
    db_path = str(tmp_path / "vault_test.db")

    # Build DatabaseService for this test
    db_service = DatabaseService(db_path)

    # Create minimal schema
    with db_service.connection() as conn:
        conn.executescript(_VAULT_TEST_SCHEMA)

    # Reset thread-local connection cache so the new path is used
    _db_mod._local.conn = None
    _db_mod._local.db_path = None

    # Patch the process-wide singleton
    original_singleton = _db_mod._shared_db_service
    _db_mod._shared_db_service = db_service

    # Yield a raw connection for test data seeding
    raw_conn = db_service._get_connection()
    try:
        yield raw_conn
    finally:
        db_service.close_pool()
        _db_mod._shared_db_service = original_singleton
        _db_mod._local.conn = None
        _db_mod._local.db_path = None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_vault(vault_db) -> VaultService:
    """Return a VaultService wired to the test DB singleton.

    Args:
        vault_db: Raw connection from the ``vault_db`` fixture (acts as a
            dependency signal that the singleton is patched).

    Returns:
        A :class:`VaultService` instance.
    """
    return get_vault_service()


def _make_legacy_fernet_token(raw_key: str, plaintext: str) -> bytes:
    """Create a legacy Fernet-encrypted token matching the existing app pattern.

    Replicates the key-derivation used by ``ProviderDbService._encrypt`` so
    the migration-window fallback tests use a realistic ciphertext.

    Args:
        raw_key:    The raw key string as stored in ``encryption_keys.key_value``.
        plaintext:  The plaintext string to encrypt.

    Returns:
        UTF-8 bytes of the Fernet token (the form stored in legacy DB rows).
    """
    from cryptography.fernet import Fernet

    key_bytes = hashlib.sha256(raw_key.encode()).digest()
    fernet_key = base64.urlsafe_b64encode(key_bytes)
    f = Fernet(fernet_key)
    return f.encrypt(plaintext.encode("utf-8"))


# ── Test class ─────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestVaultService:
    """Full test coverage for VaultService, VaultLockedError, and _vault_state."""

    @pytest.fixture(autouse=True)
    def _clear_vault_state(self):
        """Reset the module-level _vault_state before and after every test.

        Prevents vault lock/unlock state from one test bleeding into the next,
        regardless of execution order.
        """
        _vault_state.dek = None
        yield
        _vault_state.dek = None

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def test_initialize_creates_vault_config_row(self, vault_db):
        """initialize() inserts exactly one vault_config row with expected columns."""
        vault = _make_vault(vault_db)
        vault.initialize("s3cr3tP@ss!")

        cursor = vault_db.cursor()
        cursor.execute(
            "SELECT kdf_salt, kdf_algorithm, kdf_iterations, wrapped_dek, dek_nonce "
            "FROM vault_config WHERE id = 1"
        )
        row = cursor.fetchone()
        cursor.close()

        assert row is not None, "vault_config row should exist after initialize()"
        kdf_salt, kdf_algorithm, kdf_iterations, wrapped_dek, dek_nonce = row

        assert len(bytes(kdf_salt)) == 32, "kdf_salt should be 32 bytes"
        assert kdf_algorithm == "pbkdf2_sha256"
        assert kdf_iterations == 600_000
        assert len(bytes(wrapped_dek)) > 0, "wrapped_dek should be non-empty"
        assert len(bytes(dek_nonce)) == 12, "dek_nonce should be 12 bytes"

    def test_initialize_replaces_existing_row(self, vault_db):
        """Calling initialize() twice replaces the previous vault_config row."""
        vault = _make_vault(vault_db)
        vault.initialize("first-password")

        cursor = vault_db.cursor()
        cursor.execute("SELECT kdf_salt FROM vault_config WHERE id = 1")
        first_salt = bytes(cursor.fetchone()[0])
        cursor.close()

        vault.initialize("second-password")

        cursor = vault_db.cursor()
        cursor.execute("SELECT kdf_salt FROM vault_config WHERE id = 1")
        second_salt = bytes(cursor.fetchone()[0])

        # Count rows — should still be exactly 1
        cursor.execute("SELECT COUNT(*) FROM vault_config")
        count = cursor.fetchone()[0]
        cursor.close()

        assert count == 1, "vault_config should still have exactly one row"
        assert first_salt != second_salt, "new salt should differ from old salt"

    # ------------------------------------------------------------------
    # Unlock
    # ------------------------------------------------------------------

    def test_unlock_with_correct_password(self, vault_db):
        """unlock() with the correct password returns True and caches the DEK."""
        vault = _make_vault(vault_db)
        vault.initialize("correct-password")

        assert not vault.is_unlocked()
        result = vault.unlock("correct-password")

        assert result is True
        assert vault.is_unlocked()
        assert _vault_state.dek is not None
        assert len(_vault_state.dek) == 32

    def test_unlock_with_wrong_password_returns_false(self, vault_db):
        """unlock() with a wrong password returns False and leaves vault sealed."""
        vault = _make_vault(vault_db)
        vault.initialize("correct-password")

        result = vault.unlock("WRONG-password")

        assert result is False
        assert not vault.is_unlocked()
        assert _vault_state.dek is None

    def test_unlock_raises_when_not_initialised(self, vault_db):
        """unlock() raises RuntimeError when vault_config has no row."""
        vault = _make_vault(vault_db)
        # Do NOT call initialize() — table is empty

        with pytest.raises(RuntimeError, match="vault_config is empty"):
            vault.unlock("any-password")

    # ------------------------------------------------------------------
    # is_unlocked / lock
    # ------------------------------------------------------------------

    def test_is_unlocked_false_before_unlock(self, vault_db):
        """is_unlocked() returns False before any unlock call."""
        vault = _make_vault(vault_db)
        vault.initialize("pw")
        assert not vault.is_unlocked()

    def test_lock_clears_dek(self, vault_db):
        """lock() seals the vault so subsequent encrypt/decrypt raise VaultLockedError."""
        vault = _make_vault(vault_db)
        vault.initialize("pw")
        vault.unlock("pw")
        assert vault.is_unlocked()

        vault.lock()

        assert not vault.is_unlocked()
        assert _vault_state.dek is None

    # ------------------------------------------------------------------
    # Encrypt / Decrypt round-trips
    # ------------------------------------------------------------------

    def test_encrypt_decrypt_roundtrip_bytes(self, vault_db):
        """encrypt() followed by decrypt() returns the original plaintext bytes."""
        vault = _make_vault(vault_db)
        vault.initialize("pw")
        vault.unlock("pw")

        plaintext = b"super secret data \x00\xff\x42"
        blob = vault.encrypt(plaintext)
        recovered = vault.decrypt(blob)

        assert recovered == plaintext

    def test_encrypt_decrypt_roundtrip_str(self, vault_db):
        """encrypt_str() followed by decrypt_str() returns the original string."""
        vault = _make_vault(vault_db)
        vault.initialize("pw")
        vault.unlock("pw")

        original = "Hello, vault! \U0001f512"
        blob = vault.encrypt_str(original)
        recovered = vault.decrypt_str(blob)

        assert recovered == original

    def test_encrypt_blob_starts_with_12_byte_nonce(self, vault_db):
        """encrypt() output is at least 29 bytes (12 nonce + 1 ct + 16 tag)."""
        vault = _make_vault(vault_db)
        vault.initialize("pw")
        vault.unlock("pw")

        blob = vault.encrypt(b"x")
        # nonce(12) + ciphertext(1) + tag(16) = 29 minimum
        assert len(blob) >= 29, "Encrypted blob should be at least 29 bytes"

    def test_encrypt_produces_unique_nonces(self, vault_db):
        """Two calls to encrypt() with the same plaintext produce different blobs."""
        vault = _make_vault(vault_db)
        vault.initialize("pw")
        vault.unlock("pw")

        plaintext = b"same data"
        blob1 = vault.encrypt(plaintext)
        blob2 = vault.encrypt(plaintext)

        assert blob1 != blob2, "Each encrypt() call should produce a unique nonce"
        nonce1, nonce2 = blob1[:12], blob2[:12]
        assert nonce1 != nonce2, "Nonces should differ between calls"

    def test_decrypt_rejects_tampered_ciphertext(self, vault_db):
        """decrypt() raises ValueError when the ciphertext has been tampered with."""
        vault = _make_vault(vault_db)
        vault.initialize("pw")
        vault.unlock("pw")

        blob = bytearray(vault.encrypt(b"original"))
        # Flip a byte in the ciphertext portion (after the 12-byte nonce)
        blob[15] ^= 0xFF
        tampered = bytes(blob)

        with pytest.raises(ValueError):
            vault.decrypt(tampered)

    # ------------------------------------------------------------------
    # Locked-state guards
    # ------------------------------------------------------------------

    def test_encrypt_raises_vault_locked_error(self, vault_db):
        """encrypt() raises VaultLockedError when the vault is sealed."""
        vault = _make_vault(vault_db)
        with pytest.raises(VaultLockedError):
            vault.encrypt(b"data")

    def test_decrypt_raises_vault_locked_error(self, vault_db):
        """decrypt() raises VaultLockedError when the vault is sealed."""
        vault = _make_vault(vault_db)
        with pytest.raises(VaultLockedError):
            vault.decrypt(b"\x00" * 40)

    def test_encrypt_str_raises_vault_locked_error(self, vault_db):
        """encrypt_str() raises VaultLockedError when the vault is sealed."""
        vault = _make_vault(vault_db)
        with pytest.raises(VaultLockedError):
            vault.encrypt_str("secret")

    def test_decrypt_str_raises_vault_locked_error(self, vault_db):
        """decrypt_str() raises VaultLockedError when the vault is sealed."""
        vault = _make_vault(vault_db)
        with pytest.raises(VaultLockedError):
            vault.decrypt_str(b"\x00" * 40)

    # ------------------------------------------------------------------
    # change_password
    # ------------------------------------------------------------------

    def test_change_password_rewraps_dek(self, vault_db):
        """change_password() updates vault_config with a new salt and wrapped DEK."""
        vault = _make_vault(vault_db)
        vault.initialize("old-pw")
        vault.unlock("old-pw")

        cursor = vault_db.cursor()
        cursor.execute("SELECT kdf_salt, wrapped_dek FROM vault_config WHERE id = 1")
        old_row = cursor.fetchone()
        old_salt = bytes(old_row[0])
        old_wrapped = bytes(old_row[1])
        cursor.close()

        result = vault.change_password("old-pw", "new-pw")
        assert result is True

        cursor = vault_db.cursor()
        cursor.execute("SELECT kdf_salt, wrapped_dek FROM vault_config WHERE id = 1")
        new_row = cursor.fetchone()
        new_salt = bytes(new_row[0])
        new_wrapped = bytes(new_row[1])
        cursor.close()

        assert new_salt != old_salt, "Salt should be re-randomised on password change"
        assert new_wrapped != old_wrapped, "Wrapped DEK should differ (new nonce/KEK)"

    def test_change_password_wrong_old_returns_false(self, vault_db):
        """change_password() returns False when old_password is incorrect."""
        vault = _make_vault(vault_db)
        vault.initialize("correct-old")

        result = vault.change_password("WRONG-old", "new-pw")

        assert result is False

    def test_change_password_does_not_reencrypt_data(self, vault_db):
        """change_password() does not alter any consumer table rows."""
        vault = _make_vault(vault_db)
        vault.initialize("pw")
        vault.unlock("pw")

        # Encrypt a value and store it in the providers table
        ciphertext = vault.encrypt_str("api-key-value")

        vault_db.execute(
            "INSERT INTO providers (name, platform, model, api_key) VALUES (?,?,?,?)",
            ("test-provider", "openai", "gpt-4", ciphertext),
        )
        vault_db.commit()

        # Change password — consumer data must NOT be touched
        vault.change_password("pw", "new-pw")

        cursor = vault_db.cursor()
        cursor.execute("SELECT api_key FROM providers WHERE name = 'test-provider'")
        stored = bytes(cursor.fetchone()[0])
        cursor.close()

        assert stored == ciphertext, "change_password() must not touch consumer table data"

    def test_change_password_dek_unchanged_after_relock(self, vault_db):
        """After change_password and re-unlock with new password, old data decrypts."""
        vault = _make_vault(vault_db)
        vault.initialize("pw")
        vault.unlock("pw")

        original = "important secret"
        blob = vault.encrypt_str(original)

        vault.change_password("pw", "new-pw")
        vault.lock()
        vault.unlock("new-pw")

        recovered = vault.decrypt_str(blob)
        assert recovered == original, "DEK must remain the same after password change"

    # ------------------------------------------------------------------
    # Fernet fallback (migration window — AC16)
    # ------------------------------------------------------------------

    def test_fernet_fallback_decryption(self, vault_db):
        """decrypt() transparently decrypts legacy Fernet ciphertexts when
        the encryption_keys row is present (migration window is open)."""
        # Seed the legacy encryption_keys table
        legacy_key = "legacy-secret-key-value"
        vault_db.execute(
            "INSERT OR IGNORE INTO encryption_keys (id, key_value) VALUES (1, ?)",
            (legacy_key,),
        )
        vault_db.commit()

        # Create a Fernet ciphertext the same way the old code does
        fernet_token = _make_legacy_fernet_token(legacy_key, "my-api-key")

        vault = _make_vault(vault_db)
        vault.initialize("pw")
        vault.unlock("pw")

        # Pass the Fernet token bytes directly — decrypt() should fall back
        recovered = vault.decrypt(fernet_token)

        assert recovered == b"my-api-key"

    def test_no_fernet_fallback_when_encryption_keys_absent(self, vault_db):
        """decrypt() raises ValueError for truly invalid ciphertext (no legacy key)."""
        vault = _make_vault(vault_db)
        vault.initialize("pw")
        vault.unlock("pw")

        junk = b"\xde\xad\xbe\xef" * 20  # 80 bytes, not valid AES-GCM or Fernet

        with pytest.raises(ValueError):
            vault.decrypt(junk)

    # ------------------------------------------------------------------
    # Migration: encryption_keys row deletion disables fallback (AC15)
    # ------------------------------------------------------------------

    def test_migration_deletes_encryption_keys_row(self, vault_db):
        """After deleting the encryption_keys row the Fernet fallback is disabled."""
        legacy_key = "migration-key"
        vault_db.execute(
            "INSERT OR IGNORE INTO encryption_keys (id, key_value) VALUES (1, ?)",
            (legacy_key,),
        )
        vault_db.commit()

        vault = _make_vault(vault_db)
        vault.initialize("pw")
        vault.unlock("pw")

        # Verify fallback works while the key exists
        token = _make_legacy_fernet_token(legacy_key, "secret")
        assert vault.decrypt(token) == b"secret"

        # Simulate completed migration by removing the legacy key
        vault_db.execute("DELETE FROM encryption_keys WHERE id = 1")
        vault_db.commit()

        # Fallback should no longer work
        with pytest.raises(ValueError):
            vault.decrypt(token)

    # ------------------------------------------------------------------
    # get_vault_service factory
    # ------------------------------------------------------------------

    def test_get_vault_service_returns_vault_service_instance(self, vault_db):
        """get_vault_service() returns a VaultService instance."""
        vault = get_vault_service()
        assert isinstance(vault, VaultService)

    def test_get_vault_service_shares_vault_state(self, vault_db):
        """Two instances from get_vault_service() share the same _vault_state."""
        vault_a = get_vault_service()
        vault_b = get_vault_service()

        vault_a.initialize("shared-pw")
        vault_a.unlock("shared-pw")

        # vault_b uses the same module-level singleton, so it should be unlocked too
        assert vault_b.is_unlocked()

    # ------------------------------------------------------------------
    # VaultLockedError is a proper exception
    # ------------------------------------------------------------------

    def test_vault_locked_error_is_exception(self, vault_db):
        """VaultLockedError is a proper Exception subclass."""
        assert issubclass(VaultLockedError, Exception)
        with pytest.raises(VaultLockedError):
            raise VaultLockedError("test message")
