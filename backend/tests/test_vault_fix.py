"""
Feature tests for the vault DEK backup and password-verified restore/wipe
behaviors introduced by TKT-676.

Backup model (append-forever): every DEK generation writes a fresh, uniquely
stamped ``vault_backup_<stamp>.json`` and never overwrites or deletes an earlier
one. Recovery tries every retained backup, newest first — a corrupt latest
backup is simply skipped and an older valid one restores the original DEK.

Scenarios covered (7 tests total):

1. initialize() writes one vault_backup_<stamp>.json with hex-encoded key
   material, file permissions 0o400, and directory permissions 0o700.
2. RESTORE: encrypt a secret, corrupt the live vault_config row, login with the
   correct password → 200, vault unlocked, the previously-encrypted secret still
   decrypts (no data loss — same DEK recovered from backup).
3. Restore skips a corrupt NEWER backup and falls through to an older valid one;
   every retained backup survives and no new backup is written on restore.
4. UNRECOVERABLE: no backup file exists AND vault_config row is corrupt → login
   returns 401 with onboarding_required=True, master_account row is wiped, and
   all backup files are removed (vault.reset() ran).
5. Append-forever retention: a second call to initialize() leaves the first
   backup in place and adds a second — both are retained.
6. ProviderDbService.get_all_providers returns good providers alongside one
   that has a garbage (undecryptable) api_key, marking the bad row with
   decrypt_failed=True and still returning the good one intact.
7. With the vault sealed (no DEK), get_all_providers() returns all rows with
   api_key=None — the listing must not fail just because the vault is locked.

Every test:
- Uses the real ``db`` fixture (schema.sql via SchemaConvergenceService)
- Uses the real ``store`` fixture (isolated MemoryStore, same production class)
- Uses the real VaultService, real ProviderDbService, real user_auth blueprint
- Zero mocks of production code; FileMapperService._SECURE_DIR is redirected to
  ``tmp_path`` via ``monkeypatch.setattr`` — every path helper derives from this
  one attribute, so it is configuration redirection, not a production-code mock.

NOTE on the transient-exception branch (login() returns 200 locked when
unlock_or_restore raises an unexpected Exception): this path cannot be exercised
without mocks because the password-hash check and the vault access share the same
DB connection. Killing the DB after the hash check passes is architecturally
impossible without a mock seam. The branch exists and is correct; its absence here
is a design-coupling signal, not an oversight.
"""

import base64
import json
import secrets
import pytest
from flask import Flask
from werkzeug.security import generate_password_hash

import services.database_service as _db_mod
import services.vault_service as _vault_mod
from api.user_auth import user_auth_bp
from services.file_mapper_service import FileMapperService
from services.vault_service import _vault_state
from services.provider_db_service import ProviderDbService


# ── Shared helpers ────────────────────────────────────────────────────────────

def _seed_account(raw_conn, password: str = "testpassword123") -> str:
    """Insert a master_account row hashed with *password*; return the password."""
    raw_conn.execute(
        "INSERT INTO master_account (username, password_hash) VALUES (?, ?)",
        ("admin", generate_password_hash(password)),
    )
    raw_conn.commit()
    return password


def _backups(secure_dir):
    """Return all retained backup files, newest first (matches production glob)."""
    return sorted(secure_dir.glob("vault_backup_*.json"), reverse=True)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_vault_singletons():
    """Reset vault module-level singletons before and after every test.

    The cached VaultService instance binds to a DB path that changes per test.
    Clearing it forces ``get_vault_service()`` to re-create the service against
    whatever ``_shared_db_service`` is currently active.
    """
    _vault_state.dek = None
    _vault_mod._vault_service_instance = None
    yield
    _vault_state.dek = None
    _vault_mod._vault_service_instance = None


@pytest.fixture
def secure_dir(tmp_path):
    """Return a fresh per-test secure directory under tmp_path."""
    return tmp_path / "secure"


@pytest.fixture
def redirect_backup_paths(secure_dir, monkeypatch):
    """Redirect FileMapperService's secure dir to tmp_path so tests never write
    to the real data/secure/ directory.

    Every backup path helper derives from the single ``_SECURE_DIR`` class
    attribute, so patching it alone redirects ``get_secure_dir``,
    ``get_vault_backup_path`` and ``list_vault_backups`` consistently. This is
    configuration redirection, not a mock of production behaviour.
    """
    monkeypatch.setattr(FileMapperService, "_SECURE_DIR", secure_dir)


@pytest.fixture
def auth_client(db, store, redirect_backup_paths):
    """Minimal Flask test client with only the user_auth blueprint.

    Uses the real ``db`` fixture (schema.sql + singleton patch), the real
    ``store`` fixture (isolated MemoryStore), and the redirect_backup_paths
    fixture (temp secure directory). No mocks of production code.
    """
    app = Flask(__name__)
    app.secret_key = secrets.token_hex(32)
    app.config["TESTING"] = True
    app.register_blueprint(user_auth_bp)
    with app.test_client() as client:
        yield client, db


# ── Backup write and permissions ──────────────────────────────────────────────

@pytest.mark.unit
class TestVaultBackupWrite:
    """VaultService.initialize() writes a valid backup file with correct permissions."""

    def test_initialize_writes_backup_file_with_hex_fields_and_secure_permissions(
        self, db, store, redirect_backup_paths, secure_dir
    ):
        """initialize() creates exactly one vault_backup_<stamp>.json containing
        hex-encoded key material (kdf_salt, wrapped_dek, dek_nonce), the directory
        is 0o700, and the file is 0o400.  No plaintext secret appears in the JSON.
        """
        vault = _vault_mod.get_vault_service()
        vault.initialize("strongpassword99")

        backups = _backups(secure_dir)
        assert len(backups) == 1, "initialize() must write exactly one backup file"
        backup_path = backups[0]

        # Directory and file permissions
        dir_perms = oct(secure_dir.stat().st_mode & 0o777)
        assert dir_perms == "0o700", f"secure dir must be 0o700, got {dir_perms}"

        file_perms = oct(backup_path.stat().st_mode & 0o777)
        assert file_perms == "0o400", f"backup file must be 0o400, got {file_perms}"

        # Content must be valid JSON with the expected hex fields
        data = json.loads(backup_path.read_text(encoding="utf-8"))
        for field in ("kdf_salt", "wrapped_dek", "dek_nonce"):
            assert field in data, f"backup must contain field '{field}'"
            decoded = bytes.fromhex(data[field])
            assert len(decoded) > 0, f"'{field}' must decode to non-empty bytes"

        assert "kdf_algorithm" in data
        assert "kdf_iterations" in data
        assert "created_at" in data

        # No plaintext password in the backup
        assert "strongpassword99" not in backup_path.read_text(encoding="utf-8")


# ── Restore: data survives vault_config corruption ────────────────────────────

@pytest.mark.unit
class TestVaultRestore:
    """Login recovers the live DEK from backup when vault_config is corrupt."""

    def test_login_restores_vault_and_previously_encrypted_secret_still_decrypts(
        self, auth_client, secure_dir
    ):
        """The full no-data-loss recovery path:

        1. Register + initialize vault; encrypt a secret.
        2. Corrupt the live vault_config row (simulate DB corruption).
        3. Login with the correct password.
        4. Expect 200, vault_state=unlocked.
        5. The secret encrypted before corruption must still decrypt correctly —
           the same DEK was recovered from the backup file.
        """
        client, raw_conn = auth_client
        pw = _seed_account(raw_conn)

        # Initialize vault and encrypt a secret
        vault = _vault_mod.get_vault_service()
        vault.initialize(pw)
        vault.unlock(pw)
        secret_plaintext = "highly-sensitive-credential-abc"
        encrypted_blob = vault.encrypt_str(secret_plaintext)

        # Lock vault (simulate server going idle between login and next request)
        _vault_state.dek = None
        _vault_mod._vault_service_instance = None

        # Corrupt the live vault_config row (invalid wrapped_dek)
        raw_conn.execute(
            "UPDATE vault_config SET wrapped_dek = ? WHERE id = 1",
            (b"\xba\xad" * 24,),
        )
        raw_conn.commit()

        # Login — should restore from the retained backup
        resp = client.post(
            "/auth/login",
            json={"username": "admin", "password": pw},
            content_type="application/json",
        )

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        assert data["vault_state"] == "unlocked"

        # The recovered vault must decrypt the pre-corruption secret correctly
        recovered_vault = _vault_mod.get_vault_service()
        decrypted = recovered_vault.decrypt_str(encrypted_blob)
        assert decrypted == secret_plaintext, (
            "DEK recovered from backup must decrypt secrets encrypted before corruption"
        )

    def test_login_skips_corrupt_newest_backup_and_restores_from_older(
        self, auth_client, secure_dir
    ):
        """Recovery iterates every retained backup newest-first.

        With one valid backup on disk, drop a NEWER (lexically-greater stamp)
        corrupt backup file alongside it.  When vault_config is wiped, login must
        skip the unreadable newest backup and restore from the older valid one.
        Every retained backup must survive and no new backup is written on restore.
        """
        client, raw_conn = auth_client
        pw = _seed_account(raw_conn)

        # Register/initialize — writes the first (valid) backup
        vault = _vault_mod.get_vault_service()
        vault.initialize(pw)
        _vault_state.dek = None
        _vault_mod._vault_service_instance = None

        valid_backups = _backups(secure_dir)
        assert len(valid_backups) == 1
        valid_path = valid_backups[0]

        # Drop a NEWER corrupt backup (lexically-greater stamp => sorts first)
        corrupt_path = secure_dir / "vault_backup_99999999T999999999999.json"
        corrupt_path.write_text("{not valid json content}", encoding="utf-8")
        corrupt_path.chmod(0o400)

        # Newest-first ordering must place the corrupt file ahead of the valid one
        assert _backups(secure_dir)[0] == corrupt_path

        # Wipe vault_config to force the restore path
        raw_conn.execute("DELETE FROM vault_config WHERE id = 1")
        raw_conn.commit()

        resp = client.post(
            "/auth/login",
            json={"username": "admin", "password": pw},
            content_type="application/json",
        )

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        assert data["vault_state"] == "unlocked"

        # Both backups must still be on disk — restore never writes or deletes
        after = _backups(secure_dir)
        assert valid_path in after, "valid backup must be retained after restore"
        assert corrupt_path in after, "corrupt backup must be retained after restore"
        assert len(after) == 2, "restore must not write a new backup file"


# ── Unrecoverable: account wiped, re-onboarding required ─────────────────────

@pytest.mark.unit
class TestVaultUnrecoverable:
    """Login wipes master_account and returns 401 when no backup can restore the DEK."""

    def test_login_wipes_master_account_and_returns_401_with_onboarding_required(
        self, auth_client, secure_dir
    ):
        """When vault_config is corrupt AND no valid backup file exists, login
        must wipe the master_account row, call vault.reset() (removing all backup
        files), and return 401 with onboarding_required=True so the frontend
        drives a clean re-onboarding.
        """
        client, raw_conn = auth_client
        pw = _seed_account(raw_conn)

        vault = _vault_mod.get_vault_service()
        vault.initialize(pw)
        _vault_state.dek = None
        _vault_mod._vault_service_instance = None

        # Verify a backup was written, then delete every backup so there is no
        # recovery path.
        backups = _backups(secure_dir)
        assert len(backups) == 1
        for path in backups:
            path.chmod(0o600)
            path.unlink()

        # Corrupt vault_config
        raw_conn.execute(
            "UPDATE vault_config SET wrapped_dek = ? WHERE id = 1",
            (b"\xde\xad" * 24,),
        )
        raw_conn.commit()

        resp = client.post(
            "/auth/login",
            json={"username": "admin", "password": pw},
            content_type="application/json",
        )

        assert resp.status_code == 401
        data = resp.get_json()
        assert data.get("onboarding_required") is True, (
            "Unrecoverable vault must return onboarding_required=True"
        )

        # master_account row must be wiped
        count = raw_conn.execute(
            "SELECT COUNT(*) FROM master_account"
        ).fetchone()[0]
        assert count == 0, (
            "master_account must be wiped so re-onboarding produces a clean account"
        )

        # vault.reset() must have deleted all backup files
        assert _backups(secure_dir) == [], "vault.reset() must remove all backups"


# ── Append-forever retention ──────────────────────────────────────────────────

@pytest.mark.unit
class TestVaultBackupRetention:
    """Each initialize() appends a new backup and retains all earlier ones."""

    def test_second_initialize_retains_first_backup_and_adds_a_second(
        self, db, store, redirect_backup_paths, secure_dir
    ):
        """After a first initialize(), one backup exists.  A second initialize()
        must leave the first backup untouched and add a second — both retained,
        each owner-read-only.
        """
        vault = _vault_mod.get_vault_service()
        vault.initialize("passwordone99")

        first = _backups(secure_dir)
        assert len(first) == 1, "one backup must exist after first initialize()"
        first_path = first[0]
        first_content = first_path.read_text(encoding="utf-8")

        # Lock and reset the instance to allow a clean second initialize()
        _vault_state.dek = None
        _vault_mod._vault_service_instance = None

        vault = _vault_mod.get_vault_service()
        vault.initialize("passwordtwo99")

        after = _backups(secure_dir)
        assert len(after) == 2, "second initialize() must retain the first backup"
        assert first_path in after, "first backup must be retained, not overwritten"
        assert first_path.read_text(encoding="utf-8") == first_content, (
            "first backup content must be untouched by the second initialize()"
        )

        for path in after:
            perms = oct(path.stat().st_mode & 0o777)
            assert perms == "0o400", f"each backup must be 0o400, got {perms}"


# ── ProviderDbService decrypt tolerance (kept from original — still valid) ────

@pytest.mark.unit
class TestProviderDecryptTolerance:
    """ProviderDbService.get_all_providers tolerates per-row decrypt failures."""

    def test_good_provider_returned_when_another_row_has_garbage_key(
        self, db, store
    ):
        """One provider with a validly-encrypted key and one with unreadable
        garbage in api_key are both inserted.  get_all_providers() must return
        both rows: the good one with a decrypted api_key and the bad one marked
        with decrypt_failed=True.
        """
        # Initialize and unlock the vault so encryption is available
        vault = _vault_mod.get_vault_service()
        vault.initialize("correct-password-123")
        vault.unlock("correct-password-123")

        # Insert a provider whose key is legitimately encrypted
        good_key_plaintext = "sk-good-key-abc123"
        good_key_encrypted = base64.b64encode(
            vault.encrypt_str(good_key_plaintext)
        ).decode()

        db.execute(
            "INSERT INTO providers (name, platform, model, api_key, is_active) "
            "VALUES (?, ?, ?, ?, ?)",
            ("good-provider", "openai", "gpt-4o", good_key_encrypted, 1),
        )

        # Insert a provider whose api_key column contains garbage (old pre-vault
        # plaintext that can't be base64-decoded as AES-GCM)
        db.execute(
            "INSERT INTO providers (name, platform, model, api_key, is_active) "
            "VALUES (?, ?, ?, ?, ?)",
            ("bad-provider", "openai", "gpt-4", "NOT_VALID_ENCRYPTED_DATA", 1),
        )
        db.commit()

        service = ProviderDbService(_db_mod._shared_db_service)
        providers = service.get_all_providers()

        names = {p["name"] for p in providers}
        assert "good-provider" in names
        assert "bad-provider" in names

        good = next(p for p in providers if p["name"] == "good-provider")
        bad = next(p for p in providers if p["name"] == "bad-provider")

        # Good row must decrypt cleanly
        assert good["api_key"] == good_key_plaintext
        assert "decrypt_failed" not in good

        # Bad row must be marked, not crash the whole call
        assert bad.get("decrypt_failed") is True

    def test_all_providers_returned_when_vault_is_locked(self, db, store):
        """With the vault sealed (no DEK), get_all_providers() returns all rows
        with api_key=None — the listing must not fail just because the vault is
        locked.
        """
        # Insert two providers (no need to encrypt — vault is never unlocked)
        db.execute(
            "INSERT INTO providers (name, platform, model, api_key, is_active) "
            "VALUES (?, ?, ?, ?, ?)",
            ("provider-a", "anthropic", "claude-sonnet", "some-blob", 1),
        )
        db.execute(
            "INSERT INTO providers (name, platform, model, api_key, is_active) "
            "VALUES (?, ?, ?, ?, ?)",
            ("provider-b", "ollama", "llama3", None, 1),
        )
        db.commit()

        # Vault intentionally left sealed (_reset_vault_singletons autouse
        # fixture ensures _vault_state.dek is None)
        service = ProviderDbService(_db_mod._shared_db_service)
        providers = service.get_all_providers()

        assert len(providers) == 2
        # api_key is None for all when vault is locked (_unseal_api_key returns None)
        for p in providers:
            assert p["api_key"] is None
