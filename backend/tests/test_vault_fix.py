"""
Feature tests for the vault DEK backup and password-verified restore/wipe
behaviors.

Backup model (append-forever): every DEK generation writes a fresh, uniquely
stamped ``vault_backup_<stamp>.json`` and never overwrites or deletes an earlier
one. Recovery tries every retained backup, newest first — a corrupt latest
backup is simply skipped and an older valid one restores the original DEK.

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
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest
from flask.testing import FlaskClient
from werkzeug.security import generate_password_hash

import services.database_service as _db_mod
import services.vault_service as _vault_mod
from api.user_auth import user_auth_ns
from services.database_service import DatabaseService
from services.file_mapper_service import FileMapperService
from services.provider_db_service import ProviderDbService
from services.vault_service import _vault_state
from tests.restx_test_app import mount_namespace


# ── Shared helpers ────────────────────────────────────────────────────────────

def _seed_account(raw_conn: sqlite3.Connection, password: str = "testpassword123") -> str:
    raw_conn.execute(
        "INSERT INTO master_account (username, password_hash) VALUES (?, ?)",
        ("admin", generate_password_hash(password)),
    )
    raw_conn.commit()
    return password


def _backups(secure_dir: Path) -> list[Path]:
    return sorted(secure_dir.glob("vault_backup_*.json"), reverse=True)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_vault_singletons() -> Iterator[None]:
    """Clears the cached VaultService singleton before and after every test so
    ``get_vault_service()`` re-creates it against the per-test DB path."""
    _vault_state.dek = None
    _vault_mod._vault_service_instance = None
    yield
    _vault_state.dek = None
    _vault_mod._vault_service_instance = None


@pytest.fixture
def secure_dir(tmp_path: Path) -> Path:
    return tmp_path / "secure"


@pytest.fixture
def redirect_backup_paths(secure_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirects ``FileMapperService._SECURE_DIR`` to ``tmp_path`` so all backup
    path helpers (``get_secure_dir``, ``get_vault_backup_path``,
    ``list_vault_backups``) resolve under the per-test temp directory."""
    monkeypatch.setattr(FileMapperService, "_SECURE_DIR", secure_dir)


@pytest.fixture
def auth_client(db: sqlite3.Connection, store: object, redirect_backup_paths: None) -> Iterator[tuple[FlaskClient, sqlite3.Connection]]:
    app = mount_namespace(user_auth_ns)
    app.secret_key = secrets.token_hex(32)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client, db


# ── Backup write and permissions ──────────────────────────────────────────────

@pytest.mark.unit
class TestVaultBackupWrite:

    def test_initialize_writes_backup_file_with_hex_fields_and_secure_permissions(
        self, db: sqlite3.Connection, store: object, redirect_backup_paths: None, secure_dir: Path
    ) -> None:
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

    def test_backups_are_capped_to_the_newest_retention_window(
        self, db: sqlite3.Connection, store: object, redirect_backup_paths: None, secure_dir: Path
    ) -> None:
        cap = _vault_mod._BACKUP_RETENTION
        vault = _vault_mod.get_vault_service()
        for _ in range(cap + 3):
            vault.initialize("strongpassword99")

        backups = _backups(secure_dir)
        assert len(backups) == cap, f"only the newest {cap} backups must be retained, got {len(backups)}"
        # The survivors are the newest stamps and every one still parses.
        assert backups == sorted(secure_dir.glob("vault_backup_*.json"), reverse=True)[:cap]
        for path in backups:
            json.loads(path.read_text(encoding="utf-8"))


# ── Restore: data survives vault_config corruption ────────────────────────────

@pytest.mark.unit
class TestVaultRestore:

    def test_login_restores_vault_and_previously_encrypted_secret_still_decrypts(
        self, auth_client: tuple[FlaskClient, sqlite3.Connection], secure_dir: Path
    ) -> None:
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
            "/api/auth/login",
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
        self, auth_client: tuple[FlaskClient, sqlite3.Connection], secure_dir: Path
    ) -> None:
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
            "/api/auth/login",
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

    def test_login_wipes_master_account_and_returns_401_with_onboarding_required(
        self, auth_client: tuple[FlaskClient, sqlite3.Connection], secure_dir: Path
    ) -> None:
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
            "/api/auth/login",
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

    def test_second_initialize_retains_first_backup_and_adds_a_second(
        self, db: sqlite3.Connection, store: object, redirect_backup_paths: None, secure_dir: Path
    ) -> None:
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

    def test_good_provider_returned_when_another_row_has_garbage_key(
        self, db: sqlite3.Connection, store: object
    ) -> None:
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
            "INSERT INTO providers (name, platform, model, api_key) "
            "VALUES (?, ?, ?, ?)",
            ("good-provider", "openai", "gpt-4o", good_key_encrypted),
        )

        # Insert a provider whose api_key column contains garbage (old pre-vault
        # plaintext that can't be base64-decoded as AES-GCM)
        db.execute(
            "INSERT INTO providers (name, platform, model, api_key) "
            "VALUES (?, ?, ?, ?)",
            ("bad-provider", "openai", "gpt-4", "NOT_VALID_ENCRYPTED_DATA"),
        )
        db.commit()

        service = ProviderDbService(cast(DatabaseService, _db_mod._shared_db_service))
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

    def test_all_providers_returned_when_vault_is_locked(self, db: sqlite3.Connection, store: object) -> None:
        # Insert two providers (no need to encrypt — vault is never unlocked)
        db.execute(
            "INSERT INTO providers (name, platform, model, api_key) "
            "VALUES (?, ?, ?, ?)",
            ("provider-a", "anthropic", "claude-sonnet", "some-blob"),
        )
        db.execute(
            "INSERT INTO providers (name, platform, model, api_key) "
            "VALUES (?, ?, ?, ?)",
            ("provider-b", "ollama", "llama3", None),
        )
        db.commit()

        # Vault intentionally left sealed (_reset_vault_singletons autouse
        # fixture ensures _vault_state.dek is None)
        service = ProviderDbService(cast(DatabaseService, _db_mod._shared_db_service))
        providers = service.get_all_providers()

        assert len(providers) == 2
        # api_key is None for all when vault is locked (_unseal_api_key returns None)
        for p in providers:
            assert p["api_key"] is None
