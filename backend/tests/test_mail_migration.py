"""Unit tests for the legacy CalDAV / IMAP → mail credential migration."""

from __future__ import annotations

import base64
import json
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vault(encrypt_fn=None, decrypt_fn=None):
    """Return a MagicMock vault with controllable encrypt/decrypt behaviour.

    By default the mock uses a simple reversible XOR-alike that just
    wraps the input in a predictable bytes envelope — good enough for
    round-trip tests without a real KMS.
    """
    vault = MagicMock()

    if encrypt_fn is None:
        # encrypt_str(plaintext) → bytes: prefix plaintext bytes with b"ENC:"
        vault.encrypt_str.side_effect = lambda pt: b"ENC:" + pt.encode()
    else:
        vault.encrypt_str.side_effect = encrypt_fn

    if decrypt_fn is None:
        # decrypt_str(raw: bytes) → str: strip the b"ENC:" prefix
        vault.decrypt_str.side_effect = lambda raw: raw[4:].decode()
    else:
        vault.decrypt_str.side_effect = decrypt_fn

    return vault


def _encode(plaintext: str) -> str:
    """Simulate what store_credential writes: base64(vault.encrypt_str(pt))."""
    encrypted = b"ENC:" + plaintext.encode()
    return base64.b64encode(encrypted).decode()


def _seed_tool_config(conn, tool_name: str, key: str, value: str) -> None:
    """Insert a raw (already-encrypted) tool_config row."""
    conn.execute(
        """
        INSERT INTO tool_configs (tool_name, config_key, config_value)
        VALUES (?, ?, ?)
        ON CONFLICT (tool_name, config_key) DO UPDATE
            SET config_value = EXCLUDED.config_value
        """,
        (tool_name, key, value),
    )
    conn.commit()


def _get_config(conn, tool_name: str) -> dict[str, str]:
    """Fetch all tool_config rows for a tool_name into a dict."""
    rows = conn.execute(
        "SELECT config_key, config_value FROM tool_configs WHERE tool_name = ?",
        (tool_name,),
    ).fetchall()
    return {r[0]: r[1] for r in rows}


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMigrateNoLegacyCredentials:
    """Fresh install with no legacy credentials should skip migration."""

    def test_returns_false_when_no_legacy_rows(self, db):
        vault = _make_vault()
        with patch(
            "services.vault_service.get_vault_service",
            return_value=vault,
        ):
            from capabilities.mail_capability.capability import migrate_legacy_credentials

            result = migrate_legacy_credentials()

        assert result is False
        # Nothing written under mail tool_name
        assert _get_config(db, "mail") == {}


@pytest.mark.unit
class TestMigrateImapOnly:
    """IMAP-only credentials (no CalDAV rows) should migrate to mail:*."""

    def test_imap_email_and_password_migrated(self, db):
        _seed_tool_config(db, "imap", "imap:email", _encode("user@gmail.com"))
        _seed_tool_config(db, "imap", "imap:password", _encode("app-pw-imap"))

        vault = _make_vault()
        with patch(
            "services.vault_service.get_vault_service",
            return_value=vault,
        ):
            from importlib import reload
            import capabilities.mail_capability.capability as mod
            reload(mod)
            result = mod.migrate_legacy_credentials()

        assert result is True

        mail_cfg = _get_config(db, "mail")
        # mail:email must be present and decrypt to the original address
        assert "mail:email" in mail_cfg
        decrypted_email = base64.b64decode(mail_cfg["mail:email"]).decode()[4:]
        assert decrypted_email == "user@gmail.com"

        # mail:password must be present
        assert "mail:password" in mail_cfg

        # Provider settings for Gmail should be written
        assert "mail:imap_host" in mail_cfg
        assert "mail:smtp_host" in mail_cfg
        assert "mail:caldav_url" in mail_cfg  # Google supports CalDAV
        assert "mail:carddav_url" in mail_cfg  # Google supports CardDAV

        # Protocols list must include at least imap
        assert "mail:protocols" in mail_cfg
        protocols_raw = base64.b64decode(mail_cfg["mail:protocols"]).decode()[4:]
        protocols = json.loads(protocols_raw)
        assert "imap" in protocols

    def test_old_imap_rows_deleted_after_migration(self, db):
        _seed_tool_config(db, "imap", "imap:email", _encode("user@icloud.com"))
        _seed_tool_config(db, "imap", "imap:password", _encode("secret"))

        vault = _make_vault()
        with patch(
            "services.vault_service.get_vault_service",
            return_value=vault,
        ):
            from importlib import reload
            import capabilities.mail_capability.capability as mod
            reload(mod)
            mod.migrate_legacy_credentials()

        assert _get_config(db, "imap") == {}

    def test_imap_watermark_migrated(self, db):
        _seed_tool_config(db, "imap", "imap:email", _encode("user@yahoo.com"))
        _seed_tool_config(db, "imap", "imap:password", _encode("pw"))
        _seed_tool_config(db, "imap", "imap:last_uid", _encode("42"))

        vault = _make_vault()
        with patch(
            "services.vault_service.get_vault_service",
            return_value=vault,
        ):
            from importlib import reload
            import capabilities.mail_capability.capability as mod
            reload(mod)
            mod.migrate_legacy_credentials()

        mail_cfg = _get_config(db, "mail")
        assert "mail:imap_watermark" in mail_cfg
        wm_plain = base64.b64decode(mail_cfg["mail:imap_watermark"]).decode()[4:]
        assert wm_plain == "42"

    def test_provider_name_written(self, db):
        _seed_tool_config(db, "imap", "imap:email", _encode("u@outlook.com"))
        _seed_tool_config(db, "imap", "imap:password", _encode("pw"))

        vault = _make_vault()
        with patch(
            "services.vault_service.get_vault_service",
            return_value=vault,
        ):
            from importlib import reload
            import capabilities.mail_capability.capability as mod
            reload(mod)
            mod.migrate_legacy_credentials()

        mail_cfg = _get_config(db, "mail")
        assert "mail:provider_name" in mail_cfg
        provider_plain = base64.b64decode(mail_cfg["mail:provider_name"]).decode()[4:]
        assert provider_plain == "Outlook"


@pytest.mark.unit
class TestMigrateCalDavOnly:
    """CalDAV-only credentials (no IMAP rows) should fall back to caldav:username."""

    def test_caldav_username_and_password_migrated(self, db):
        _seed_tool_config(db, "caldav", "caldav:username", _encode("user@icloud.com"))
        _seed_tool_config(db, "caldav", "caldav:password", _encode("caldav-pw"))

        vault = _make_vault()
        with patch(
            "services.vault_service.get_vault_service",
            return_value=vault,
        ):
            from importlib import reload
            import capabilities.mail_capability.capability as mod
            reload(mod)
            result = mod.migrate_legacy_credentials()

        assert result is True
        mail_cfg = _get_config(db, "mail")
        assert "mail:email" in mail_cfg

        decrypted_email = base64.b64decode(mail_cfg["mail:email"]).decode()[4:]
        assert decrypted_email == "user@icloud.com"

    def test_old_caldav_rows_deleted(self, db):
        _seed_tool_config(db, "caldav", "caldav:username", _encode("x@icloud.com"))
        _seed_tool_config(db, "caldav", "caldav:password", _encode("pw"))

        vault = _make_vault()
        with patch(
            "services.vault_service.get_vault_service",
            return_value=vault,
        ):
            from importlib import reload
            import capabilities.mail_capability.capability as mod
            reload(mod)
            mod.migrate_legacy_credentials()

        assert _get_config(db, "caldav") == {}


@pytest.mark.unit
class TestMigrateBothImapAndCalDav:
    """When both IMAP and CalDAV rows exist, IMAP email takes priority."""

    def test_imap_email_preferred_over_caldav_username(self, db):
        _seed_tool_config(db, "imap", "imap:email", _encode("user@gmail.com"))
        _seed_tool_config(db, "imap", "imap:password", _encode("imap-pw"))
        _seed_tool_config(db, "caldav", "caldav:username", _encode("other@icloud.com"))
        _seed_tool_config(db, "caldav", "caldav:password", _encode("caldav-pw"))

        vault = _make_vault()
        with patch(
            "services.vault_service.get_vault_service",
            return_value=vault,
        ):
            from importlib import reload
            import capabilities.mail_capability.capability as mod
            reload(mod)
            result = mod.migrate_legacy_credentials()

        assert result is True
        mail_cfg = _get_config(db, "mail")
        email_plain = base64.b64decode(mail_cfg["mail:email"]).decode()[4:]
        assert email_plain == "user@gmail.com"  # IMAP wins

    def test_both_old_tool_names_deleted(self, db):
        _seed_tool_config(db, "imap", "imap:email", _encode("user@gmail.com"))
        _seed_tool_config(db, "imap", "imap:password", _encode("pw"))
        _seed_tool_config(db, "caldav", "caldav:username", _encode("user@gmail.com"))
        _seed_tool_config(db, "caldav", "caldav:password", _encode("pw2"))

        vault = _make_vault()
        with patch(
            "services.vault_service.get_vault_service",
            return_value=vault,
        ):
            from importlib import reload
            import capabilities.mail_capability.capability as mod
            reload(mod)
            mod.migrate_legacy_credentials()

        assert _get_config(db, "imap") == {}
        assert _get_config(db, "caldav") == {}


@pytest.mark.unit
class TestMigrateIdempotent:
    """Second call with mail:email already present must be a no-op."""

    def test_skips_if_mail_email_exists(self, db):
        # Pre-populate mail:email as if a previous migration already ran
        _seed_tool_config(db, "mail", "mail:email", _encode("user@gmail.com"))
        # Also leave stale imap rows present
        _seed_tool_config(db, "imap", "imap:email", _encode("user@gmail.com"))
        _seed_tool_config(db, "imap", "imap:password", _encode("pw"))

        vault = _make_vault()
        with patch(
            "services.vault_service.get_vault_service",
            return_value=vault,
        ):
            from importlib import reload
            import capabilities.mail_capability.capability as mod
            reload(mod)
            result = mod.migrate_legacy_credentials()

        assert result is False
        # Stale imap rows must NOT have been touched
        assert "imap:email" in _get_config(db, "imap")

    def test_decrypt_not_called_when_skipping(self, db):
        _seed_tool_config(db, "mail", "mail:email", _encode("user@gmail.com"))

        vault = _make_vault()
        with patch(
            "services.vault_service.get_vault_service",
            return_value=vault,
        ):
            from importlib import reload
            import capabilities.mail_capability.capability as mod
            reload(mod)
            mod.migrate_legacy_credentials()

        vault.decrypt_str.assert_not_called()


@pytest.mark.unit
class TestScheduledItemsSourceUpdate:
    """Scheduled items with source='caldav' or 'imap' should be repointed to 'mail'."""

    def _insert_item(self, conn, uid: str, source: str) -> None:
        conn.execute(
            """
            INSERT INTO scheduled_items
                (id, message, due_at, status, source, external_uid)
            VALUES (?, ?, datetime('now'), 'pending', ?, ?)
            """,
            (uid, f"item-{uid}", source, uid),
        )
        conn.commit()

    def test_caldav_items_repointed_to_mail(self, db):
        self._insert_item(db, "caldav-001", "caldav")
        _seed_tool_config(db, "caldav", "caldav:username", _encode("u@icloud.com"))
        _seed_tool_config(db, "caldav", "caldav:password", _encode("pw"))

        vault = _make_vault()
        with patch(
            "services.vault_service.get_vault_service",
            return_value=vault,
        ):
            from importlib import reload
            import capabilities.mail_capability.capability as mod
            reload(mod)
            mod.migrate_legacy_credentials()

        rows = db.execute(
            "SELECT source FROM scheduled_items WHERE external_uid = 'caldav-001'"
        ).fetchall()
        assert rows[0][0] == "mail"

    def test_imap_items_repointed_to_mail(self, db):
        self._insert_item(db, "imap-999", "imap")
        _seed_tool_config(db, "imap", "imap:email", _encode("u@gmail.com"))
        _seed_tool_config(db, "imap", "imap:password", _encode("pw"))

        vault = _make_vault()
        with patch(
            "services.vault_service.get_vault_service",
            return_value=vault,
        ):
            from importlib import reload
            import capabilities.mail_capability.capability as mod
            reload(mod)
            mod.migrate_legacy_credentials()

        rows = db.execute(
            "SELECT source FROM scheduled_items WHERE external_uid = 'imap-999'"
        ).fetchall()
        assert rows[0][0] == "mail"

    def test_unrelated_items_untouched(self, db):
        self._insert_item(db, "system-001", "system")
        _seed_tool_config(db, "imap", "imap:email", _encode("u@gmail.com"))
        _seed_tool_config(db, "imap", "imap:password", _encode("pw"))

        vault = _make_vault()
        with patch(
            "services.vault_service.get_vault_service",
            return_value=vault,
        ):
            from importlib import reload
            import capabilities.mail_capability.capability as mod
            reload(mod)
            mod.migrate_legacy_credentials()

        rows = db.execute(
            "SELECT source FROM scheduled_items WHERE external_uid = 'system-001'"
        ).fetchall()
        assert rows[0][0] == "system"


@pytest.mark.unit
class TestMigrateUnknownDomain:
    """An email with an unknown domain should still migrate credentials, just
    without provider-derived server settings."""

    def test_unknown_domain_migrates_email_without_provider_fields(self, db):
        _seed_tool_config(db, "imap", "imap:email", _encode("user@custom.corp"))
        _seed_tool_config(db, "imap", "imap:password", _encode("pw"))

        vault = _make_vault()
        with patch(
            "services.vault_service.get_vault_service",
            return_value=vault,
        ):
            from importlib import reload
            import capabilities.mail_capability.capability as mod
            reload(mod)
            result = mod.migrate_legacy_credentials()

        assert result is True
        mail_cfg = _get_config(db, "mail")
        assert "mail:email" in mail_cfg
        # No provider-specific fields written for unknown domain
        assert "mail:provider_name" not in mail_cfg
        assert "mail:imap_host" not in mail_cfg
