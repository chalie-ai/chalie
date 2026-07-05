

import logging
import secrets
from typing import Optional, cast

from services.database import Database

logger = logging.getLogger(__name__)


class SettingsService:

    # Well-known non-sensitive setting keys — single source for the cross-layer
    # consumers (REST CORS, the TLS serving path, and the session-cookie Secure flag).
    SSL_ENABLED = "ssl_enabled"
    DEPLOYMENT_DOMAIN = "deployment_domain"
    _BOOL_TRUE = "true"
    _BOOL_FALSE = "false"

    def get_bool(self, key: str) -> bool:
        """Read a boolean setting — True only for the stored literal ``'true'``."""
        return self.get(key) == self._BOOL_TRUE

    def set_bool(self, key: str, value: bool) -> None:
        """Persist a boolean setting as ``'true'``/``'false'`` (non-sensitive plain text)."""
        self.set(key, self._BOOL_TRUE if value else self._BOOL_FALSE, "boolean")

    def get(self, key: str) -> Optional[str]:
        """Sensitive settings are decrypted via the VaultService (AES-256-GCM).
        The vault must be unlocked before sensitive settings can be read.

        Raises:
            :exc:`~services.vault_service.VaultLockedError`: If the setting is
                sensitive and the vault has not been unlocked yet.
        """
        row = Database.conn().execute(
            "SELECT value, encrypted_value, is_sensitive "
            "FROM settings WHERE key = :key",
            {"key": key}
        ).fetchone()
        if not row:
            return None

        is_sensitive = row[2]
        if is_sensitive and row[1] is not None:
            import base64
            from services.vault_service import get_vault_service
            return get_vault_service().decrypt_str(base64.b64decode(row[1]))
        return cast(str | None, row[0])

    def set(self, key: str, value: str, value_type: str = 'string', description: str | None = None) -> str:
        """Sensitive settings are encrypted via the VaultService (AES-256-GCM)
        and stored as base64-encoded blobs in ``encrypted_value``.  Non-sensitive
        settings are stored as plain text in ``value``.

        The vault must be unlocked before a sensitive setting can be written.

        Raises:
            :exc:`~services.vault_service.VaultLockedError`: If the setting is
                sensitive and the vault has not been unlocked yet.
        """
        with Database.transaction() as conn:
            # Check if exists and get its sensitivity flag
            existing = conn.execute(
                "SELECT id, is_sensitive FROM settings WHERE key = :key",
                {"key": key}
            ).fetchone()
            row_is_sensitive = existing[1] if existing else False

            if existing:
                # Update
                if row_is_sensitive:
                    # Encrypt sensitive value via VaultService
                    import base64
                    from services.vault_service import get_vault_service
                    encrypted = base64.b64encode(
                        get_vault_service().encrypt_str(value)
                    ).decode()
                    conn.execute(
                        "UPDATE settings SET encrypted_value = :enc_value, "
                        "value = NULL, updated_at = datetime('now') WHERE key = :key",
                        {"key": key, "enc_value": encrypted}
                    )
                else:
                    # Plain text value
                    conn.execute(
                        "UPDATE settings SET value = :value, encrypted_value = NULL, "
                        "updated_at = datetime('now') WHERE key = :key",
                        {"key": key, "value": value}
                    )
            else:
                # Insert (non-sensitive only; sensitive rows must be seeded by migration)
                conn.execute(
                    "INSERT INTO settings (key, value, value_type, description) VALUES (:key, :value, :value_type, :description)",
                    {"key": key, "value": value, "value_type": value_type, "description": description}
                )
        return value

    def delete(self, key: str) -> bool:
        with Database.transaction() as conn:
            conn.execute(
                "DELETE FROM settings WHERE key = :key",
                {"key": key}
            )
        return True

    def get_api_key_or_generate(self) -> str:
        """Generates and stores a new API key in the database if one does not already exist."""
        # Try to get existing key
        existing = self.get('api_key')
        if existing:
            logger.info("[SettingsService] Using existing API key from database")
            return existing

        # Generate new key
        new_key = secrets.token_urlsafe(32)
        self.set('api_key', new_key, 'string', 'REST API authentication key (auto-generated on startup)')
        logger.info("[SettingsService] Generated and stored new API key in database")
        return new_key
