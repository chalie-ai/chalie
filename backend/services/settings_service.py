"""Settings service — manages application-wide configuration in database."""

import logging
import secrets
from typing import Optional

from services.database_service import text

logger = logging.getLogger(__name__)


class SettingsService:
    """Manages application settings stored in database."""

    def __init__(self, database_service):
        """Initialise the service with a shared database connection.

        Args:
            database_service: Active database service instance used for all
                settings reads and writes.
        """
        self.db = database_service

    def get(self, key: str) -> Optional[str]:
        """Get a setting value by key.

        Sensitive settings are decrypted via the VaultService (AES-256-GCM).
        The vault must be unlocked before sensitive settings can be read.

        Args:
            key: The settings key to look up.

        Returns:
            The plaintext setting value, or ``None`` if the key does not exist.

        Raises:
            :exc:`~services.vault_service.VaultLockedError`: If the setting is
                sensitive and the vault has not been unlocked yet.
        """
        with self.db.get_session() as session:
            result = session.execute(
                text("SELECT value, encrypted_value, is_sensitive "
                     "FROM settings WHERE key = :key"),
                {"key": key}
            )
            row = result.fetchone()
            if not row:
                return None

            is_sensitive = row[2]
            if is_sensitive and row[1] is not None:
                import base64
                from services.vault_service import get_vault_service
                return get_vault_service().decrypt_str(base64.b64decode(row[1]))
            return row[0]

    def set(self, key: str, value: str, value_type: str = 'string', description: str = None) -> str:
        """Create or update a setting.

        Sensitive settings are encrypted via the VaultService (AES-256-GCM)
        and stored as base64-encoded blobs in ``encrypted_value``.  Non-sensitive
        settings are stored as plain text in ``value``.

        The vault must be unlocked before a sensitive setting can be written.

        Args:
            key:         The settings key to create or update.
            value:       The plaintext value to store.
            value_type:  SQLite type hint stored alongside the row (default
                         ``'string'``).  Used only on INSERT.
            description: Optional human-readable description stored on INSERT.

        Returns:
            The original *value* string (unchanged).

        Raises:
            :exc:`~services.vault_service.VaultLockedError`: If the setting is
                sensitive and the vault has not been unlocked yet.
        """
        with self.db.get_session() as session:
            # Check if exists and get its sensitivity flag
            result = session.execute(
                text("SELECT id, is_sensitive FROM settings WHERE key = :key"),
                {"key": key}
            )
            existing = result.fetchone()
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
                    session.execute(
                        text("UPDATE settings SET encrypted_value = :enc_value, "
                             "value = NULL, updated_at = datetime('now') WHERE key = :key"),
                        {"key": key, "enc_value": encrypted}
                    )
                else:
                    # Plain text value
                    session.execute(
                        text("UPDATE settings SET value = :value, encrypted_value = NULL, "
                             "updated_at = datetime('now') WHERE key = :key"),
                        {"key": key, "value": value}
                    )
            else:
                # Insert (non-sensitive only; sensitive rows must be seeded by migration)
                session.execute(
                    text("INSERT INTO settings (key, value, value_type, description) VALUES (:key, :value, :value_type, :description)"),
                    {"key": key, "value": value, "value_type": value_type, "description": description}
                )

            session.commit()
        return value

    def delete(self, key: str) -> bool:
        """Delete a setting."""
        with self.db.get_session() as session:
            session.execute(
                text("DELETE FROM settings WHERE key = :key"),
                {"key": key}
            )
            session.commit()
        return True

    def get_api_key_or_generate(self) -> str:
        """
        Get API key from settings, or generate and store a new one if not present.

        Returns:
            API key string
        """
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
