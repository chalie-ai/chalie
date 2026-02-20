"""Settings service — manages application-wide configuration in database."""

import logging
import secrets
from typing import Optional, Any

logger = logging.getLogger(__name__)


class SettingsService:
    """Manages application settings stored in database."""

    def __init__(self, database_service):
        self.db = database_service
        self._enc_key = None

    def _get_enc_key(self):
        """Lazily load encryption key from .key file."""
        if self._enc_key is None:
            from services.encryption_key_service import get_encryption_key
            self._enc_key = get_encryption_key()
        return self._enc_key

    def get(self, key: str) -> Optional[str]:
        """Get a setting value by key."""
        with self.db.get_session() as session:
            from sqlalchemy import text
            result = session.execute(
                text("SELECT "
                     "  CASE WHEN is_sensitive = TRUE AND encrypted_value IS NOT NULL "
                     "       THEN pgp_sym_decrypt(encrypted_value, :enc_key) "
                     "       ELSE value "
                     "  END AS resolved_value "
                     "FROM settings WHERE key = :key"),
                {"key": key, "enc_key": self._get_enc_key()}
            )
            row = result.fetchone()
            return row[0] if row else None

    def set(self, key: str, value: str, value_type: str = 'string', description: str = None) -> str:
        """Create or update a setting."""
        with self.db.get_session() as session:
            from sqlalchemy import text

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
                    # Encrypt sensitive value
                    session.execute(
                        text("UPDATE settings SET encrypted_value = pgp_sym_encrypt(:value, :enc_key), "
                             "value = NULL, updated_at = CURRENT_TIMESTAMP WHERE key = :key"),
                        {"key": key, "value": value, "enc_key": self._get_enc_key()}
                    )
                else:
                    # Plain text value
                    session.execute(
                        text("UPDATE settings SET value = :value, encrypted_value = NULL, "
                             "updated_at = CURRENT_TIMESTAMP WHERE key = :key"),
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
            from sqlalchemy import text
            session.execute(
                text("DELETE FROM settings WHERE key = :key"),
                {"key": key}
            )
            session.commit()
        return True

    def get_all(self) -> dict:
        """Get all settings (mask sensitive values, never decrypt)."""
        with self.db.get_session() as session:
            from sqlalchemy import text
            result = session.execute(
                text("SELECT key, "
                     "  CASE WHEN is_sensitive = TRUE THEN '***' ELSE value END AS value "
                     "FROM settings ORDER BY key")
            )
            rows = result.fetchall()
            return {row[0]: row[1] for row in rows}

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
