"""Setting — one ``settings`` row: an opaque key/value pair, optionally
vault-sealed.

Active-record row-model (Rule 5 / §4.1). ``key`` is the natural identifier
callers reach through (``get``/``set``/``delete`` take a ``key``, not the
``id`` PK), so — like :class:`~models.thread_gist.ThreadGist` — the read/write
verbs are bespoke ``@classmethod``\\ s rather than the base's id-centric
``get``/``save``/``delete``, ported verbatim from the dissolved
``SettingsService`` (owner ruling: settings need no service).

Sensitive settings (``is_sensitive = 1``) are sealed through the vault
(AES-256-GCM) as a base64 blob in ``encrypted_value``; non-sensitive settings
are plain text in ``value``. This is the one owner-sanctioned model that
imports :mod:`services.vault_service` directly (mirrors the ``Database``/
``FileMapperService`` carve-out on :class:`~models.skill.Skill`) — a locked
vault must fail exactly as loudly here as it did through the service:
:meth:`get`/:meth:`set` propagate
:exc:`~services.vault_service.VaultLockedError` unchanged.

``set``'s check-then-write (existing row? sensitive?) is a genuine multi-step
block, so it runs inside ``Database.transaction()`` — the same reentrant
atomic-block primitive the dissolved service used, imported here alongside
the vault as the model's second owner-sanctioned carve-out (Rule-3 depth is
otherwise pure CRUD)."""

from __future__ import annotations

import base64
import logging
import secrets
from typing import ClassVar

from models.model import Model
from services.database import Database
from services.vault_service import get_vault_service

logger = logging.getLogger(__name__)


class Setting(Model):
    """One ``settings`` row, keyed by its unique ``key`` column."""

    __columns__: ClassVar[tuple[str, ...]] = (
        "id", "key", "value", "value_type", "description",
        "is_sensitive", "encrypted_value", "created_at", "updated_at",
    )

    # Well-known non-sensitive setting keys — single source for the cross-layer
    # consumers (REST CORS, the TLS serving path, and the session-cookie Secure flag).
    SSL_ENABLED: ClassVar[str] = "ssl_enabled"
    DEPLOYMENT_DOMAIN: ClassVar[str] = "deployment_domain"
    _BOOL_TRUE: ClassVar[str] = "true"
    _BOOL_FALSE: ClassVar[str] = "false"

    @classmethod
    def get_table(cls) -> str:
        return "settings"

    # Real columns (annotation-only; populated by Model.__init__ from kwargs /
    # hydrate, so mypy knows their types on attribute access).
    key: str
    value: str | None
    value_type: str
    description: str | None
    is_sensitive: int
    encrypted_value: bytes | None
    created_at: str
    updated_at: str

    @classmethod
    def get_bool(cls, key: str) -> bool:
        """Read a boolean setting — True only for the stored literal ``'true'``."""
        return cls.get(key) == cls._BOOL_TRUE

    @classmethod
    def set_bool(cls, key: str, value: bool) -> None:
        """Persist a boolean setting as ``'true'``/``'false'`` (non-sensitive plain text)."""
        cls.set(key, cls._BOOL_TRUE if value else cls._BOOL_FALSE, "boolean")

    @classmethod
    def get(cls, key: str) -> str | None:  # type: ignore[override]
        """Sensitive settings are decrypted via the VaultService (AES-256-GCM).
        The vault must be unlocked before sensitive settings can be read.

        Raises:
            :exc:`~services.vault_service.VaultLockedError`: If the setting is
                sensitive and the vault has not been unlocked yet.
        """
        row = cls.filter("key", key).first()
        if row is None:
            return None

        if row.is_sensitive and row.encrypted_value is not None:
            return get_vault_service().decrypt_str(base64.b64decode(row.encrypted_value))
        return row.value

    @classmethod
    def set(cls, key: str, value: str, value_type: str = 'string', description: str | None = None) -> str:
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

    @classmethod
    def delete(cls, key: str) -> bool:  # type: ignore[override]
        with Database.transaction():
            cls.filter("key", key).delete()
        return True

    @classmethod
    def get_api_key_or_generate(cls) -> str:
        """Generates and stores a new API key in the database if one does not already exist."""
        # Try to get existing key
        existing = cls.get('api_key')
        if existing:
            logger.info("[Setting] Using existing API key from database")
            return existing

        # Generate new key
        new_key = secrets.token_urlsafe(32)
        cls.set('api_key', new_key, 'string', 'REST API authentication key (auto-generated on startup)')
        logger.info("[Setting] Generated and stored new API key in database")
        return new_key
