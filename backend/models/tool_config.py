"""ToolConfig — one ``tool_configs`` row: per-tool configuration storage.

Active-record row-model (Rule 5 / §4.1). The table's natural identifier is the
``(tool_name, config_key)`` pair, so the read/write verbs are bespoke
``@classmethod``\\ s rather than the base's id-centric ``get``/``save``/``delete``.
This model is the SOLE home of ``tool_configs`` SQL; all callers reach through
it (the dissolved ``ToolConfigService`` is gone).

Encrypted values (credentials) ride on top of the plain read/write path:
:meth:`get_encrypted`/:meth:`set_encrypted` call :meth:`get`/:meth:`set` and add
only crypto — no SQL of their own. This is the second owner-sanctioned model
that imports :mod:`services.vault_service` directly (the first being
:class:`~models.setting.Setting`); a locked vault must fail exactly as loudly
here as it did through the service: :meth:`get_encrypted`/:meth:`set_encrypted`
propagate :exc:`~services.vault_service.VaultLockedError` unchanged.

``set`` uses an upsert with ``ON CONFLICT`` to handle the unique constraint on
``(tool_name, config_key)``; the generated TEXT id is ignored on conflict, which
is correct (the id is opaque and not part of the natural key)."""

from __future__ import annotations

import base64
import logging
import secrets
from typing import ClassVar

from models.model import Model
from services.database import Database
from services.vault_service import get_vault_service
from exceptions import VaultLockedError

logger = logging.getLogger(__name__)


class ToolConfig(Model):
    """One ``tool_configs`` row, keyed by the ``(tool_name, config_key)`` pair."""

    __columns__: ClassVar[tuple[str, ...]] = (
        "id", "tool_name", "config_key", "config_value", "created_at", "updated_at",
    )

    @classmethod
    def get_table(cls) -> str:
        return "tool_configs"

    # Real columns (annotation-only; populated by Model.__init__ from kwargs /
    # hydrate, so mypy knows their types on attribute access).
    id: str  # type: ignore[override]
    tool_name: str
    config_key: str
    config_value: str
    created_at: str
    updated_at: str

    @classmethod
    def get(cls, tool_name: str, key: str) -> str | None:  # type: ignore[override]
        """Read a tool config value by its ``(tool_name, config_key)`` pair.

        Returns ``None`` if the row is absent.

        This is the SOLE DB read path for ``tool_configs`` — all other methods
        (including :meth:`get_encrypted`) route through this."""
        row = cls.filter("tool_name", tool_name).filter("config_key", key).first()
        return row.config_value if row is not None else None

    @classmethod
    def set(cls, tool_name: str, key: str, value: str) -> None:
        """Upsert a tool config value for the ``(tool_name, config_key)`` pair.

        Generates a TEXT id (opaque, not part of the natural key) for new rows;
        on conflict the generated id is ignored (the existing row is updated).

        This is the SOLE DB write path for ``tool_configs`` — all other methods
        (including :meth:`set_encrypted`) route through this."""
        new_id = secrets.token_hex(8)
        with Database.transaction() as conn:
            conn.execute(
                "INSERT INTO tool_configs (id, tool_name, config_key, config_value) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(tool_name, config_key) DO UPDATE SET "
                "config_value = excluded.config_value, "
                "updated_at = datetime('now')",
                (new_id, tool_name, key, value),
            )

    @classmethod
    def get_encrypted(cls, tool_name: str, key: str) -> str | None:
        """Load and AES-256-GCM–decrypt a stored credential via VaultService.

        Returns ``None`` if the key is absent, the vault is locked, or
        decryption fails (e.g. the vault has been re-initialised and the DEK
        has rotated).

        The stored value is expected to be a base64-encoded blob as written by
        :meth:`set_encrypted`.

        Raises:
            :exc:`~services.vault_service.VaultLockedError`: If the vault has
                not yet been unlocked via
                :meth:`~services.vault_service.VaultService.unlock`.
        """
        encrypted = cls.get(tool_name, key)
        if encrypted is None:
            return None
        try:
            raw = base64.b64decode(encrypted.encode())
            return get_vault_service().decrypt_str(raw)
        except VaultLockedError:
            raise
        except Exception as exc:
            logger.warning(
                "[ToolConfig] get_encrypted('%s', '%s') failed (returning None): %s",
                tool_name, key, exc,
            )
            return None

    @classmethod
    def set_encrypted(cls, tool_name: str, key: str, value: str) -> None:
        """Encrypt *value* with AES-256-GCM via VaultService and persist it.

        The value is encrypted, base64-encoded, then stored via :meth:`set`.

        Raises:
            :exc:`~services.vault_service.VaultLockedError`: If the vault has
                not yet been unlocked via
                :meth:`~services.vault_service.VaultService.unlock`.
        """
        try:
            blob = get_vault_service().encrypt_str(value)
            encrypted = base64.b64encode(blob).decode()
            cls.set(tool_name, key, encrypted)
        except Exception as exc:
            logger.error(
                "[ToolConfig] set_encrypted('%s', '%s') failed: %s",
                tool_name, key, exc,
                exc_info=True,
            )
            raise

    @classmethod
    def delete_all(cls, tool_name: str) -> None:
        """Delete ALL ``tool_configs`` rows for a tool (uninstall wipe).

        Removes every row where ``tool_name`` matches, including any reserved
        system keys — this is the intended behavior for uninstall."""
        with Database.transaction():
            cls.filter("tool_name", tool_name).delete()
