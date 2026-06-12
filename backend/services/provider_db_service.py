"""Provider database service — manages provider configuration in DB (SQLite)."""

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


def _infer_vision_support(platform: str, model: str) -> bool:
    """Infer vision support from platform and model name."""
    model_lower = model.lower()
    if platform == 'anthropic':
        return any(x in model_lower for x in ('sonnet', 'opus', 'haiku'))
    elif platform == 'openai':
        return any(x in model_lower for x in ('gpt-4', 'gpt-5'))
    elif platform == 'gemini':
        return True  # All Gemini models support vision
    elif platform == 'ollama':
        return any(x in model_lower for x in ('llava', 'vision', 'bakllava'))
    # Catalog / OpenAI-compatible: check for vision indicators in model name
    return any(x in model_lower for x in ('vision', 'vl-', 'omni', 'image'))


class ProviderDbService:
    """Manages provider configuration in database."""

    # Column list used by all SELECT queries — order matters for positional access
    _PROVIDER_COLS = (
        "id, name, platform, model, host, api_key, "
        "dimensions, timeout, is_active, supports_vision"
    )

    def __init__(self, database_service):
        """Initialise the service with an injected database dependency.

        Args:
            database_service: The shared database service instance used for
                all provider table reads and writes.
        """
        self.db = database_service

    @staticmethod
    def _seal_api_key(value: str) -> str:
        """Protect *value* with the vault DEK and return a base64-encoded string.

        Uses :func:`~services.vault_service.get_vault_service` to obtain the
        process-wide :class:`~services.vault_service.VaultService` singleton.
        The raw AES-256-GCM blob is base64-encoded so it can be stored safely
        in the TEXT ``api_key`` column.

        Args:
            value: Plaintext API-key string to protect.

        Returns:
            Base64-encoded ciphertext string (safe for TEXT column storage).

        Raises:
            :exc:`~services.vault_service.VaultLockedError`: If the vault has
                not been unlocked before this call.
        """
        import base64
        from services.vault_service import get_vault_service
        return base64.b64encode(get_vault_service().encrypt_str(value)).decode()

    @staticmethod
    def _unseal_api_key(encrypted_val) -> Optional[str]:
        """Stored value is always a base64-encoded AES-256-GCM blob (TEXT column)."""
        if not encrypted_val:
            return None
        import base64
        from services.vault_service import get_vault_service, VaultLockedError
        try:
            return get_vault_service().decrypt_str(base64.b64decode(encrypted_val))
        except VaultLockedError:
            return None

    def _row_to_provider(self, row) -> Dict[str, Any]:
        """Convert a database row to a provider dict, decrypting api_key.

        Column order: id, name, platform, model, host, api_key,
                      dimensions, timeout, is_active, supports_vision

        The ``api_key`` field is decrypted via :meth:`_unseal_api_key`.  If the
        vault is currently locked the field is returned as ``None`` so that
        listing and read operations still succeed before the user has logged in.

        Args:
            row: A ``sqlite3.Row`` dict-like object or a positional tuple
                as returned by ``cursor.fetchone()`` / ``fetchall()``.

        Returns:
            Provider dict with all fields populated (``api_key`` may be
            ``None`` when the vault is sealed).
        """
        if isinstance(row, dict):
            result = {
                "id": row['id'],
                "name": row['name'],
                "platform": row['platform'],
                "model": row['model'],
                "host": row['host'],
                "api_key": None,
                "dimensions": row['dimensions'],
                "timeout": row['timeout'],
                "is_active": bool(row['is_active']),
                "supports_vision": bool(row.get('supports_vision', 0)),
            }
            if row.get('api_key'):
                try:
                    result["api_key"] = self._unseal_api_key(row['api_key'])
                except Exception:
                    logger.warning(
                        "[Provider] Decrypt failed for provider id=%s", row['id'],
                    )
                    result["decrypt_failed"] = True
            return result
        # Positional access (tuple row)
        result = {
            "id": row[0],
            "name": row[1],
            "platform": row[2],
            "model": row[3],
            "host": row[4],
            "api_key": None,
            "dimensions": row[6],
            "timeout": row[7],
            "is_active": bool(row[8]),
            "supports_vision": bool(row[9]) if len(row) > 9 else False,
        }
        if row[5]:
            try:
                result["api_key"] = self._unseal_api_key(row[5])
            except Exception:
                logger.warning(
                    "[Provider] Decrypt failed for provider id=%s", row[0],
                )
                result["decrypt_failed"] = True
        return result

    def get_all_providers(self) -> List[Dict[str, Any]]:
        """Get all active providers."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT {self._PROVIDER_COLS} "
                "FROM providers WHERE is_active = 1 ORDER BY name"
            )
            rows = cursor.fetchall()
            cursor.close()
            return [self._row_to_provider(row) for row in rows]

    def list_providers_summary(self) -> List[Dict[str, Any]]:
        """Get all active providers without decrypting api_key (for REST listings)."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, name, platform, model, host, "
                "(api_key IS NOT NULL) AS has_api_key, "
                "dimensions, timeout, is_active, supports_vision "
                "FROM providers WHERE is_active = 1 ORDER BY name"
            )
            rows = cursor.fetchall()
            cursor.close()
            return [
                {
                    "id": row[0],
                    "name": row[1],
                    "platform": row[2],
                    "model": row[3],
                    "host": row[4],
                    "api_key": "***" if row[5] else None,
                    "dimensions": row[6],
                    "timeout": row[7],
                    "is_active": bool(row[8]),
                    "supports_vision": bool(row[9]),
                }
                for row in rows
            ]

    def get_provider_by_id(self, provider_id: int) -> Optional[Dict[str, Any]]:
        """Get provider by ID."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT {self._PROVIDER_COLS} "
                "FROM providers WHERE id = ? AND is_active = 1",
                (provider_id,)
            )
            row = cursor.fetchone()
            cursor.close()
            if not row:
                return None
            return self._row_to_provider(row)

    def create_provider(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new provider.

        Args:
            data: Dict with at least ``name``, ``platform``, and ``model``.

        Returns:
            The newly created provider dict.
        """
        default_model = data.get("model")
        if not default_model:
            raise ValueError("'model' is required")

        platform = data.get("platform", "")

        # Catalog provider — auto-inject host from models.dev data
        from services.provider_catalog_service import is_catalog_provider, get_catalog_provider
        if is_catalog_provider(platform):
            entry = get_catalog_provider(platform)
            if entry and not data.get("host"):
                data["host"] = entry.get("api", "")
            if not data.get("api_key"):
                raise ValueError("Catalog provider requires 'api_key' field")

        if platform == "openai_compatible":
            if not data.get("host"):
                raise ValueError(
                    "openai_compatible provider requires 'host' field "
                    "(base URL, e.g. 'https://api.minimax.io/v1')"
                )
            if not data.get("api_key"):
                raise ValueError(
                    "openai_compatible provider requires 'api_key' field"
                )

        api_key_val = data.get("api_key")
        encrypted_key = self._seal_api_key(api_key_val) if api_key_val else None

        # Auto-infer vision support if not explicitly provided
        if 'supports_vision' in data:
            vision = 1 if data['supports_vision'] else 0
        else:
            vision = (
                1 if _infer_vision_support(
                    data.get('platform', ''), default_model,
                ) else 0
            )

        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO providers (name, platform, model, host, api_key, "
                "dimensions, timeout, is_active, supports_vision) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    data["name"],
                    data["platform"],
                    default_model,
                    data.get("host"),
                    encrypted_key,
                    data.get("dimensions"),
                    data.get("timeout", 120),
                    1 if data.get("is_active", True) else 0,
                    vision,
                )
            )
            new_id = cursor.lastrowid
            cursor.close()

        # Backfill max_tokens + compact_at for the newly-created provider.
        # Same code path as the boot-time backfill — single source of truth.
        # Failure here is non-fatal: the row exists, the values can be
        # populated by the next boot or a manual retry.
        try:
            from services.provider_token_limits import backfill_one
            with self.db.connection() as conn:
                backfill_one(conn, new_id)
                conn.commit()
        except Exception as exc:
            logger.warning(
                "[ProviderDBService] post-create token-limit backfill failed for id=%s: %s",
                new_id, exc,
            )

        # Auto-activate this provider if none is currently selected. Atomic in
        # the service layer so it cannot race with a concurrent UI selection.
        # ``get_selected_provider`` returns ``None`` when the settings row is
        # missing/empty OR when the previously-selected provider has been
        # soft-deleted (is_active = 0) — in either case the new provider takes
        # over the active slot.
        if self.get_selected_provider() is None:
            self.set_selected_provider(new_id)

        # Fetch the newly created row and return it
        return self.get_provider_by_id(new_id)

    def update_provider(self, provider_id: int, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update a provider."""
        updates = []
        params = []

        for key in ["name", "platform", "model", "host", "dimensions", "timeout"]:
            if key in data:
                updates.append(f"{key} = ?")
                params.append(data[key])

        if "supports_vision" in data:
            updates.append("supports_vision = ?")
            params.append(1 if data["supports_vision"] else 0)

        if "is_active" in data:
            updates.append("is_active = ?")
            params.append(1 if data["is_active"] else 0)

        # Auto-infer vision support if platform or model changed
        # and supports_vision not explicit
        if 'supports_vision' not in data and ('platform' in data or 'model' in data):
            current = self.get_provider_by_id(provider_id)
            if current:
                platform = data.get('platform', current.get('platform', ''))
                model = data.get('model', current.get('model', ''))
                updates.append("supports_vision = ?")
                params.append(1 if _infer_vision_support(platform, model) else 0)

        # Handle api_key separately for encryption
        if "api_key" in data:
            if data["api_key"] is None:
                updates.append("api_key = NULL")
            else:
                updates.append("api_key = ?")
                params.append(self._seal_api_key(data["api_key"]))

        if not updates:
            return self.get_provider_by_id(provider_id)

        updates.append("updated_at = CURRENT_TIMESTAMP")
        params.append(provider_id)

        query = f"UPDATE providers SET {', '.join(updates)} WHERE id = ?"

        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(params))
            cursor.close()

        return self.get_provider_by_id(provider_id)

    def delete_provider(self, provider_id: int) -> bool:
        """Delete a provider (sets is_active to FALSE)."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE providers SET is_active = 0 WHERE id = ?",
                (provider_id,)
            )
            cursor.close()
        return True

    # ── Selected Provider ──────────────────────────────────────────

    def get_selected_provider(self) -> Optional[Dict[str, Any]]:
        """Return the currently selected provider, or None if none is set."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT value FROM settings WHERE key = 'selected_provider_id'"
            )
            row = cursor.fetchone()
            cursor.close()
            if not row or not row[0]:
                return None
            try:
                provider_id = int(row[0])
                return self.get_provider_by_id(provider_id)
            except (ValueError, TypeError):
                return None

    def set_selected_provider(self, provider_id: int) -> None:
        """Set the selected provider ID in settings."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO settings (key, value, value_type, description, is_sensitive) "
                "VALUES ('selected_provider_id', ?, 'int', 'ID of the active LLM provider', 0) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP",
                (str(provider_id),)
            )
            cursor.close()
