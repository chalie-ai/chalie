"""Provider database service — manages provider configuration in DB (SQLite)."""

import logging
from typing import Dict, Any, Optional, List

from services.log_utils import safe

logger = logging.getLogger(__name__)

# User-facing 409 message when a provider still holds an assigned role.
# Must stay byte-identical to the entry in api/providers._SAFE_VALIDATION_MESSAGES
# (imported there) — it is the single source of truth surfaced to the admin.
PROVIDER_IN_USE_MSG = (
    "This provider is in use as the main, vision, or delegate provider and "
    "cannot be deleted. Clear or reassign that role first."
)


class ProviderDbService:
    """Manages provider configuration in database."""

    # Column list used by all SELECT queries — order matters for positional access
    _PROVIDER_COLS = (
        "id, name, platform, model, host, api_key, "
        "dimensions, timeout, supports_vision, max_tokens"
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
                      dimensions, timeout, supports_vision, max_tokens

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
                "supports_vision": bool(row.get('supports_vision', 0)),
                "max_tokens": row.get('max_tokens'),
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
            "supports_vision": bool(row[8]) if len(row) > 8 else False,
            "max_tokens": row[9] if len(row) > 9 else None,
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
        """Get all providers."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT {self._PROVIDER_COLS} "
                "FROM providers ORDER BY name"
            )
            rows = cursor.fetchall()
            cursor.close()
            return [self._row_to_provider(row) for row in rows]

    def list_providers_summary(self) -> List[Dict[str, Any]]:
        """Get all providers without decrypting api_key (for REST listings)."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, name, platform, model, host, "
                "(api_key IS NOT NULL) AS has_api_key, "
                "dimensions, timeout, supports_vision "
                "FROM providers ORDER BY name"
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
                    "supports_vision": bool(row[8]),
                }
                for row in rows
            ]

    def get_provider_by_id(self, provider_id: int) -> Optional[Dict[str, Any]]:
        """Get provider by ID."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT {self._PROVIDER_COLS} "
                "FROM providers WHERE id = ?",
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

        # Verify vision support with a live probe (content-verifying).
        # Uses the plaintext api_key from `data` (pre-seal). Any failure → 0.
        # A key-requiring platform with no key cannot be probed — skip the
        # (guaranteed-to-fail) network call and default to 0; a later key edit
        # re-probes. Mirrors the update-path guard for create/update symmetry.
        if platform in self._KEY_REQUIRING and not api_key_val:
            logger.warning(
                "[Provider] Skipping vision probe on create for '%s' — "
                "no api_key available",
                safe(data.get('name')),
            )
            vision = 0
        else:
            from services.vision_probe import probe_provider
            probe_config = {
                'platform': data.get('platform', ''),
                'model': default_model,
                'api_key': api_key_val,
                'host': data.get('host'),
                'name': data.get('name'),
            }
            vision = 1 if probe_provider(probe_config) else 0

        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO providers (name, platform, model, host, api_key, "
                "dimensions, timeout, supports_vision) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    data["name"],
                    data["platform"],
                    default_model,
                    data.get("host"),
                    encrypted_key,
                    data.get("dimensions"),
                    data.get("timeout", 120),
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
        # deleted — in either case the new provider takes over the active slot.
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

        # Re-probe vision support only when a probe-relevant field changes
        # (platform/model/host/api_key). Name-only edits never re-probe.
        _probe_fields = {'platform', 'model', 'host', 'api_key'}
        if _probe_fields & set(data.keys()):
            current = self.get_provider_by_id(provider_id) or {}
            eff_platform = data.get('platform', current.get('platform', ''))
            eff_model = data.get('model', current.get('model', ''))
            eff_host = data.get('host', current.get('host'))
            # explicit new api_key wins; else reuse current (decrypted) value
            eff_api_key = data['api_key'] if 'api_key' in data else current.get('api_key')
            needs_key = eff_platform in self._KEY_REQUIRING
            if needs_key and not eff_api_key:
                # vault locked / no credential — cannot probe, leave column as-is
                logger.warning(
                    "[Provider] Skipping vision re-probe for id=%s — no api_key available",
                    provider_id,
                )
            else:
                from services.vision_probe import probe_provider
                probe_config = {
                    'platform': eff_platform, 'model': eff_model,
                    'api_key': eff_api_key, 'host': eff_host,
                    'name': current.get('name'),
                }
                updates.append("supports_vision = ?")
                params.append(1 if probe_provider(probe_config) else 0)

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

    def _provider_roles(self, provider_id: int) -> List[str]:
        """Return which assigned roles reference this provider id.

        A subset of ['main', 'vision', 'delegate'], built from the persisted
        settings pins: selected_provider_id (main), vision_provider_id,
        delegate_provider_id. Auto-fallbacks — vision/delegate defaulting to the
        selected provider when unpinned — are covered transitively by the 'main'
        role and are deliberately not counted here. An empty list means the
        provider holds no role and is safe to delete.
        """
        role_by_key = {
            'selected_provider_id': 'main',
            'vision_provider_id': 'vision',
            'delegate_provider_id': 'delegate',
        }
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT key, value FROM settings WHERE key IN "
                "('selected_provider_id', 'vision_provider_id', 'delegate_provider_id')"
            )
            rows = cursor.fetchall()
            cursor.close()
        assigned = set()
        for key, value in rows:
            if not value:
                continue
            try:
                if int(value) == provider_id:
                    assigned.add(role_by_key[key])
            except (ValueError, TypeError):
                # A non-numeric settings value means a corrupted pin: surface it
                # rather than silently treating the role as unassigned (which
                # could let a functionally-pinned provider be deleted).
                logger.warning(
                    "Ignoring non-numeric %s settings value %s while resolving "
                    "provider roles", key, safe(value)
                )
                continue
        return [role for role in ('main', 'vision', 'delegate') if role in assigned]

    def delete_provider(self, provider_id: int) -> bool:
        """Permanently delete a provider row.

        A provider currently assigned as the main (selected), vision, or
        delegate provider cannot be deleted: removing it would leave a dangling
        reference the resolver can no longer satisfy. Raises ValueError
        (surfaced as HTTP 409) when the provider still holds any of those roles —
        the caller must clear or reassign the role first.
        """
        if self._provider_roles(provider_id):
            raise ValueError(PROVIDER_IN_USE_MSG)
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM providers WHERE id = ?",
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

    # ── Vision Provider ────────────────────────────────────────────

    _KEY_REQUIRING = ('anthropic', 'openai', 'gemini', 'openai_compatible')

    def _resolve_vision_provider(self):
        """Resolve (provider_or_None, source) where source ∈ explicit|auto|none.

        explicit — an active, vision-capable provider id is stored in settings.
        auto     — no explicit id, but the active selected provider supports
                   vision (NOT persisted; surfaced to the UI so the user can
                   lock it in).
        none     — nothing usable.
        """
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT value FROM settings WHERE key = 'vision_provider_id'"
            )
            row = cursor.fetchone()
            cursor.close()
        if row and row[0]:
            try:
                pid = int(row[0])
                provider = self.get_provider_by_id(pid)  # active-only
                if provider and provider.get('supports_vision'):
                    return provider, 'explicit'
            except (ValueError, TypeError):
                pass
        selected = self.get_selected_provider()
        if selected and selected.get('supports_vision'):
            return selected, 'auto'
        return None, 'none'

    def get_vision_provider(self) -> Optional[Dict[str, Any]]:
        """Runtime resolver — the provider to use for image understanding, or None."""
        provider, _ = self._resolve_vision_provider()
        return provider

    def get_vision_provider_status(self) -> Dict[str, Any]:
        """UI-facing — {'provider': dict|None, 'source': 'explicit'|'auto'|'none'}."""
        provider, source = self._resolve_vision_provider()
        return {'provider': provider, 'source': source}

    def set_vision_provider(self, provider_id: Optional[int]) -> None:
        """Persist the explicit vision provider id, or clear it when None."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            if provider_id is None:
                cursor.execute(
                    "DELETE FROM settings WHERE key = 'vision_provider_id'"
                )
            else:
                cursor.execute(
                    "INSERT INTO settings (key, value, value_type, description, is_sensitive) "
                    "VALUES ('vision_provider_id', ?, 'int', 'ID of the provider used for image understanding', 0) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP",
                    (str(provider_id),)
                )
            cursor.close()

    # ── Delegate Provider ──────────────────────────────────────────
    #
    # The provider that subagent (delegate) turns — web_search, web_browse,
    # and friends — run on, independent of the main chat provider. Unlike
    # vision there is NO 'Disabled'/none clear state: clearing the pin falls
    # back to the selected (main) provider, never the last-pinned one. There is
    # also no supports_vision requirement — any active provider can be pinned.

    def _resolve_delegate_provider(self):
        """Resolve (provider_or_None, source) where source ∈ explicit|auto|none.

        explicit — an active provider id is stored in settings.
        auto     — no explicit id, but a selected (main) provider exists; the
                   delegate defaults to it (NOT persisted; the "use main
                   provider" default surfaced to the UI).
        none     — no pin and no selected provider.
        """
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT value FROM settings WHERE key = 'delegate_provider_id'"
            )
            row = cursor.fetchone()
            cursor.close()
        if row and row[0]:
            try:
                pid = int(row[0])
                provider = self.get_provider_by_id(pid)  # active-only
                if provider:
                    return provider, 'explicit'
            except (ValueError, TypeError):
                pass
        selected = self.get_selected_provider()
        if selected:
            return selected, 'auto'
        return None, 'none'

    def get_delegate_provider(self) -> Optional[Dict[str, Any]]:
        """Runtime resolver — the provider to use for delegate turns, or None."""
        provider, _ = self._resolve_delegate_provider()
        return provider

    def get_delegate_provider_status(self) -> Dict[str, Any]:
        """UI-facing — {'provider': dict|None, 'source': 'explicit'|'auto'|'none'}."""
        provider, source = self._resolve_delegate_provider()
        return {'provider': provider, 'source': source}

    def set_delegate_provider(self, provider_id: Optional[int]) -> None:
        """Persist the explicit delegate provider id, or clear it when None.

        Clearing does NOT disable delegate turns — resolution then falls back
        to the selected (main) provider (see _resolve_delegate_provider)."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            if provider_id is None:
                cursor.execute(
                    "DELETE FROM settings WHERE key = 'delegate_provider_id'"
                )
            else:
                cursor.execute(
                    "INSERT INTO settings (key, value, value_type, description, is_sensitive) "
                    "VALUES ('delegate_provider_id', ?, 'int', 'ID of the provider used for subagent (delegate) turns', 0) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP",
                    (str(provider_id),)
                )
            cursor.close()
