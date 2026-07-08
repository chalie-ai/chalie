"""Provider database service — manages provider configuration in DB (SQLite)."""

import logging
from typing import Dict, Optional, List, cast

from models.provider import Provider as ProviderModel
from models.setting import Setting
from services.database import Database
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

    @staticmethod
    def _seal_api_key(value: str) -> str:
        import base64
        from services.vault_service import get_vault_service
        return base64.b64encode(get_vault_service().encrypt_str(value)).decode()

    @staticmethod
    def _unseal_api_key(encrypted_val: Optional[str]) -> Optional[str]:
        if not encrypted_val:
            return None
        import base64
        from services.vault_service import get_vault_service, VaultLockedError
        try:
            return get_vault_service().decrypt_str(base64.b64decode(encrypted_val))
        except VaultLockedError:
            return None

    def _provider_to_dict(self, provider: ProviderModel) -> Dict[str, object]:
        result: Dict[str, object] = {
            "id": provider.id,
            "name": provider.name,
            "platform": provider.platform,
            "model": provider.model,
            "host": provider.host,
            "api_key": None,
            "dimensions": provider.dimensions,
            "timeout": provider.timeout,
            "supports_vision": bool(provider.supports_vision),
            "max_tokens": provider.max_tokens,
        }
        if provider.api_key:
            try:
                result["api_key"] = self._unseal_api_key(provider.api_key)
            except Exception:
                logger.warning(
                    "[Provider] Decrypt failed for provider id=%s", provider.id,
                )
                result["decrypt_failed"] = True
        return result

    def get_all_providers(self) -> List[Dict[str, object]]:
        return [self._provider_to_dict(p) for p in ProviderModel.order_by("name").get()]

    def list_providers_summary(self) -> List[Dict[str, object]]:
        return [
            {
                "id": row["id"],
                "name": row["name"],
                "platform": row["platform"],
                "model": row["model"],
                "host": row["host"],
                "api_key": "***" if row["has_api_key"] else None,
                "dimensions": row["dimensions"],
                "timeout": row["timeout"],
                "supports_vision": bool(cast(int, row["supports_vision"])),
            }
            for row in ProviderModel.list_summaries()
        ]

    def get_provider_by_id(self, provider_id: int) -> Optional[Dict[str, object]]:
        provider = ProviderModel.get(provider_id)
        if provider is None:
            return None
        return self._provider_to_dict(provider)

    def create_provider(self, data: Dict[str, object]) -> Optional[Dict[str, object]]:
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

        api_key_val = cast(Optional[str], data.get("api_key"))
        encrypted_key = self._seal_api_key(api_key_val) if api_key_val else None

        # Verify vision support with a live probe (content-verifying).
        # Uses the plaintext api_key from `data` (pre-seal). Any failure → 0.
        # A key-requiring platform with no key cannot be probed — skip the
        # (guaranteed-to-fail) network call and default to 0; a later key edit
        # re-probes. Mirrors the update-path guard for create/update symmetry.
        # codex_cli is chat-only and has no api_key — never probe it.
        if platform == 'codex_cli' or (platform in self._KEY_REQUIRING and not api_key_val):
            logger.warning(
                "[Provider] Skipping vision probe on create for '%s' — "
                "%s",
                safe(data.get('name')),
                "codex_cli is chat-only" if platform == 'codex_cli' else "no api_key available",
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

        with Database.transaction():
            provider = ProviderModel(
                name=data["name"],
                platform=data["platform"],
                model=default_model,
                host=data.get("host"),
                api_key=encrypted_key,
                dimensions=data.get("dimensions"),
                timeout=data.get("timeout", 120),
                supports_vision=vision,
            ).save()
        new_id = provider.id

        # Backfill max_tokens + compact_at for the newly-created provider.
        # Same code path as the boot-time backfill — single source of truth.
        # Failure here is non-fatal: the row exists, the values can be
        # populated by the next boot or a manual retry.
        try:
            from services.provider_token_limits import backfill_one
            with Database.transaction() as conn:
                backfill_one(conn, cast(int, new_id))
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
            self.set_selected_provider(cast(int, new_id))

        # Fetch the newly created row and return it
        return self.get_provider_by_id(cast(int, new_id))

    def update_provider(self, provider_id: int, data: Dict[str, object]) -> Optional[Dict[str, object]]:
        updates: Dict[str, object] = {}

        for key in ["name", "platform", "model", "host", "dimensions", "timeout"]:
            if key in data:
                updates[key] = data[key]

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
            if eff_platform == 'codex_cli':
                # codex_cli is chat-only — no vision probe, no api_key needed
                updates["supports_vision"] = 0
            elif eff_platform in self._KEY_REQUIRING and not eff_api_key:
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
                updates["supports_vision"] = 1 if probe_provider(probe_config) else 0

        # Handle api_key separately for encryption
        if "api_key" in data:
            if data["api_key"] is None:
                updates["api_key"] = None
            else:
                updates["api_key"] = self._seal_api_key(cast(str, data["api_key"]))

        if not updates:
            return self.get_provider_by_id(provider_id)

        with Database.transaction():
            ProviderModel.filter("id", provider_id).update(**updates)
            ProviderModel.touch(provider_id)

        return self.get_provider_by_id(provider_id)

    def _provider_roles(self, provider_id: int) -> List[str]:
        """An empty list means the provider holds no role and is safe to delete.
        """
        role_by_key = {
            'selected_provider_id': 'main',
            'vision_provider_id': 'vision',
            'delegate_provider_id': 'delegate',
        }
        rows = Setting.filter_in(
            "key",
            ['selected_provider_id', 'vision_provider_id', 'delegate_provider_id'],
        ).select("key", "value")
        assigned = set()
        for row in rows:
            key = cast(str, row["key"])
            value = cast(str, row["value"])
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
        if self._provider_roles(provider_id):
            raise ValueError(PROVIDER_IN_USE_MSG)
        with Database.transaction():
            ProviderModel.filter("id", provider_id).delete()
        return True

    # ── Selected Provider ──────────────────────────────────────────

    def get_selected_provider(self) -> Optional[Dict[str, object]]:
        value = Setting.get('selected_provider_id')
        if not value:
            return None
        try:
            provider_id = int(value)
            return self.get_provider_by_id(provider_id)
        except (ValueError, TypeError):
            return None

    def set_selected_provider(self, provider_id: int) -> None:
        with Database.transaction():
            Setting.set(
                'selected_provider_id', str(provider_id), 'int',
                'ID of the active LLM provider',
            )

    # ── Vision Provider ────────────────────────────────────────────

    _KEY_REQUIRING = ('anthropic', 'openai', 'gemini', 'openai_compatible')

    def _resolve_vision_provider(self) -> tuple[Optional[Dict[str, object]], str]:
        value = Setting.get('vision_provider_id')
        if value:
            try:
                pid = int(value)
                provider = self.get_provider_by_id(pid)  # active-only
                if provider and provider.get('supports_vision'):
                    return provider, 'explicit'
            except (ValueError, TypeError):
                pass
        selected = self.get_selected_provider()
        if selected and selected.get('supports_vision'):
            return selected, 'auto'
        return None, 'none'

    def get_vision_provider(self) -> Optional[Dict[str, object]]:
        provider, _ = self._resolve_vision_provider()
        return provider

    def get_vision_provider_status(self) -> Dict[str, object]:
        """UI-facing — {'provider': dict|None, 'source': 'explicit'|'auto'|'none'}."""
        provider, source = self._resolve_vision_provider()
        return {'provider': provider, 'source': source}

    def set_vision_provider(self, provider_id: Optional[int]) -> None:
        """Persist the explicit vision provider id, or clear it when None."""
        with Database.transaction():
            if provider_id is None:
                Setting.delete('vision_provider_id')
            else:
                Setting.set(
                    'vision_provider_id', str(provider_id), 'int',
                    'ID of the provider used for image understanding',
                )

    # ── Delegate Provider ──────────────────────────────────────────
    #
    # The provider that subagent (delegate) turns — web_search, web_browse,
    # and friends — run on, independent of the main chat provider. Unlike
    # vision there is NO 'Disabled'/none clear state: clearing the pin falls
    # back to the selected (main) provider, never the last-pinned one. There is
    # also no supports_vision requirement — any active provider can be pinned.

    def _resolve_delegate_provider(self) -> tuple[Optional[Dict[str, object]], str]:
        value = Setting.get('delegate_provider_id')
        if value:
            try:
                pid = int(value)
                provider = self.get_provider_by_id(pid)  # active-only
                if provider:
                    return provider, 'explicit'
            except (ValueError, TypeError):
                pass
        selected = self.get_selected_provider()
        if selected:
            return selected, 'auto'
        return None, 'none'

    def get_delegate_provider(self) -> Optional[Dict[str, object]]:
        """Runtime resolver — the provider to use for delegate turns, or None."""
        provider, _ = self._resolve_delegate_provider()
        return provider

    def get_delegate_provider_status(self) -> Dict[str, object]:
        """UI-facing — {'provider': dict|None, 'source': 'explicit'|'auto'|'none'}."""
        provider, source = self._resolve_delegate_provider()
        return {'provider': provider, 'source': source}

    def set_delegate_provider(self, provider_id: Optional[int]) -> None:
        """Persist the explicit delegate provider id, or clear it when None.

        Clearing does NOT disable delegate turns — resolution then falls back
        to the selected (main) provider (see _resolve_delegate_provider)."""
        with Database.transaction():
            if provider_id is None:
                Setting.delete('delegate_provider_id')
            else:
                Setting.set(
                    'delegate_provider_id', str(provider_id), 'int',
                    'ID of the provider used for subagent (delegate) turns',
                )
