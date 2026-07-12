"""Provider cache service — in-memory lazy cache with MemoryStore-backed invalidation."""

import logging
from typing import cast

logger = logging.getLogger(__name__)


class ProviderCacheService:
    """
    In-memory lazy cache for provider configurations.

    Cross-process invalidation via MemoryStore versioning: API mutations in one
    process invalidate caches in others on the next get_providers() call.
    Decryption happens ONLY on cache miss.
    """

    # Class-level state (shared across all calls in this process)
    _providers: dict[str, dict[str, object]] = {}  # {name: {platform, model, host, api_key, ...}}
    _version: int | None = None  # Last seen MemoryStore version


    @staticmethod
    def get_providers() -> dict[str, dict[str, object]]:

        # Check if MemoryStore version has changed (cross-process invalidation)
        current_version: int
        try:
            from services.memory_client import MemoryClientService
            store = MemoryClientService.create_connection()
            current_version = cast(int, store.get("providers:cache_version"))
            current_version = int(current_version) if current_version else 0
        except Exception as e:
            logger.warning(f"[ProviderCache] MemoryStore version check failed: {e}, using local cache")
            current_version = ProviderCacheService._version or 0

        # If version changed, invalidate local cache
        if current_version != ProviderCacheService._version:
            logger.debug(f"[ProviderCache] Cache invalidated (version {ProviderCacheService._version} → {current_version})")
            ProviderCacheService._providers = {}
            ProviderCacheService._version = current_version

        # Return cached providers if available
        if ProviderCacheService._providers:
            logger.debug(f"[ProviderCache] Cache hit: {len(ProviderCacheService._providers)} providers")
            return ProviderCacheService._providers

        # Cache miss — fetch from DB (cold start or after invalidation)
        logger.debug("[ProviderCache] Cache miss, fetching from DB")
        try:
            from services.provider_db_service import ProviderDbService

            service = ProviderDbService()

            # Fetch all active providers from DB (decryption happens here)
            db_providers = service.get_all_providers()

            # Convert to providers dict keyed by name
            providers_dict: dict[str, dict[str, object]] = {}
            for p in db_providers:
                # Include 'name' in the entry so downstream consumers
                # (ProviderCacheService.get_job_assignment, ProviderService._resolve)
                # can read the provider name from the resolved
                # config dict directly.
                # Without this, the resolved config has no way to identify
                # which provider row backs it, breaking DB lookups keyed by
                # provider name (e.g. compact_at threshold queries).
                entry: dict[str, object] = {
                    'name': p['name'],
                    'platform': p['platform'],
                    'model': p['model'],
                }
                if p.get('host'):
                    entry['host'] = p['host']
                if p.get('api_key'):
                    entry['api_key'] = p['api_key']
                if p.get('dimensions'):
                    entry['dimensions'] = p['dimensions']
                if p.get('timeout'):
                    entry['timeout'] = p['timeout']
                providers_dict[cast(str, p['name'])] = entry

            # Check vault state — api_keys decrypt to None when vault is locked.
            # Caching a vault-locked result would persist null api_keys until
            # the next provider DB change, causing "requires api_key" errors.
            vault_locked = False
            try:
                from services.vault_service import get_vault_service
                vault_locked = not get_vault_service().is_unlocked()
            except Exception:
                pass

            # Store in local cache — but only if we got results AND the vault
            # is unlocked (so api_keys were actually decrypted).
            # An empty fetch or a vault-locked fetch must NOT be cached so we
            # retry on the next call.
            if providers_dict and not vault_locked:
                ProviderCacheService._providers = providers_dict
                ProviderCacheService._version = current_version
                logger.debug(f"[ProviderCache] Loaded {len(providers_dict)} providers from DB")
            elif vault_locked:
                # Reset version so next call retries once the vault is unlocked
                ProviderCacheService._version = None
                logger.debug("[ProviderCache] Vault locked — not caching providers, will retry on next call")
            else:
                # Reset version so next call retries the DB fetch
                ProviderCacheService._version = None
                logger.debug("[ProviderCache] No providers found, will retry on next call")

            return providers_dict

        except Exception as e:
            # Don't cache failures — reset version so next call retries
            ProviderCacheService._version = None
            logger.warning(f"[ProviderCache] DB fetch failed: {e}, returning empty dict")
            return {}


    @staticmethod
    def get_selected_provider() -> dict[str, object] | None:

        try:
            from services.provider_db_service import ProviderDbService

            service = ProviderDbService()
            selected = service.get_selected_provider()
            if selected:
                return {
                    'platform': selected['platform'],
                    'model': selected['model'],
                    'host': selected.get('host'),
                    'api_key': selected.get('api_key'),
                    'dimensions': selected.get('dimensions'),
                    'timeout': selected.get('timeout'),
                    'max_tokens': selected.get('max_tokens'),
                }
        except Exception as e:
            logger.debug(f"[ProviderCache] Failed to get selected provider: {e}")
        return None


    @staticmethod
    def invalidate() -> None:
        """Invalidate and bump the MemoryStore version counter so other processes detect a cache miss on next call."""
        try:
            from services.memory_client import MemoryClientService
            store = MemoryClientService.create_connection()
            new_version = store.incr("providers:cache_version")
            logger.debug(f"[ProviderCache] Invalidated cache (new version: {new_version})")
        except Exception as e:
            logger.warning(f"[ProviderCache] MemoryStore version increment failed: {e}")

        # Clear local cache
        ProviderCacheService._providers = {}
        ProviderCacheService._version = None
