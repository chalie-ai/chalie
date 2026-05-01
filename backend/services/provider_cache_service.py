"""Provider cache service — in-memory lazy cache with MemoryStore-backed invalidation."""

import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class ProviderCacheService:
    """
    In-memory lazy cache for provider configurations.

    Solves cache staleness by using MemoryStore versioning:
    - API process mutates DB → increments MemoryStore version → local caches invalidate
    - Worker processes: next get_providers() sees version mismatch → cache miss → re-fetch

    Decryption happens ONLY on cache miss (cold start or after provider change).
    """

    # Class-level state (shared across all calls in this process)
    _providers: Dict[str, Any] = {}  # {name: {platform, model, models, host, api_key, ...}}
    _job_assignments: Dict[str, Tuple[str, Optional[str]]] = {}  # {job_name: (provider_name, model_override)}
    _version: Optional[int] = None  # Last seen MemoryStore version


    @staticmethod
    def get_providers() -> Dict[str, Any]:
        """
        Get all providers with lazy caching and MemoryStore-based invalidation.

        Returns:
            dict: {provider_name: {platform, model, models, host, api_key, ...}}
        """
        # Check if MemoryStore version has changed (cross-process invalidation)
        try:
            from services.memory_client import MemoryClientService
            store = MemoryClientService.create_connection()
            current_version = store.get("providers:cache_version")
            current_version = int(current_version) if current_version else 0
        except Exception as e:
            logger.warning(f"[ProviderCache] MemoryStore version check failed: {e}, using local cache")
            current_version = ProviderCacheService._version or 0

        # If version changed, invalidate local cache
        if current_version != ProviderCacheService._version:
            logger.debug(f"[ProviderCache] Cache invalidated (version {ProviderCacheService._version} → {current_version})")
            ProviderCacheService._providers = {}
            ProviderCacheService._job_assignments = {}
            ProviderCacheService._version = current_version

        # Return cached providers if available
        if ProviderCacheService._providers:
            logger.debug(f"[ProviderCache] Cache hit: {len(ProviderCacheService._providers)} providers")
            return ProviderCacheService._providers

        # Cache miss — fetch from DB (cold start or after invalidation)
        logger.debug("[ProviderCache] Cache miss, fetching from DB")
        try:
            from services.database_service import get_shared_db_service
            from services.provider_db_service import ProviderDbService

            db = get_shared_db_service()
            service = ProviderDbService(db)

            # Fetch all active providers from DB (decryption happens here)
            db_providers = service.get_all_providers()

            # Convert to providers dict keyed by name
            providers_dict = {}
            for p in db_providers:
                # Include 'name' in the entry so downstream consumers
                # (Providers.get_compact_at, ProviderCacheService.get_job_assignment,
                # any code calling ConfigService.resolve_agent_config) can read
                # the provider name from the resolved config dict directly.
                # Without this, the resolved config has no way to identify
                # which provider row backs it, breaking DB lookups keyed by
                # provider name (e.g. compact_at threshold queries).
                entry = {
                    'name': p['name'],
                    'platform': p['platform'],
                    'model': p['model'],
                    'models': p.get('models', [p['model']]),
                }
                if p.get('host'):
                    entry['host'] = p['host']
                if p.get('api_key'):
                    entry['api_key'] = p['api_key']
                if p.get('dimensions'):
                    entry['dimensions'] = p['dimensions']
                if p.get('timeout'):
                    entry['timeout'] = p['timeout']
                providers_dict[p['name']] = entry

            # Fetch job assignments (skip assignments pointing to inactive/deleted providers)
            job_assignments = {}
            try:
                all_assignments = service.get_all_job_assignments()
                for assignment in all_assignments:
                    provider = service.get_provider_by_id(assignment['provider_id'])
                    if provider and provider.get('is_active', True):
                        job_assignments[assignment['job_name']] = (
                            provider['name'],
                            assignment.get('model'),  # model override or None
                        )
            except Exception as e:
                logger.debug(f"[ProviderCache] Could not load job assignments: {e}")

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
                ProviderCacheService._job_assignments = job_assignments
                ProviderCacheService._version = current_version
                logger.debug(f"[ProviderCache] Loaded {len(providers_dict)} providers and {len(job_assignments)} job assignments from DB")
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
    def get_job_assignment(job_name: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Get the assigned provider name and model override for a job.

        Args:
            job_name: Job identifier (e.g., 'frontal-cortex', 'chat-vision')

        Returns:
            Tuple of (provider_name, model_override). Both None if unassigned.
            model_override is None when the job should use the provider's default model.
        """
        # Ensure cache is warm
        ProviderCacheService.get_providers()

        assignment = ProviderCacheService._job_assignments.get(job_name)
        if assignment:
            provider_name, model_override = assignment
            logger.debug(f"[ProviderCache] Job '{job_name}' assigned to provider '{provider_name}'"
                         f"{f' (model: {model_override})' if model_override else ''}")
            return provider_name, model_override
        return None, None


    @staticmethod
    def resolve_for_job(job_name: str, platforms: Optional[set] = None) -> Optional[Dict[str, Any]]:
        """
        Resolve a provider config for a given job.

        Resolution order:
          1. Job-specific assignment (set via /api/providers/jobs)
          2. First available provider, optionally filtered by platform

        If the assignment has a model override, the returned config's 'model'
        field is set to that override. Downstream consumers always see a single
        model string.

        Args:
            job_name:  Job identifier (e.g. 'frontal-cortex', 'chat-vision')
            platforms: Optional set of platform names to restrict the fallback
                       (e.g. {'gemini', 'anthropic', 'openai'}). Has no effect
                       on the job-assigned provider — that is always returned as-is.

        Returns:
            Provider config dict {platform, model, api_key, host, ...} or None.
        """
        providers = ProviderCacheService.get_providers()
        if not providers:
            return None

        # 1. Job-specific assignment
        assigned_name, model_override = ProviderCacheService.get_job_assignment(job_name)
        if assigned_name and assigned_name in providers:
            config = dict(providers[assigned_name])
            if model_override:
                config['model'] = model_override
            logger.debug(f"[ProviderCache] Resolved job '{job_name}' → provider '{assigned_name}'"
                         f"{f' model={model_override}' if model_override else ''}")
            return config

        # 2. Fallback: first provider matching the platform filter (or any provider)
        for name, config in providers.items():
            if platforms is None or config.get('platform') in platforms:
                logger.debug(f"[ProviderCache] No assignment for '{job_name}', falling back to '{name}'")
                return dict(config)

        logger.warning(f"[ProviderCache] No provider found for job '{job_name}' (platforms={platforms})")
        return None


    @staticmethod
    def invalidate() -> None:
        """
        Invalidate provider cache across all processes.
        Called after create/update/delete provider.

        Uses MemoryStore version counter for cross-process invalidation:
        - Increments MemoryStore version
        - Clears local cache (this process)
        - Other processes detect version mismatch on next get_providers()
        """
        try:
            from services.memory_client import MemoryClientService
            store = MemoryClientService.create_connection()
            new_version = store.incr("providers:cache_version")
            logger.debug(f"[ProviderCache] Invalidated cache (new version: {new_version})")
        except Exception as e:
            logger.warning(f"[ProviderCache] MemoryStore version increment failed: {e}")

        # Clear local cache
        ProviderCacheService._providers = {}
        ProviderCacheService._job_assignments = {}
        ProviderCacheService._version = None
