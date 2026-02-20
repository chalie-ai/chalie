"""Provider cache service — in-memory lazy cache with Redis-backed invalidation."""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ProviderCacheService:
    """
    In-memory lazy cache for provider configurations.

    Solves cache staleness by using Redis versioning:
    - API process mutates DB → increments Redis version → local caches invalidate
    - Worker processes: next get_providers() sees version mismatch → cache miss → re-fetch

    Decryption happens ONLY on cache miss (cold start or after provider change).
    """

    # Class-level state (shared across all calls in this process)
    _providers: Dict[str, Any] = {}  # {name: {platform, model, host, api_key, ...}}
    _job_assignments: Dict[str, str] = {}  # {job_name: provider_name}
    _version: Optional[int] = None  # Last seen Redis version


    @staticmethod
    def get_providers() -> Dict[str, Any]:
        """
        Get all providers with lazy caching and Redis-based invalidation.

        Returns:
            dict: {provider_name: {platform, model, host, api_key, ...}}
        """
        # Check if Redis version has changed (cross-process invalidation)
        try:
            from services.redis_client import RedisClientService
            redis_client = RedisClientService.create_connection()
            current_version = redis_client.get("providers:cache_version")
            current_version = int(current_version) if current_version else 0
        except Exception as e:
            logger.warning(f"[ProviderCache] Redis version check failed: {e}, using local cache")
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
                providers_dict[p['name']] = {
                    'platform': p['platform'],
                    'model': p['model'],
                }
                if p.get('host'):
                    providers_dict[p['name']]['host'] = p['host']
                if p.get('api_key'):
                    providers_dict[p['name']]['api_key'] = p['api_key']
                if p.get('dimensions'):
                    providers_dict[p['name']]['dimensions'] = p['dimensions']
                if p.get('timeout'):
                    providers_dict[p['name']]['timeout'] = p['timeout']

            # Fetch job assignments
            job_assignments = {}
            try:
                all_assignments = service.get_all_job_assignments()
                for assignment in all_assignments:
                    provider = service.get_provider_by_id(assignment['provider_id'])
                    if provider:
                        job_assignments[assignment['job_name']] = provider['name']
            except Exception as e:
                logger.debug(f"[ProviderCache] Could not load job assignments: {e}")

            # Store in local cache
            ProviderCacheService._providers = providers_dict
            ProviderCacheService._job_assignments = job_assignments
            ProviderCacheService._version = current_version

            logger.debug(f"[ProviderCache] Loaded {len(providers_dict)} providers and {len(job_assignments)} job assignments from DB")
            return providers_dict

        except Exception as e:
            logger.warning(f"[ProviderCache] DB fetch failed: {e}, returning empty dict")
            return {}


    @staticmethod
    def get_job_assignment(job_name: str) -> Optional[str]:
        """
        Get the assigned provider name for a job.

        Args:
            job_name: Job identifier (e.g., 'frontal-cortex', 'memory-chunker')

        Returns:
            Provider name if assigned, None otherwise
        """
        # Ensure cache is warm
        ProviderCacheService.get_providers()

        assignment = ProviderCacheService._job_assignments.get(job_name)
        if assignment:
            logger.debug(f"[ProviderCache] Job '{job_name}' assigned to provider '{assignment}'")
        return assignment


    @staticmethod
    def invalidate() -> None:
        """
        Invalidate provider cache across all processes.
        Called after create/update/delete provider.

        Uses Redis version counter for cross-process invalidation:
        - Increments Redis version
        - Clears local cache (this process)
        - Other processes detect version mismatch on next get_providers()
        """
        try:
            from services.redis_client import RedisClientService
            redis_client = RedisClientService.create_connection()
            new_version = redis_client.incr("providers:cache_version")
            logger.debug(f"[ProviderCache] Invalidated cache (new version: {new_version})")
        except Exception as e:
            logger.warning(f"[ProviderCache] Redis version increment failed: {e}")

        # Clear local cache
        ProviderCacheService._providers = {}
        ProviderCacheService._job_assignments = {}
        ProviderCacheService._version = None
