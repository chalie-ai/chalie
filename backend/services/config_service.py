"""JSON loading and connection settings."""

import json
import logging
from typing import Dict, cast

from services.file_mapper_service import FileMapperService

logger = logging.getLogger(__name__)


class ConfigService:
    """Static configuration service for connections and providers."""

    CONFIGS_DIR         = FileMapperService.get_configs_path()
    CONNECTIONS_CONFIG  = str(FileMapperService.get_configs_path("connections.json"))

    @staticmethod
    def load_json(file_path: str) -> Dict[str, object]:
        with open(file_path, 'r') as f:
            return cast(Dict[str, object], json.load(f))

    @staticmethod
    def connections() -> Dict[str, object]:
        """Load connections config (MemoryStore key prefixes)."""
        try:
            base_config = ConfigService.load_json(ConfigService.CONNECTIONS_CONFIG)
        except Exception:
            base_config = {}

        mem_config = base_config.get("memory") or base_config.get("redis", {})
        return {
            "memory": mem_config,
        }

    @staticmethod
    def get_providers() -> Dict[str, Dict[str, object]]:
        """Load providers from cache (with MemoryStore-backed invalidation)."""
        from services.provider_cache_service import ProviderCacheService
        return ProviderCacheService.get_providers()
