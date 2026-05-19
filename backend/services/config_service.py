"""
Config Service — JSON loading and connection settings.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ConfigService:
    """Static configuration service for connections and providers."""

    CONFIGS_DIR         = Path(__file__).resolve().parent.parent / "configs"
    CONNECTIONS_CONFIG  = str(CONFIGS_DIR / "connections.json")

    @staticmethod
    def load_json(file_path: str) -> Dict[str, Any]:
        with open(file_path, 'r') as f:
            return json.load(f)

    @staticmethod
    def connections() -> Dict[str, Any]:
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
    def get_providers() -> Dict[str, Any]:
        """Load providers from cache (with MemoryStore-backed invalidation)."""
        from services.provider_cache_service import ProviderCacheService
        return ProviderCacheService.get_providers()
