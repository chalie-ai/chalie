import json
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ConfigService:
    CONFIGS_DIR         = Path(__file__).resolve().parent.parent / "configs"
    PROMPTS_DIR         = Path(__file__).resolve().parent.parent / "prompts"
    CONNECTIONS_CONFIG  = str(CONFIGS_DIR / "connections.json")
    AGENTS_CONFIGS      = CONFIGS_DIR / "agents"

    @staticmethod
    def load_json(file_path: str) -> Dict[str, Any]:
        """Load JSON configuration file."""
        with open(file_path, 'r') as f:
            return json.load(f)

    @staticmethod
    def load_text(file_path: str) -> str:
        """Load text file content."""
        path = Path(file_path)
        if not path.exists():
            return ""
        return path.read_text().strip()

    @staticmethod
    def connections() -> Dict[str, Any]:
        """Load connections config (topic/queue names for MemoryStore key prefixes).

        Returns only the redis block from connections.json. Server bind address
        is managed by CLI args via runtime_config, not here.
        """
        try:
            base_config = ConfigService.load_json(ConfigService.CONNECTIONS_CONFIG)
        except Exception:
            base_config = {}

        return {
            "redis": base_config.get("redis", {}),
        }

    @staticmethod
    def get_providers() -> Dict[str, Any]:
        """Load providers from cache (with Redis-backed invalidation)."""
        from services.provider_cache_service import ProviderCacheService
        return ProviderCacheService.get_providers()

    @staticmethod
    def resolve_provider(config: dict) -> dict:
        """
        Resolve provider references in a config dict.

        If config has a "provider" key, merge provider defaults under agent overrides.
        If config has an "embedding_provider" key, merge embedding provider fields
        prefixed with "embedding_" (e.g. embedding_host, embedding_model, embedding_dimensions).
        If no provider key, return as-is (backward compatible).

        Merge order: provider defaults < agent config overrides.
        """
        result = dict(config)

        # Resolve main provider
        provider_name = result.pop('provider', None)
        if provider_name:
            providers = ConfigService.get_providers()
            provider = providers.get(provider_name)
            if provider is None:
                logger.warning(f"Unknown provider '{provider_name}', using config as-is")
            else:
                # Provider defaults, then agent overrides on top
                merged = dict(provider)
                merged.update(result)
                result = merged

        # Resolve embedding provider
        embed_provider_name = result.pop('embedding_provider', None)
        if embed_provider_name:
            providers = ConfigService.get_providers()
            embed_provider = providers.get(embed_provider_name)
            if embed_provider is None:
                logger.warning(f"Unknown embedding provider '{embed_provider_name}'")
            else:
                for key, value in embed_provider.items():
                    prefixed_key = f"embedding_{key}"
                    # Don't overwrite if agent config already has this key
                    if prefixed_key not in result:
                        result[prefixed_key] = value

        return result

    @staticmethod
    def get_agent_config(agent_name: str) -> Dict[str, Any]:
        return ConfigService.load_json(str(ConfigService.AGENTS_CONFIGS / (agent_name + ".json")))

    @staticmethod
    def resolve_agent_config(agent_name: str) -> Dict[str, Any]:
        """Load agent config and resolve any provider references.

        Checks DB for job assignment first, then uses JSON provider resolution.
        """
        config = ConfigService.get_agent_config(agent_name)

        # Check if there's a DB-level job assignment (via cached service)
        try:
            from services.provider_cache_service import ProviderCacheService

            assignment_provider = ProviderCacheService.get_job_assignment(agent_name)
            if assignment_provider:
                config['provider'] = assignment_provider
                logger.debug(f"[ConfigService] Using cached provider '{assignment_provider}' for job '{agent_name}'")
            else:
                # No explicit assignment — fall back to the first active provider.
                providers = ConfigService.get_providers()
                if providers:
                    fallback_name = next(iter(providers))
                    config['provider'] = fallback_name
                    logger.warning(f"[ConfigService] No provider assigned for job '{agent_name}', "
                                   f"falling back to '{fallback_name}'")
        except Exception as e:
            logger.warning(f"[ConfigService] Job assignment lookup failed for '{agent_name}': {e}")

        return ConfigService.resolve_provider(config)

    @staticmethod
    def get_agent_prompt(agent_name: str) -> str:
        return ConfigService.load_text(str(ConfigService.PROMPTS_DIR / (agent_name + ".md")))

    @staticmethod
    def get_all_agents() -> list[str]:
        agents = []

        for entry in ConfigService.AGENTS_CONFIGS.iterdir():
            if entry.is_file():
                agents.append(entry.stem)

        return agents