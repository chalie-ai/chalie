"""
Config Service — Centralized configuration loading and provider resolution.

Loads agent configs from JSON files, merges the globally selected provider
into the resolved config, and exposes helpers for prompt text, connection
settings, and registered agent names.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ConfigService:
    """Static configuration service for agents, providers, prompts, and connections."""

    CONFIGS_DIR         = Path(__file__).resolve().parent.parent / "configs"
    PROMPTS_DIR         = Path(__file__).resolve().parent.parent / "prompts"
    CONNECTIONS_CONFIG  = str(CONFIGS_DIR / "connections.json")
    AGENTS_CONFIGS      = CONFIGS_DIR / "agents"

    @staticmethod
    def load_json(file_path: str) -> Dict[str, Any]:
        """Load and parse a JSON configuration file.

        Args:
            file_path: Absolute or relative path to the JSON file.

        Returns:
            Parsed dict from the JSON file contents.

        Raises:
            FileNotFoundError: If the file does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        with open(file_path, 'r') as f:
            return json.load(f)

    @staticmethod
    def load_text(file_path: str) -> str:
        """Load the text content of a file, returning empty string if missing.

        Args:
            file_path: Absolute or relative path to the text file.

        Returns:
            Stripped text content of the file, or empty string if it does not exist.
        """
        path = Path(file_path)
        if not path.exists():
            return ""
        return path.read_text().strip()

    @staticmethod
    def connections() -> Dict[str, Any]:
        """Load connections config (topic/queue names for MemoryStore key prefixes).

        Returns only the memory block from connections.json. Server bind address
        is managed by CLI args via runtime_config, not here.

        Returns:
            Dict with a single ``"memory"`` key containing the MemoryStore
            connection configuration dict.
        """
        try:
            base_config = ConfigService.load_json(ConfigService.CONNECTIONS_CONFIG)
        except Exception:
            base_config = {}

        # Support both "memory" (new) and "redis" (legacy) keys
        mem_config = base_config.get("memory") or base_config.get("redis", {})
        return {
            "memory": mem_config,
        }

    @staticmethod
    def get_providers() -> Dict[str, Any]:
        """Load providers from cache (with MemoryStore-backed invalidation).

        Returns:
            Dict mapping provider name to provider config dict.
        """
        from services.provider_cache_service import ProviderCacheService
        return ProviderCacheService.get_providers()

    @staticmethod
    def get_agent_config(agent_name: str) -> Dict[str, Any]:
        """Load raw agent config JSON without provider resolution.

        Args:
            agent_name: Agent config name (filename stem under configs/agents/)

        Returns:
            Dict containing the raw agent configuration.
        """
        return ConfigService.load_json(str(ConfigService.AGENTS_CONFIGS / (agent_name + ".json")))

    @staticmethod
    def resolve_agent_config(agent_name: str) -> Dict[str, Any]:
        """Load agent config and merge in the globally selected provider.

        Merge order: selected provider fields first, agent JSON fields on top
        (agent-specific overrides like temperature, timeout, format win).
        ``_job_name`` is always injected as the agent name — ``LoggingLLMService``
        uses it to name its per-job log file.

        Falls back to the first active provider when none is selected.
        Falls back to an empty provider dict on any lookup error so the agent
        config is still returned (providers missing → loud warning in logs).
        """
        agent_config = ConfigService.get_agent_config(agent_name)

        provider_config: Dict[str, Any] = {}
        try:
            from services.provider_cache_service import ProviderCacheService

            selected = ProviderCacheService.get_selected_provider()
            if selected:
                provider_config = selected
                logger.debug(
                    "[ConfigService] Using selected provider for agent '%s'", agent_name
                )
            else:
                # No selected provider — fall back to the first active provider.
                providers = ProviderCacheService.get_providers()
                if providers:
                    fallback_name = next(iter(providers))
                    provider_config = dict(providers[fallback_name])
                    logger.warning(
                        "[ConfigService] No provider selected for agent '%s', "
                        "falling back to '%s'",
                        agent_name, fallback_name,
                    )
                else:
                    logger.warning(
                        "[ConfigService] No providers configured — agent '%s' will "
                        "use agent-config fields only",
                        agent_name,
                    )
        except Exception as exc:
            logger.warning(
                "[ConfigService] Provider lookup failed for agent '%s': %s",
                agent_name, exc,
            )

        # Provider fields are the base; agent-specific fields override them.
        result = {**provider_config, **agent_config, '_job_name': agent_name}
        return result

