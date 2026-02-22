import json
import socket
import os
import logging
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

logger = logging.getLogger(__name__)

# Global cache for resolved hostnames
_resolved_hosts = {}


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
    def _get_env_or_json(env_key: str, json_value, default=None, value_type=str):
        """
        Get configuration value from environment variable with fallback to JSON.

        Args:
            env_key: Environment variable name
            json_value: Value from JSON config
            default: Default value if neither env nor JSON provide a value
            value_type: Type to convert the value to (str, int, etc.)

        Returns:
            Configuration value from env (priority 1), JSON (priority 2), or default (priority 3)
        """
        env_value = os.getenv(env_key)

        if env_value is not None:
            logger.debug(f"Loading {env_key} from environment: {env_value}")
            try:
                return value_type(env_value)
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to convert {env_key}={env_value} to {value_type.__name__}, using JSON value")
                return json_value if json_value is not None else default

        if json_value is not None:
            logger.debug(f"Loading {env_key} from JSON: {json_value}")
            return json_value

        logger.debug(f"Loading {env_key} from default: {default}")
        return default

    @staticmethod
    def connections() -> Dict[str, Any]:
        """Load connections config with environment variable support.

        Environment variables take precedence over JSON values.
        Supported variables:
        - PostgreSQL: POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DATABASE, POSTGRES_USERNAME, POSTGRES_PASSWORD
        - Redis: REDIS_HOST, REDIS_PORT
        - REST API: REST_API_HOST, REST_API_PORT, API_KEY

        Note: LLM provider configuration (hosts, models, api keys) is managed exclusively
        via the database and REST API, not environment variables.
        """
        # Load base JSON config
        base_config = ConfigService.load_json(ConfigService.CONNECTIONS_CONFIG)

        # Override with environment variables
        config = {
            "postgresql": {
                "host": ConfigService._get_env_or_json(
                    "POSTGRES_HOST",
                    base_config.get("postgresql", {}).get("host"),
                    "localhost"
                ),
                "port": ConfigService._get_env_or_json(
                    "POSTGRES_PORT",
                    base_config.get("postgresql", {}).get("port"),
                    5432,
                    value_type=int
                ),
                "database": ConfigService._get_env_or_json(
                    "POSTGRES_DATABASE",
                    base_config.get("postgresql", {}).get("database"),
                    "chalie"
                ),
                "username": ConfigService._get_env_or_json(
                    "POSTGRES_USERNAME",
                    base_config.get("postgresql", {}).get("username"),
                    "postgres"
                ),
                "password": ConfigService._get_env_or_json(
                    "POSTGRES_PASSWORD",
                    base_config.get("postgresql", {}).get("password"),
                    ""
                ),
            },
            "redis": {
                "host": ConfigService._get_env_or_json(
                    "REDIS_HOST",
                    base_config.get("redis", {}).get("host"),
                    "localhost"
                ),
                "port": ConfigService._get_env_or_json(
                    "REDIS_PORT",
                    base_config.get("redis", {}).get("port"),
                    6379,
                    value_type=int
                ),
                # Preserve nested config from JSON (topics, queues)
                **{k: v for k, v in base_config.get("redis", {}).items()
                   if k not in ["host", "port"]}
            },
            "rest_api": {
                "host": ConfigService._get_env_or_json(
                    "REST_API_HOST",
                    base_config.get("rest_api", {}).get("host"),
                    "0.0.0.0"
                ),
                "port": ConfigService._get_env_or_json(
                    "REST_API_PORT",
                    base_config.get("rest_api", {}).get("port"),
                    8080,
                    value_type=int
                ),
                "api_key": ConfigService._get_env_or_json(
                    "API_KEY",
                    base_config.get("rest_api", {}).get("api_key"),
                    ""
                ),
            }
        }

        return config

    @staticmethod
    def resolve_hostnames():
        """Pre-resolve all hostnames in connections config to avoid DNS lookups in child processes.
        Useful on macOS where getaddrinfo can segfault in forked/spawned processes."""
        global _resolved_hosts

        connections = ConfigService.connections()

        # Resolve Redis hostname
        redis_config = connections.get("redis", {})
        redis_host = redis_config.get("host")
        if redis_host:
            try:
                resolved_ip = socket.gethostbyname(redis_host)
                _resolved_hosts[redis_host] = resolved_ip
                print(f"[ConfigService] Resolved '{redis_host}' -> {resolved_ip}")
            except socket.gaierror as e:
                print(f"[ConfigService] WARNING: Failed to resolve '{redis_host}': {e}")
                _resolved_hosts[redis_host] = redis_host  # Use as-is

        # Add more hostname resolutions here as needed for other services

    @staticmethod
    def get_resolved_host(hostname: str) -> str:
        """Get pre-resolved IP for hostname, or return hostname if not resolved."""
        global _resolved_hosts
        return _resolved_hosts.get(hostname, hostname)

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