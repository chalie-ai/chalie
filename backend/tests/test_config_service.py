"""
Tests for backend/services/config_service.py

ConfigService is a static utility class (no instances) providing layered
configuration: environment variables > JSON files > hardcoded defaults.
"""

import json
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from services.config_service import ConfigService


@pytest.mark.unit
class TestGetEnvOrJson:
    """Tests for ConfigService._get_env_or_json precedence logic."""

    def test_returns_env_var_when_set(self, monkeypatch):
        """Environment variable takes highest precedence."""
        monkeypatch.setenv("TEST_CFG_KEY", "from_env")
        result = ConfigService._get_env_or_json(
            "TEST_CFG_KEY", "from_json", "from_default"
        )
        assert result == "from_env"

    def test_returns_json_value_when_no_env_var(self, monkeypatch):
        """JSON value is used when no environment variable is set."""
        monkeypatch.delenv("TEST_CFG_KEY", raising=False)
        result = ConfigService._get_env_or_json(
            "TEST_CFG_KEY", "from_json", "from_default"
        )
        assert result == "from_json"

    def test_returns_default_when_both_none(self, monkeypatch):
        """Default is the final fallback when env and JSON are both absent."""
        monkeypatch.delenv("TEST_CFG_KEY", raising=False)
        result = ConfigService._get_env_or_json(
            "TEST_CFG_KEY", None, "from_default"
        )
        assert result == "from_default"

    def test_type_conversion_failure_falls_back_to_json(self, monkeypatch):
        """When env value cannot be converted to requested type, fall back to JSON."""
        monkeypatch.setenv("TEST_CFG_INT", "not_a_number")
        result = ConfigService._get_env_or_json(
            "TEST_CFG_INT", 9999, default=0, value_type=int
        )
        assert result == 9999

    def test_type_conversion_failure_falls_back_to_default_when_json_none(self, monkeypatch):
        """When env conversion fails and JSON is None, fall back to default."""
        monkeypatch.setenv("TEST_CFG_INT", "not_a_number")
        result = ConfigService._get_env_or_json(
            "TEST_CFG_INT", None, default=42, value_type=int
        )
        assert result == 42

    def test_int_type_conversion_succeeds(self, monkeypatch):
        """Environment variable is converted to the requested int type."""
        monkeypatch.setenv("TEST_CFG_PORT", "5433")
        result = ConfigService._get_env_or_json(
            "TEST_CFG_PORT", 5432, default=5432, value_type=int
        )
        assert result == 5433
        assert isinstance(result, int)


@pytest.mark.unit
class TestConnections:
    """Tests for ConfigService.connections() merging logic."""

    def test_env_vars_override_json_values(self, monkeypatch):
        """Environment variables must override corresponding JSON fields."""
        json_config = {
            "postgresql": {"host": "json-host", "port": 5432},
            "redis": {"host": "json-redis", "port": 6379},
            "rest_api": {"host": "0.0.0.0", "port": 8080},
        }
        monkeypatch.setenv("POSTGRES_HOST", "env-pg-host")
        monkeypatch.setenv("REDIS_PORT", "7777")

        with patch.object(ConfigService, 'load_json', return_value=json_config):
            result = ConfigService.connections()

        assert result["postgresql"]["host"] == "env-pg-host"
        assert result["redis"]["port"] == 7777
        # Non-overridden values remain from JSON
        assert result["redis"]["host"] == "json-redis"
        assert result["postgresql"]["port"] == 5432

    def test_defaults_when_no_env_or_json(self, monkeypatch):
        """When JSON has no values and no env vars are set, defaults apply."""
        # Remove any env vars that could interfere
        for var in ("POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DATABASE",
                     "POSTGRES_USERNAME", "POSTGRES_PASSWORD",
                     "REDIS_HOST", "REDIS_PORT",
                     "REST_API_HOST", "REST_API_PORT", "API_KEY"):
            monkeypatch.delenv(var, raising=False)

        empty_json = {}
        with patch.object(ConfigService, 'load_json', return_value=empty_json):
            result = ConfigService.connections()

        assert result["postgresql"]["host"] == "localhost"
        assert result["postgresql"]["port"] == 5432
        assert result["postgresql"]["database"] == "chalie"
        assert result["postgresql"]["username"] == "postgres"
        assert result["redis"]["host"] == "localhost"
        assert result["redis"]["port"] == 6379
        assert result["rest_api"]["host"] == "0.0.0.0"
        assert result["rest_api"]["port"] == 8080

    def test_preserves_nested_redis_config(self, monkeypatch):
        """Topics and queues from JSON are preserved alongside host/port."""
        for var in ("REDIS_HOST", "REDIS_PORT"):
            monkeypatch.delenv(var, raising=False)

        json_config = {
            "redis": {
                "host": "redis-server",
                "port": 6379,
                "topics": {"chat_history": "llm-chat"},
                "queues": {"prompt_queue": {"name": "prompt-queue"}},
            },
            "postgresql": {},
            "rest_api": {},
        }
        with patch.object(ConfigService, 'load_json', return_value=json_config):
            result = ConfigService.connections()

        assert result["redis"]["topics"] == {"chat_history": "llm-chat"}
        assert result["redis"]["queues"]["prompt_queue"]["name"] == "prompt-queue"


@pytest.mark.unit
class TestFileIO:
    """Tests for load_text and load_json file reading."""

    def test_load_text_returns_empty_for_missing_file(self, tmp_path):
        """Missing file should return empty string, not raise."""
        result = ConfigService.load_text(str(tmp_path / "nonexistent.md"))
        assert result == ""

    def test_load_text_returns_stripped_content(self, tmp_path):
        """Existing file content is returned with whitespace stripped."""
        test_file = tmp_path / "prompt.md"
        test_file.write_text("  Hello, world!  \n\n")
        result = ConfigService.load_text(str(test_file))
        assert result == "Hello, world!"

    def test_load_json_parses_valid_file(self, tmp_path):
        """Valid JSON file is parsed into a dict."""
        test_file = tmp_path / "config.json"
        data = {"model": "test-model", "temperature": 0.7}
        test_file.write_text(json.dumps(data))
        result = ConfigService.load_json(str(test_file))
        assert result == data

    def test_load_json_raises_on_missing_file(self):
        """Missing JSON file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ConfigService.load_json("/nonexistent/path/config.json")


@pytest.mark.unit
class TestAgentConfigAndPrompt:
    """Tests for get_agent_config and get_agent_prompt delegation."""

    def test_get_agent_config_loads_correct_file(self):
        """get_agent_config delegates to load_json with the correct path."""
        expected = {"model": "qwen:8b", "temperature": 0.5}
        with patch.object(ConfigService, 'load_json', return_value=expected) as mock_load:
            result = ConfigService.get_agent_config("frontal-cortex")

        mock_load.assert_called_once()
        call_path = mock_load.call_args[0][0]
        assert call_path.endswith("configs/agents/frontal-cortex.json")
        assert result == expected

    def test_get_agent_prompt_loads_correct_file(self):
        """get_agent_prompt delegates to load_text with the correct path."""
        with patch.object(ConfigService, 'load_text', return_value="System prompt here") as mock_load:
            result = ConfigService.get_agent_prompt("memory-chunker")

        mock_load.assert_called_once()
        call_path = mock_load.call_args[0][0]
        assert call_path.endswith("prompts/memory-chunker.md")
        assert result == "System prompt here"


@pytest.mark.unit
class TestResolveProvider:
    """Tests for resolve_provider merging logic."""

    def test_merges_provider_defaults_under_agent_overrides(self):
        """Agent-level overrides take precedence over provider defaults."""
        providers = {
            "ollama-local": {
                "platform": "ollama",
                "model": "qwen:8b",
                "host": "http://localhost:11434",
            }
        }
        agent_config = {
            "provider": "ollama-local",
            "model": "llama3:8b",  # override the provider's model
            "temperature": 0.3,
        }

        with patch.object(ConfigService, 'get_providers', return_value=providers):
            result = ConfigService.resolve_provider(agent_config)

        # Agent override wins for model
        assert result["model"] == "llama3:8b"
        # Provider defaults fill in missing fields
        assert result["platform"] == "ollama"
        assert result["host"] == "http://localhost:11434"
        # Agent-only field preserved
        assert result["temperature"] == 0.3
        # provider key is consumed (popped)
        assert "provider" not in result

    def test_returns_config_as_is_when_no_provider_key(self):
        """Config without a provider key is returned unchanged (backward compat)."""
        config = {"model": "direct-model", "temperature": 0.5}
        result = ConfigService.resolve_provider(dict(config))
        assert result == config

    def test_warns_on_unknown_provider(self):
        """Unknown provider name logs a warning and returns config as-is."""
        config = {"provider": "nonexistent", "model": "test"}
        with patch.object(ConfigService, 'get_providers', return_value={}):
            result = ConfigService.resolve_provider(config)

        assert result["model"] == "test"
        assert "provider" not in result


@pytest.mark.unit
class TestResolveAgentConfig:
    """Tests for resolve_agent_config with DB job assignment lookup."""

    def test_uses_db_job_assignment(self):
        """When a DB job assignment exists, it overrides the JSON provider."""
        agent_json = {"model": "default-model", "temperature": 0.5}
        providers = {
            "openai-prod": {
                "platform": "openai",
                "model": "gpt-4",
                "api_key": "sk-test",
            }
        }

        with patch.object(ConfigService, 'get_agent_config', return_value=agent_json), \
             patch('services.provider_cache_service.ProviderCacheService.get_job_assignment',
                   return_value="openai-prod"), \
             patch.object(ConfigService, 'get_providers', return_value=providers):
            result = ConfigService.resolve_agent_config("frontal-cortex")

        # DB assignment provider is merged in
        assert result["platform"] == "openai"
        # Agent override for model wins (agent_json had model set)
        assert result["model"] == "default-model"
        assert result["temperature"] == 0.5

    def test_falls_back_to_first_provider_when_no_assignment(self):
        """When no DB assignment exists, falls back to first active provider."""
        agent_json = {"temperature": 0.7}
        providers = {
            "anthropic-main": {
                "platform": "anthropic",
                "model": "claude-3",
            }
        }

        with patch.object(ConfigService, 'get_agent_config', return_value=agent_json), \
             patch('services.provider_cache_service.ProviderCacheService.get_job_assignment',
                   return_value=None), \
             patch.object(ConfigService, 'get_providers', return_value=providers):
            result = ConfigService.resolve_agent_config("some-agent")

        assert result["platform"] == "anthropic"
        assert result["model"] == "claude-3"
        assert result["temperature"] == 0.7


@pytest.mark.unit
class TestGetAllAgents:
    """Tests for get_all_agents directory scanning."""

    def test_returns_agent_names_from_directory(self, tmp_path):
        """Scans configs/agents/ and returns stem names of JSON files."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "frontal-cortex.json").write_text("{}")
        (agents_dir / "memory-chunker.json").write_text("{}")
        (agents_dir / "mode-router.json").write_text("{}")

        with patch.object(ConfigService, 'AGENTS_CONFIGS', agents_dir):
            result = ConfigService.get_all_agents()

        assert sorted(result) == ["frontal-cortex", "memory-chunker", "mode-router"]

    def test_returns_empty_list_for_empty_directory(self, tmp_path):
        """Empty directory yields empty list."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        with patch.object(ConfigService, 'AGENTS_CONFIGS', agents_dir):
            result = ConfigService.get_all_agents()

        assert result == []
