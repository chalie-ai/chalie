"""
Tests for backend/services/config_service.py

ConfigService is a static utility class (no instances) providing layered
configuration: JSON files > hardcoded defaults. Runtime config (port, host)
is managed by runtime_config module, not ConfigService.
"""

import json
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from services.config_service import ConfigService


@pytest.mark.unit
class TestConnections:
    """Tests for ConfigService.connections() — loads MemoryStore topic/queue names."""

    def test_returns_memory_block_from_json(self):
        """MemoryStore topics and queues are loaded from connections.json."""
        json_config = {
            "memory": {
                "topics": {"chat_history": "llm-chat"},
                "queues": {"prompt_queue": {"name": "prompt-queue"}},
            },
        }
        with patch.object(ConfigService, 'load_json', return_value=json_config):
            result = ConfigService.connections()

        assert result["memory"]["topics"] == {"chat_history": "llm-chat"}
        assert result["memory"]["queues"]["prompt_queue"]["name"] == "prompt-queue"

    def test_returns_empty_memory_when_json_missing(self):
        """When connections.json is missing, returns empty memory dict."""
        with patch.object(ConfigService, 'load_json', side_effect=FileNotFoundError):
            result = ConfigService.connections()

        assert result["memory"] == {}

    def test_no_rest_api_or_voice_keys(self):
        """connections() no longer returns rest_api or voice sections."""
        json_config = {"memory": {"topics": {}}}
        with patch.object(ConfigService, 'load_json', return_value=json_config):
            result = ConfigService.connections()

        assert "rest_api" not in result
        assert "voice" not in result


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
