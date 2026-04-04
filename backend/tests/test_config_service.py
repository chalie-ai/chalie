"""
Tests for backend/services/config_service.py

ConfigService is a static utility class (no instances) providing layered
configuration: JSON files > hardcoded defaults. Runtime config (port, host)
is managed by runtime_config module, not ConfigService.
"""

import json
import pytest
from unittest.mock import patch

from services.config_service import ConfigService


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
class TestGetAllAgents:
    """Tests for get_all_agents directory scanning."""

    def test_returns_agent_names_from_directory(self, tmp_path):
        """Scans configs/agents/ and returns stem names of JSON files."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "frontal-cortex.json").write_text("{}")
        (agents_dir / "mode-router.json").write_text("{}")
        (agents_dir / "trait-extraction.json").write_text("{}")

        with patch.object(ConfigService, 'AGENTS_CONFIGS', agents_dir):
            result = ConfigService.get_all_agents()

        assert sorted(result) == ["frontal-cortex", "mode-router", "trait-extraction"]

    def test_returns_empty_list_for_empty_directory(self, tmp_path):
        """Empty directory yields empty list."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        with patch.object(ConfigService, 'AGENTS_CONFIGS', agents_dir):
            result = ConfigService.get_all_agents()

        assert result == []
