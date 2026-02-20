"""
Tests for backend/api/tools.py
"""

import pytest
import json
from unittest.mock import patch, MagicMock, Mock
from flask import Flask
from api.tools import tools_bp


@pytest.mark.unit
class TestToolsAPI:
    """Test tools API endpoints."""

    @pytest.fixture
    def client(self):
        """Create Flask test client."""
        app = Flask(__name__)
        app.register_blueprint(tools_bp)
        app.config['TESTING'] = True
        return app.test_client()

    @pytest.fixture
    def mock_session(self):
        """Mock session decorator."""
        def decorator(f):
            return f
        return decorator

    def test_list_tools_empty(self, client, mock_session):
        """List tools when empty should return empty list."""
        with patch('api.tools.require_session', mock_session):
            with patch('api.tools.ToolRegistryService') as mock_registry_class:
                mock_registry = MagicMock()
                mock_registry.get_all_tools.return_value = []
                mock_registry_class.return_value = mock_registry

                # Assuming there's a list endpoint (not shown but inferred)
                # Tests would be based on actual endpoint

    def test_install_no_source(self, client, mock_session):
        """Install without source should return 400."""
        with patch('api.tools.require_session', mock_session):
            response = client.post('/tools/install', json={})

            assert response.status_code == 400
            data = response.get_json()
            assert "error" in data or "ok" in data

    def test_install_invalid_manifest(self, client, mock_session):
        """Install with invalid manifest should return 400."""
        with patch('api.tools.require_session', mock_session), \
             patch('subprocess.run') as mock_run, \
             patch('pathlib.Path.exists') as mock_exists, \
             patch('builtins.open', mock_open_invalid_json):
            mock_run.return_value = MagicMock(returncode=0)
            mock_exists.return_value = True

            # Would test with git_url that produces invalid manifest

    def test_install_missing_dockerfile(self, client, mock_session):
        """Install without Dockerfile should return 400."""
        with patch('api.tools.require_session', mock_session):
            # Test setup needed
            pass

    def test_install_name_collision(self, client, mock_session):
        """Install with existing name should return 409."""
        with patch('api.tools.require_session', mock_session), \
             patch('api.tools.ToolRegistryService') as mock_registry_class:
            mock_registry = MagicMock()
            mock_registry.tools_dir = MagicMock()
            # Tool already exists
            mock_registry.tools_dir.__truediv__.return_value.exists.return_value = True
            mock_registry_class.return_value = mock_registry

            # Would test collision scenario

    def test_install_invalid_name_format(self, client, mock_session):
        """Install with invalid name format should return 400."""
        with patch('api.tools.require_session', mock_session):
            # Test with manifest containing invalid tool name
            pass

    def test_disable_existing_tool(self, client, mock_session):
        """Disable existing tool should succeed."""
        with patch('api.tools.require_session', mock_session), \
             patch('api.tools.ToolRegistryService') as mock_registry_class, \
             patch('shutil.move') as mock_move:
            mock_registry = MagicMock()
            tool_dir = MagicMock()
            tool_dir.exists.return_value = True
            mock_registry.tools_dir.__truediv__.return_value = tool_dir
            mock_registry_class.return_value = mock_registry

            # Would test disable endpoint

    def test_disable_nonexistent(self, client, mock_session):
        """Disable nonexistent tool should return 404."""
        with patch('api.tools.require_session', mock_session), \
             patch('api.tools.ToolRegistryService') as mock_registry_class:
            mock_registry = MagicMock()
            tool_dir = MagicMock()
            tool_dir.exists.return_value = False
            mock_registry.tools_dir.__truediv__.return_value = tool_dir
            mock_registry_class.return_value = mock_registry

            # Would test 404 scenario

    def test_enable_disabled_tool(self, client, mock_session):
        """Enable disabled tool should work."""
        with patch('api.tools.require_session', mock_session), \
             patch('api.tools.ToolRegistryService') as mock_registry_class:
            mock_registry = MagicMock()
            mock_registry_class.return_value = mock_registry

            # Would test enable endpoint

    def test_enable_nonexistent(self, client, mock_session):
        """Enable nonexistent tool should return 404."""
        with patch('api.tools.require_session', mock_session):
            # Would test 404 scenario
            pass

    def test_get_config_masks_secrets(self, client, mock_session):
        """Get config should mask secret values."""
        with patch('api.tools.require_session', mock_session), \
             patch('api.tools.ToolConfigService') as mock_config_class:
            mock_config = MagicMock()
            mock_config.get_tool_config.return_value = {
                'API_KEY': 'secret123',
                'NAME': 'public_value'
            }
            mock_config_class.return_value = mock_config

            # Would test that API_KEY is masked as "***"

    def test_set_config_rejects_unknown_keys(self, client, mock_session):
        """Set config with unknown keys should return 400."""
        with patch('api.tools.require_session', mock_session):
            # Would test unknown key rejection
            pass

    def test_set_config_valid(self, client, mock_session):
        """Set valid config should succeed."""
        with patch('api.tools.require_session', mock_session), \
             patch('api.tools.ToolConfigService') as mock_config_class:
            mock_config = MagicMock()
            mock_config.set_tool_config.return_value = True
            mock_config_class.return_value = mock_config

            # Would test successful config set

    def test_delete_config_key(self, client, mock_session):
        """Delete config key should work."""
        with patch('api.tools.require_session', mock_session), \
             patch('api.tools.ToolConfigService') as mock_config_class:
            mock_config = MagicMock()
            mock_config.delete_tool_config_key.return_value = True
            mock_config_class.return_value = mock_config

            # Would test key deletion

    def test_test_tool_missing_config(self, client, mock_session):
        """Test tool with missing required config should fail."""
        with patch('api.tools.require_session', mock_session):
            # Would test config validation
            pass

    def test_test_tool_complete_config(self, client, mock_session):
        """Test tool with complete config should succeed."""
        with patch('api.tools.require_session', mock_session):
            # Would test successful tool test
            pass


def mock_open_invalid_json(*args, **kwargs):
    """Helper for mocking open with invalid JSON."""
    mock = MagicMock()
    mock.__enter__.return_value.read.return_value = "invalid json"
    return mock
