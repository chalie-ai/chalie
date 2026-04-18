"""Extended tests for api/tools.py — framework-level (tool-agnostic) endpoint tests."""

import pytest
from unittest.mock import patch, MagicMock
from flask import Flask
from api.tools import tools_bp


pytestmark = pytest.mark.unit


# ── Flask endpoint tests ─────────────────────────────────────────────
#
# tools.py uses lazy imports inside function bodies:
#   from services.tool_registry_service import ToolRegistryService
#   from services.tool_config_service import ToolConfigService
#   from services.database_service import get_shared_db_service
#   from services.oauth_service import OAuthService
#
# So we patch at the source module (services.X.Y), not at api.tools.Y.

@pytest.fixture
def client():
    """Flask test client with tools blueprint registered, auth bypassed."""
    app = Flask(__name__)
    app.register_blueprint(tools_bp)
    app.config['TESTING'] = True

    with patch('services.auth_session_service.validate_session', return_value=True):
        with app.test_client() as c:
            yield c


class TestConfigCrud:

    def test_get_config_masks_secrets(self, client):
        """GET /tools/<name>/config masks secret values."""
        with patch('services.tool_registry_service.ToolRegistryService') as mock_reg, \
             patch('services.tool_config_service.ToolConfigService') as mock_cfg_cls, \
             patch('services.database_service.get_shared_db_service'):
            mock_instance = MagicMock()
            mock_instance.tools = {'test_tool': {'manifest': {}}}
            mock_instance.get_tool_config_schema.return_value = {
                'api_key': {'description': 'Key', 'secret': True},
                'region': {'description': 'Region', 'secret': False},
            }
            mock_reg.return_value = mock_instance

            mock_cfg_cls.RESERVED_KEYS = set()
            mock_cfg = MagicMock()
            mock_cfg.get_tool_config.return_value = {
                'api_key': 'sk-secret-1234',
                'region': 'us-east-1',
            }
            mock_cfg_cls.return_value = mock_cfg

            response = client.get('/tools/test_tool/config')
            assert response.status_code == 200
            data = response.get_json()
            # Response structure: {"tool_name": ..., "config_schema": ..., "config": ...}
            assert data['config']['api_key'] == '***'
            assert data['config']['region'] == 'us-east-1'


class TestOAuthEndpoints:

    def test_oauth_start_non_oauth_tool_returns_400(self, client):
        """GET /tools/<name>/oauth/start for non-OAuth tool → 400."""
        with patch('services.tool_registry_service.ToolRegistryService') as mock_reg, \
             patch('services.oauth_service.OAuthService'):
            mock_instance = MagicMock()
            mock_instance.tools = {
                'test_tool': {
                    'manifest': {
                        'auth': {},  # No OAuth config — empty dict is falsy
                    },
                }
            }
            mock_reg.return_value = mock_instance

            response = client.get('/tools/test_tool/oauth/start')
            assert response.status_code == 400

    def test_oauth_status_returns_connected_when_token_exists(self, client):
        """GET /tools/<name>/oauth/status → connected when access token present."""
        with patch('services.tool_registry_service.ToolRegistryService') as mock_reg, \
             patch('services.oauth_service.OAuthService') as mock_oauth_cls:
            mock_instance = MagicMock()
            mock_instance.tools = {
                'oauth_tool': {
                    'manifest': {
                        'auth': {'type': 'oauth2', 'authorization_url': 'https://example.com/auth'},
                    },
                }
            }
            mock_reg.return_value = mock_instance
            mock_oauth = MagicMock()
            mock_oauth.get_oauth_status.return_value = {
                'connected': True,
                'status': 'connected',
            }
            mock_oauth_cls.return_value = mock_oauth

            response = client.get('/tools/oauth_tool/oauth/status')
            assert response.status_code == 200
            data = response.get_json()
            assert data.get('connected') is True or data.get('status') == 'connected'

    def test_oauth_status_returns_disconnected_when_no_token(self, client):
        """GET /tools/<name>/oauth/status → disconnected when no token."""
        with patch('services.tool_registry_service.ToolRegistryService') as mock_reg, \
             patch('services.oauth_service.OAuthService') as mock_oauth_cls:
            mock_instance = MagicMock()
            mock_instance.tools = {
                'oauth_tool': {
                    'manifest': {
                        'auth': {'type': 'oauth2', 'authorization_url': 'https://example.com/auth'},
                    },
                }
            }
            mock_reg.return_value = mock_instance
            mock_oauth = MagicMock()
            mock_oauth.get_oauth_status.return_value = {
                'connected': False,
                'status': 'disconnected',
            }
            mock_oauth_cls.return_value = mock_oauth

            response = client.get('/tools/oauth_tool/oauth/status')
            assert response.status_code == 200
            data = response.get_json()
            assert data.get('connected') is False or data.get('status') == 'disconnected'
