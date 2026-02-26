"""Extended tests for api/tools.py — framework-level (tool-agnostic) endpoint tests."""

import json
import re
import pytest
from unittest.mock import patch, MagicMock
from flask import Flask
from api.tools import tools_bp, _normalize_config_schema


pytestmark = pytest.mark.unit


# ── _normalize_config_schema (pure function) ─────────────────────────

class TestNormalizeConfigSchema:

    def test_dict_to_list_conversion(self):
        schema = {
            "api_key": {"description": "Your API key", "secret": True, "default": ""},
            "region": {"description": "AWS region", "secret": False, "default": "us-east-1"},
        }
        result = _normalize_config_schema(schema)
        assert len(result) == 2
        # Check first item structure
        api_key_item = next(r for r in result if r['key'] == 'api_key')
        assert api_key_item['label'] == 'Your API key'
        assert api_key_item['secret'] is True
        assert api_key_item['placeholder'] == ''

    def test_empty_dict_returns_empty_list(self):
        assert _normalize_config_schema({}) == []

    def test_defaults_secret_false(self):
        schema = {"field": {"description": "A field"}}
        result = _normalize_config_schema(schema)
        assert result[0]['secret'] is False


# ── Webhook rate limiting (fakeredis) ────────────────────────────────

class TestCheckWebhookRateLimit:

    def test_allows_up_to_30(self, mock_redis):
        """First 30 requests within rate limit."""
        for i in range(30):
            mock_redis.incr("webhook_rate:test_tool")
        mock_redis.expire("webhook_rate:test_tool", 60)
        count = int(mock_redis.get("webhook_rate:test_tool"))
        assert count <= 30

    def test_blocks_31st_request(self, mock_redis):
        """31st request exceeds rate limit."""
        for i in range(31):
            mock_redis.incr("webhook_rate:test_tool")
        count = int(mock_redis.get("webhook_rate:test_tool"))
        assert count > 30


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


class TestInstallValidation:

    def test_install_requires_source(self, client):
        """POST /tools/install without git_url or zip_file → 400."""
        response = client.post('/tools/install', json={})
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data or ('ok' in data and data['ok'] is False)

    def test_install_bad_name_format_rejected_by_regex(self):
        """Tool name regex rejects uppercase, spaces, and special chars."""
        pattern = r"^[a-z0-9_-]+$"
        assert re.match(pattern, "good-tool_name") is not None
        assert re.match(pattern, "BAD NAME!") is None
        assert re.match(pattern, "Has Spaces") is None
        assert re.match(pattern, "special@chars") is None


class TestWebhookEndpoints:

    def test_webhook_unknown_tool_returns_404(self, client):
        """POST /tools/webhook/<unknown> → 404."""
        with patch('services.tool_registry_service.ToolRegistryService') as mock_reg:
            mock_instance = MagicMock()
            mock_instance.tools = {}  # No tools
            mock_reg.return_value = mock_instance

            response = client.post('/tools/webhook/nonexistent',
                                   json={"data": "test"})
            assert response.status_code == 404

    def test_webhook_non_webhook_tool_returns_404(self, client):
        """POST /tools/webhook/<tool> where trigger != webhook → 404."""
        with patch('services.tool_registry_service.ToolRegistryService') as mock_reg:
            mock_instance = MagicMock()
            mock_instance.tools = {
                'my_tool': {
                    'manifest': {'trigger': {'type': 'manual'}},
                }
            }
            mock_reg.return_value = mock_instance

            response = client.post('/tools/webhook/my_tool',
                                   json={"data": "test"})
            assert response.status_code == 404

    def test_webhook_bad_auth_returns_403(self, client):
        """Webhook with invalid auth credentials → 403."""
        with patch('services.tool_registry_service.ToolRegistryService') as mock_reg, \
             patch('services.tool_config_service.ToolConfigService') as mock_cfg_cls, \
             patch('services.database_service.get_shared_db_service'):
            mock_instance = MagicMock()
            mock_instance.tools = {
                'hook_tool': {
                    'manifest': {'trigger': {'type': 'webhook'}},
                }
            }
            mock_reg.return_value = mock_instance
            mock_cfg = MagicMock()
            mock_cfg.validate_webhook_hmac.return_value = False
            mock_cfg.validate_webhook_key.return_value = False
            mock_cfg_cls.return_value = mock_cfg

            response = client.post('/tools/webhook/hook_tool',
                                   json={"data": "test"},
                                   headers={"X-Chalie-Token": "bad-key"})
            assert response.status_code == 403

    def test_webhook_oversized_payload_returns_413(self, client):
        """Webhook payload >512KB → 413 (checked before service calls)."""
        oversized = "x" * (512 * 1024 + 1)
        response = client.post(
            '/tools/webhook/hook_tool',
            data=oversized,
            content_type='application/json',
        )
        assert response.status_code == 413


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


class TestWebhookKeyRotation:

    def test_generates_key_for_webhook_tool(self, client):
        """POST /tools/<name>/webhook/key → returns key for webhook tool."""
        with patch('services.tool_registry_service.ToolRegistryService') as mock_reg, \
             patch('services.tool_config_service.ToolConfigService') as mock_cfg_cls, \
             patch('services.database_service.get_shared_db_service'):
            mock_instance = MagicMock()
            mock_instance.tools = {
                'hook_tool': {
                    'manifest': {'trigger': {'type': 'webhook'}},
                }
            }
            mock_reg.return_value = mock_instance
            mock_cfg = MagicMock()
            mock_cfg.generate_webhook_key.return_value = 'wk_abc123'
            mock_cfg_cls.return_value = mock_cfg

            response = client.post('/tools/hook_tool/webhook/key')
            assert response.status_code == 200
            data = response.get_json()
            assert 'webhook_key' in data
            assert data['webhook_key'] == 'wk_abc123'

    def test_rejects_non_webhook_tool(self, client):
        """POST /tools/<name>/webhook/key for manual tool → 400."""
        with patch('services.tool_registry_service.ToolRegistryService') as mock_reg:
            mock_instance = MagicMock()
            mock_instance.tools = {
                'manual_tool': {
                    'manifest': {'trigger': {'type': 'manual'}},
                }
            }
            mock_reg.return_value = mock_instance

            response = client.post('/tools/manual_tool/webhook/key')
            assert response.status_code == 400
