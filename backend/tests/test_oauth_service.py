"""
Unit tests for OAuthService — generic OAuth2 flow.
"""
import json
import time
from unittest.mock import patch, MagicMock

import pytest

from services.oauth_service import OAuthService, RESERVED_OAUTH_KEYS


@pytest.mark.unit
class TestOAuthServiceAuthUrl:
    """Test auth URL generation."""

    def test_generates_auth_url_with_pkce(self, mock_redis):
        svc = OAuthService()

        manifest_auth = {
            "type": "oauth2",
            "authorization_url": "https://accounts.google.com/o/oauth2/v2/auth",
            "token_url": "https://oauth2.googleapis.com/token",
            "scopes": ["https://www.googleapis.com/auth/gmail.readonly"],
            "pkce": True,
            "extra_auth_params": {"access_type": "offline", "prompt": "consent"},
        }

        with patch.object(svc, '_get_config_value', return_value='test-client-id'):
            result = svc.get_auth_url("test_tool", manifest_auth, "http://localhost/callback")

        assert "auth_url" in result
        assert "state" in result
        assert "accounts.google.com" in result["auth_url"]
        assert "client_id=test-client-id" in result["auth_url"]
        assert "code_challenge=" in result["auth_url"]
        assert "code_challenge_method=S256" in result["auth_url"]
        assert "access_type=offline" in result["auth_url"]
        assert "prompt=consent" in result["auth_url"]

    def test_generates_auth_url_without_pkce(self, mock_redis):
        svc = OAuthService()

        manifest_auth = {
            "type": "oauth2",
            "authorization_url": "https://example.com/auth",
            "token_url": "https://example.com/token",
            "scopes": ["read"],
            "pkce": False,
        }

        with patch.object(svc, '_get_config_value', return_value='client-123'):
            result = svc.get_auth_url("test_tool", manifest_auth, "http://localhost/cb")

        assert "auth_url" in result
        assert "code_challenge" not in result["auth_url"]

    def test_raises_without_client_id(self, mock_redis):
        svc = OAuthService()
        manifest_auth = {
            "type": "oauth2",
            "authorization_url": "https://example.com/auth",
            "token_url": "https://example.com/token",
            "scopes": [],
        }

        with patch.object(svc, '_get_config_value', return_value=None):
            with pytest.raises(ValueError, match="client_id"):
                svc.get_auth_url("test_tool", manifest_auth, "http://localhost/cb")

    def test_stores_state_in_redis(self, mock_redis):
        svc = OAuthService()
        manifest_auth = {
            "type": "oauth2",
            "authorization_url": "https://example.com/auth",
            "token_url": "https://example.com/token",
            "scopes": ["email"],
            "pkce": True,
        }

        with patch.object(svc, '_get_config_value', return_value='client-id'):
            result = svc.get_auth_url("my_tool", manifest_auth, "http://localhost/cb")

        state = result["state"]
        stored = mock_redis.get(f"oauth_state:{state}")
        assert stored is not None
        data = json.loads(stored)
        assert data["tool_name"] == "my_tool"
        assert data["code_verifier"] is not None


@pytest.mark.unit
class TestOAuthServiceExchangeCode:
    """Test code exchange."""

    def test_exchanges_code_and_stores_tokens(self, mock_redis, mock_db):
        svc = OAuthService()

        # Pre-seed state in Redis
        state = "test-state-token"
        mock_redis.setex(f"oauth_state:{state}", 300, json.dumps({
            "tool_name": "test_tool",
            "code_verifier": "test-verifier",
            "redirect_uri": "http://localhost/callback",
        }))

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "ya29.fresh-token",
            "refresh_token": "1//refresh-token",
            "expires_in": 3600,
            "scope": "email profile",
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(svc, '_get_manifest_auth', return_value={
            "type": "oauth2",
            "token_url": "https://example.com/token",
            "scopes": ["email"],
        }), \
        patch.object(svc, '_get_config_value', side_effect=lambda t, k: {
            "client_id": "cid", "client_secret": "csec"
        }.get(k)), \
        patch('requests.post', return_value=mock_response), \
        patch.object(svc, '_store_tokens') as mock_store:
            result = svc.exchange_code(state, "auth-code-123")

        assert result["tool_name"] == "test_tool"
        assert result["connected"] is True
        mock_store.assert_called_once()

    def test_invalid_state_raises(self, mock_redis):
        svc = OAuthService()
        with pytest.raises(ValueError, match="Invalid or expired"):
            svc.exchange_code("bogus-state", "code")

    def test_state_consumed_after_use(self, mock_redis):
        svc = OAuthService()
        state = "consume-test"
        mock_redis.setex(f"oauth_state:{state}", 300, json.dumps({
            "tool_name": "t", "code_verifier": None, "redirect_uri": "",
        }))

        # Consume the state
        data = svc._pop_oauth_state(state)
        assert data is not None

        # Second read should return None
        data2 = svc._pop_oauth_state(state)
        assert data2 is None


@pytest.mark.unit
class TestOAuthServiceRefresh:
    """Test token refresh."""

    def test_returns_token_if_not_expired(self):
        svc = OAuthService()
        future = str(int(time.time()) + 3600)

        with patch.object(svc, '_get_all_config', return_value={
            "_oauth_access_token": "valid-token",
            "_oauth_refresh_token": "refresh",
            "_oauth_token_expires_at": future,
        }):
            result = svc.refresh_if_needed("tool", {"token_url": "https://ex.com/token"})

        assert result == "valid-token"

    def test_refreshes_expired_token(self, mock_db):
        svc = OAuthService()
        expired = str(int(time.time()) - 100)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "new-token",
            "expires_in": 3600,
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(svc, '_get_all_config', return_value={
            "_oauth_access_token": "old-token",
            "_oauth_refresh_token": "refresh-token",
            "_oauth_token_expires_at": expired,
        }), \
        patch.object(svc, '_get_config_value', side_effect=lambda t, k: {
            "client_id": "cid", "client_secret": "csec"
        }.get(k)), \
        patch('requests.post', return_value=mock_response), \
        patch.object(svc, '_store_tokens'):
            result = svc.refresh_if_needed("tool", {
                "token_url": "https://example.com/token",
                "scopes": [],
            })

        assert result == "new-token"

    def test_returns_none_if_no_token(self):
        svc = OAuthService()
        with patch.object(svc, '_get_all_config', return_value={}):
            result = svc.refresh_if_needed("tool", {})
        assert result is None


@pytest.mark.unit
class TestOAuthServiceStatus:
    """Test status and disconnect."""

    def test_connected_status(self):
        svc = OAuthService()
        with patch.object(svc, '_get_all_config', return_value={
            "_oauth_access_token": "token",
            "_oauth_connected_at": "1700000000",
            "_oauth_scopes": "email profile",
        }):
            status = svc.get_oauth_status("tool")

        assert status["connected"] is True
        assert "email" in status["scopes"]
        assert "profile" in status["scopes"]

    def test_disconnected_status(self):
        svc = OAuthService()
        with patch.object(svc, '_get_all_config', return_value={}):
            status = svc.get_oauth_status("tool")

        assert status["connected"] is False
        assert status["scopes"] == []

    def test_disconnect(self, mock_db):
        svc = OAuthService()
        mock_config_svc = MagicMock()

        with patch('services.database_service.get_shared_db_service', return_value=mock_db), \
             patch('services.tool_config_service.ToolConfigService', return_value=mock_config_svc):
            result = svc.disconnect("tool")

        assert result is True
        # Should have called delete for each oauth key
        assert mock_config_svc.delete_tool_config_key.call_count == len(RESERVED_OAUTH_KEYS)


@pytest.mark.unit
class TestReservedOAuthKeys:
    """Verify RESERVED_OAUTH_KEYS matches tool_config_service."""

    def test_keys_match_config_service(self):
        from services.tool_config_service import ToolConfigService
        for key in RESERVED_OAUTH_KEYS:
            assert key in ToolConfigService.RESERVED_KEYS, f"{key} not in ToolConfigService.RESERVED_KEYS"
