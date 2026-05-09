"""
Tests for backend/api/providers.py

Covers all endpoints in the providers blueprint:
  - GET    /providers
  - POST   /providers
  - GET    /providers/<id>
  - PUT    /providers/<id>
  - DELETE /providers/<id>
  - POST   /providers/test
  - GET    /providers/selected
  - PUT    /providers/selected
"""

import pytest
from unittest.mock import patch, MagicMock
from flask import Flask
from api.providers import providers_bp


@pytest.mark.unit
class TestProvidersAPI:
    """Test providers API endpoints."""

    @pytest.fixture
    def client(self):
        """Create Flask test client with providers blueprint."""
        app = Flask(__name__)
        app.register_blueprint(providers_bp)
        app.config['TESTING'] = True
        return app.test_client()

    @pytest.fixture(autouse=True)
    def bypass_auth(self):
        """Bypass session auth for all tests."""
        with patch('services.auth_session_service.validate_session', return_value=True):
            yield

    @pytest.fixture
    def mock_service(self):
        """Patch get_provider_service to return a shared MagicMock."""
        with patch('api.providers.get_provider_service') as mock_factory:
            svc = MagicMock()
            mock_factory.return_value = svc
            yield svc

    @pytest.fixture(autouse=True)
    def mock_cache(self):
        """Patch ProviderCacheService.invalidate so it does not error.

        autouse so individual tests do not need to take it as a parameter.
        """
        with patch(
            'services.provider_cache_service.ProviderCacheService.invalidate'
        ):
            yield

    # ------------------------------------------------------------------
    # GET /providers
    # ------------------------------------------------------------------

    def test_list_providers_returns_masked_api_key(self, client, mock_service):
        """GET /providers returns provider list with api_key masked."""
        mock_service.list_providers_summary.return_value = [
            {"id": 1, "name": "openai-main", "platform": "openai", "api_key": "sk-abc123"},
            {"id": 2, "name": "ollama-local", "platform": "ollama", "api_key": None},
        ]

        response = client.get('/providers')

        assert response.status_code == 200
        data = response.get_json()
        assert "providers" in data
        assert len(data["providers"]) == 2
        mock_service.list_providers_summary.assert_called_once()

    # ------------------------------------------------------------------
    # POST /providers
    # ------------------------------------------------------------------

    def test_create_provider_missing_required_field(self, client, mock_service):
        """POST /providers with missing required field returns 400."""
        response = client.post('/providers', json={"name": "test", "platform": "openai"})

        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
        assert "model" in data["error"]

    def test_create_provider_success(self, client, mock_service):
        """POST /providers creates provider and returns 201 with masked api_key."""
        mock_service.list_providers_summary.return_value = [
            {"id": 99, "name": "existing"}
        ]
        mock_service.create_provider.return_value = {
            "id": 5,
            "name": "anthropic-claude",
            "platform": "anthropic",
            "model": "claude-sonnet-4-20250514",
            "api_key": "sk-ant-secret",
        }

        response = client.post('/providers', json={
            "name": "anthropic-claude",
            "platform": "anthropic",
            "model": "claude-sonnet-4-20250514",
            "api_key": "sk-ant-secret",
        })

        assert response.status_code == 201
        data = response.get_json()
        assert data["provider"]["api_key"] == "***"
        assert data["provider"]["id"] == 5
        mock_service.create_provider.assert_called_once()

    # ------------------------------------------------------------------
    # GET /providers/<id>
    # ------------------------------------------------------------------

    def test_get_provider_returns_masked_key(self, client, mock_service):
        """GET /providers/<id> returns provider with masked api_key."""
        mock_service.get_provider_by_id.return_value = {
            "id": 3,
            "name": "gemini",
            "platform": "gemini",
            "model": "gemini-pro",
            "api_key": "AIza-secret-key",
        }

        response = client.get('/providers/3')

        assert response.status_code == 200
        data = response.get_json()
        assert data["provider"]["api_key"] == "***"
        assert data["provider"]["id"] == 3
        mock_service.get_provider_by_id.assert_called_once_with(3)

    def test_get_provider_not_found(self, client, mock_service):
        """GET /providers/<id> returns 404 when provider does not exist."""
        mock_service.get_provider_by_id.return_value = None

        response = client.get('/providers/999')

        assert response.status_code == 404
        data = response.get_json()
        assert "error" in data
        assert "not found" in data["error"].lower()

    # ------------------------------------------------------------------
    # PUT /providers/<id>
    # ------------------------------------------------------------------

    def test_update_provider_returns_masked_key(self, client, mock_service):
        """PUT /providers/<id> updates and returns provider with masked api_key."""
        mock_service.update_provider.return_value = {
            "id": 3,
            "name": "gemini-updated",
            "platform": "gemini",
            "model": "gemini-2.0-flash",
            "api_key": "AIza-new-secret",
        }

        response = client.put('/providers/3', json={
            "name": "gemini-updated",
            "model": "gemini-2.0-flash",
        })

        assert response.status_code == 200
        data = response.get_json()
        assert data["provider"]["api_key"] == "***"
        assert data["provider"]["name"] == "gemini-updated"
        mock_service.update_provider.assert_called_once_with(3, {
            "name": "gemini-updated",
            "model": "gemini-2.0-flash",
        })

    # ------------------------------------------------------------------
    # DELETE /providers/<id>
    # ------------------------------------------------------------------

    def test_delete_provider_success(self, client, mock_service):
        """DELETE /providers/<id> returns status deleted on success."""
        mock_service.delete_provider.return_value = None

        response = client.delete('/providers/4')

        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "deleted"
        mock_service.delete_provider.assert_called_once_with(4)

    def test_delete_provider_conflict(self, client, mock_service):
        """DELETE /providers/<id> returns 409 when service raises ValueError."""
        mock_service.delete_provider.side_effect = ValueError(
            "Cannot delete: provider is selected"
        )

        response = client.delete('/providers/4')

        assert response.status_code == 409
        data = response.get_json()
        assert "error" in data
        assert "selected" in data["error"]

    # ------------------------------------------------------------------
    # POST /providers/test  (ollama path)
    # ------------------------------------------------------------------

    def test_test_provider_ollama_success(self, client):
        """POST /providers/test with ollama platform returns success when model found."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [
                {"name": "gemma4:31b"},
                {"name": "llama3:8b"},
            ]
        }
        mock_response.raise_for_status.return_value = None

        with patch('api.providers.get_provider_service'), \
             patch('requests.get', return_value=mock_response):
            response = client.post('/providers/test', json={
                "platform": "ollama",
                "model": "gemma4:31b",
                "host": "http://localhost:11434",
            })

        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert data["model"] == "gemma4:31b"
        assert "latency_ms" in data
        assert "2 model(s) available" in data["message"]

    # ------------------------------------------------------------------
    # POST /providers/test  (API-based, missing api_key)
    # ------------------------------------------------------------------

    def test_test_provider_api_no_key(self, client):
        """POST /providers/test for API provider without api_key returns error."""
        with patch('api.providers.get_provider_service'):
            response = client.post('/providers/test', json={
                "platform": "openai",
                "model": "gpt-4o",
            })

        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is False
        assert "API key is required" in data["error"]

    # ------------------------------------------------------------------
    # GET /providers/selected
    # ------------------------------------------------------------------

    def test_get_selected_provider(self, client, mock_service):
        """GET /providers/selected returns the currently selected provider."""
        mock_service.get_selected_provider.return_value = {
            "id": 1,
            "name": "ollama-local",
            "platform": "ollama",
            "model": "gemma4:31b",
            "api_key": None,
        }

        response = client.get('/providers/selected')

        assert response.status_code == 200
        data = response.get_json()
        assert data["provider"]["id"] == 1
        assert data["provider"]["name"] == "ollama-local"
        mock_service.get_selected_provider.assert_called_once()

    def test_get_selected_provider_none(self, client, mock_service):
        """GET /providers/selected returns null when none is selected."""
        mock_service.get_selected_provider.return_value = None

        response = client.get('/providers/selected')

        assert response.status_code == 200
        data = response.get_json()
        assert data["provider"] is None

    # ------------------------------------------------------------------
    # PUT /providers/selected
    # ------------------------------------------------------------------

    def test_set_selected_provider_success(self, client, mock_service):
        """PUT /providers/selected sets the active provider."""
        mock_service.get_provider_by_id.return_value = {
            "id": 2,
            "name": "openai-main",
            "platform": "openai",
            "model": "gpt-4o",
        }

        response = client.put('/providers/selected', json={
            "provider_id": 2,
        })

        assert response.status_code == 200
        data = response.get_json()
        assert data["provider"]["id"] == 2
        mock_service.set_selected_provider.assert_called_once_with(2)

    def test_set_selected_provider_missing_id(self, client, mock_service):
        """PUT /providers/selected without provider_id returns 400."""
        response = client.put('/providers/selected', json={})

        assert response.status_code == 400
        data = response.get_json()
        assert "provider_id" in data["error"]

    def test_set_selected_provider_not_found(self, client, mock_service):
        """PUT /providers/selected with invalid provider_id returns 404."""
        mock_service.get_provider_by_id.return_value = None

        response = client.put('/providers/selected', json={
            "provider_id": 999,
        })

        assert response.status_code == 404
        data = response.get_json()
        assert "not found" in data["error"].lower()
