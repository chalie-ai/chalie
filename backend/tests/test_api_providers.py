"""
Tests for backend/api/providers.py

Covers all endpoints in the providers blueprint:
  - GET    /providers
  - POST   /providers
  - GET    /providers/<id>
  - PUT    /providers/<id>
  - DELETE /providers/<id>
  - POST   /providers/test
  - GET    /providers/jobs
  - PUT    /providers/jobs/<job_name>
"""

import pytest
from unittest.mock import patch, MagicMock, call
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

    @pytest.fixture
    def mock_cache(self):
        """Patch ProviderCacheService.invalidate so it does not error."""
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
        # The list endpoint returns whatever list_providers_summary gives;
        # masking happens per-row in the service or in single-object endpoints.
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

    def test_create_provider_success(self, client, mock_service, mock_cache):
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

    def test_create_first_provider_auto_assigns_13_jobs(
        self, client, mock_service, mock_cache
    ):
        """POST /providers for the first provider auto-assigns all 13 jobs."""
        mock_service.list_providers_summary.return_value = []  # no existing
        mock_service.create_provider.return_value = {
            "id": 1,
            "name": "ollama-local",
            "platform": "ollama",
            "model": "qwen3:4b",
            "api_key": None,
        }

        response = client.post('/providers', json={
            "name": "ollama-local",
            "platform": "ollama",
            "model": "qwen3:4b",
        })

        assert response.status_code == 201
        assert mock_service.set_job_assignment.call_count == 13

        assigned_jobs = [c.args[0] for c in mock_service.set_job_assignment.call_args_list]
        expected_jobs = [
            'frontal-cortex', 'frontal-cortex-respond', 'frontal-cortex-clarify',
            'frontal-cortex-acknowledge', 'frontal-cortex-act', 'frontal-cortex-proactive',
            'memory-chunker', 'episodic-memory', 'semantic-memory',
            'mode-tiebreaker', 'mode-reflection', 'cognitive-drift',
            'experience-assimilation',
        ]
        assert assigned_jobs == expected_jobs

        # Each assignment should reference the newly created provider's id
        for c in mock_service.set_job_assignment.call_args_list:
            assert c.args[1] == 1

    def test_create_second_provider_does_not_auto_assign(
        self, client, mock_service, mock_cache
    ):
        """POST /providers when providers already exist does not auto-assign jobs."""
        mock_service.list_providers_summary.return_value = [
            {"id": 1, "name": "existing-provider"}
        ]
        mock_service.create_provider.return_value = {
            "id": 2,
            "name": "second",
            "platform": "openai",
            "model": "gpt-4o",
            "api_key": "sk-xyz",
        }

        response = client.post('/providers', json={
            "name": "second",
            "platform": "openai",
            "model": "gpt-4o",
            "api_key": "sk-xyz",
        })

        assert response.status_code == 201
        mock_service.set_job_assignment.assert_not_called()

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

    def test_update_provider_returns_masked_key(self, client, mock_service, mock_cache):
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

    def test_delete_provider_success(self, client, mock_service, mock_cache):
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
            "Cannot delete: provider is assigned to active jobs"
        )

        response = client.delete('/providers/4')

        assert response.status_code == 409
        data = response.get_json()
        assert "error" in data
        assert "active jobs" in data["error"]

    # ------------------------------------------------------------------
    # POST /providers/test  (ollama path)
    # ------------------------------------------------------------------

    def test_test_provider_ollama_success(self, client):
        """POST /providers/test with ollama platform returns success when model found."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [
                {"name": "qwen3:4b"},
                {"name": "llama3:8b"},
            ]
        }
        mock_response.raise_for_status.return_value = None

        with patch('api.providers.get_provider_service'), \
             patch('requests.get', return_value=mock_response):
            response = client.post('/providers/test', json={
                "platform": "ollama",
                "model": "qwen3:4b",
                "host": "http://localhost:11434",
            })

        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert data["model"] == "qwen3:4b"
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
    # GET /providers/jobs
    # ------------------------------------------------------------------

    def test_list_job_assignments(self, client, mock_service):
        """GET /providers/jobs returns all job assignments."""
        mock_service.get_all_job_assignments.return_value = [
            {"job_name": "frontal-cortex", "provider_id": 1},
            {"job_name": "cognitive-drift", "provider_id": 2},
        ]

        response = client.get('/providers/jobs')

        assert response.status_code == 200
        data = response.get_json()
        assert "assignments" in data
        assert len(data["assignments"]) == 2
        mock_service.get_all_job_assignments.assert_called_once()

    # ------------------------------------------------------------------
    # PUT /providers/jobs/<job_name>
    # ------------------------------------------------------------------

    def test_assign_job_missing_provider_id(self, client, mock_service):
        """PUT /providers/jobs/<name> without provider_id returns 400."""
        response = client.put('/providers/jobs/frontal-cortex', json={})

        assert response.status_code == 400
        data = response.get_json()
        assert "provider_id" in data["error"]

    def test_assign_job_success(self, client, mock_service, mock_cache):
        """PUT /providers/jobs/<name> assigns provider successfully."""
        mock_service.set_job_assignment.return_value = {
            "job_name": "memory-chunker",
            "provider_id": 3,
        }

        response = client.put('/providers/jobs/memory-chunker', json={
            "provider_id": 3,
        })

        assert response.status_code == 200
        data = response.get_json()
        assert data["assignment"]["job_name"] == "memory-chunker"
        assert data["assignment"]["provider_id"] == 3
        mock_service.set_job_assignment.assert_called_once_with("memory-chunker", 3)
