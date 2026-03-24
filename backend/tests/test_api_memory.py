"""
Tests for backend/api/memory.py — memory blueprint.

Covers /memory/search endpoint.
"""

import pytest
from unittest.mock import patch, MagicMock
from flask import Flask

from api.memory import memory_bp


@pytest.mark.unit
class TestMemoryAPI:
    """Test memory API endpoints."""

    @pytest.fixture
    def client(self):
        """Create Flask test client with memory blueprint."""
        app = Flask(__name__)
        app.register_blueprint(memory_bp)
        app.config['TESTING'] = True
        return app.test_client()

    @pytest.fixture(autouse=True)
    def bypass_auth(self):
        """Bypass session auth for all tests."""
        with patch('services.auth_session_service.validate_session', return_value=True):
            yield

    # ------------------------------------------------------------------
    # GET /memory/search
    # ------------------------------------------------------------------

    def test_search_missing_query_returns_400(self, client):
        """GET /memory/search without q param returns 400."""
        response = client.get('/memory/search')

        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
        assert "q" in data["error"].lower()

    def test_search_returns_results(self, client):
        """GET /memory/search with q returns results array."""
        with patch('services.database_service.get_shared_db_service') as mock_db_fn, \
             patch('services.episodic_service.EpisodicService') as mock_er_cls, \
             patch('services.semantic_service.SemanticService') as mock_sr_cls, \
             patch('services.config_service.ConfigService.resolve_agent_config', return_value={}):
            mock_db_fn.return_value = MagicMock()

            mock_er = MagicMock()
            mock_er.retrieve_episodes.return_value = [
                {"gist": "user likes coffee", "composite_score": 0.9, "created_at": "2026-01-01"},
            ]
            mock_er_cls.return_value = mock_er

            mock_sr = MagicMock()
            mock_sr.retrieve_concepts.return_value = [
                {"name": "coffee", "definition": "a beverage", "score": 0.8, "strength": 5},
            ]
            mock_sr_cls.return_value = mock_sr

            response = client.get('/memory/search?q=coffee')

            assert response.status_code == 200
            data = response.get_json()
            assert "results" in data
            assert len(data["results"]) == 2
            # Results sorted by score descending
            assert data["results"][0]["score"] >= data["results"][1]["score"]
