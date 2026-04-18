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
    def client(self, db):
        """Create Flask test client with memory blueprint.

        Requires the ``db`` fixture so that get_shared_db_service() returns
        the test database (patched at module level by conftest).
        """
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
        """GET /memory/search with q returns results array from episodic + data graph."""
        mock_dgs = MagicMock()
        mock_dgs.recall.return_value = [
            {"id": 1, "key": "coffee", "value": "a beverage",
             "retrieval_weight": 0.8, "evidence_count": 1, "composite_score": 0.8},
        ]

        # api/memory.py calls episodic_retrieval_service.retrieve() directly —
        # not EpisodicService.retrieve_episodes().  Patch the module-level function.
        with patch('services.episodic_retrieval_service.retrieve') as mock_retrieve, \
             patch('services.data_graph_service.get_data_graph_service', return_value=mock_dgs):

            mock_retrieve.return_value = [
                {"gist": "user likes coffee", "composite_score": 0.9, "created_at": "2026-01-01"},
            ]

            response = client.get('/memory/search?q=coffee')

            assert response.status_code == 200
            data = response.get_json()
            assert "results" in data
            assert len(data["results"]) == 2
            # Results sorted by score descending
            assert data["results"][0]["score"] >= data["results"][1]["score"]
