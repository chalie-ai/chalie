"""
Tests for backend/api/memory.py — memory blueprint.

Covers /memory/context, /memory/forget, and /memory/search endpoints.
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
    # GET /memory/context
    # ------------------------------------------------------------------

    def test_context_returns_expected_keys(self, client):
        """GET /memory/context returns traits/facts/significant_episodes/concepts keys."""
        with patch('services.database_service.get_shared_db_service') as mock_db_fn, \
             patch('services.user_trait_service.UserTraitService') as mock_trait_cls, \
             patch('services.thread_service.get_thread_service') as mock_ts_fn, \
             patch('services.episodic_retrieval_service.EpisodicRetrievalService') as mock_er_cls, \
             patch('services.semantic_retrieval_service.SemanticRetrievalService') as mock_sr_cls, \
             patch('services.config_service.ConfigService.resolve_agent_config', return_value={}):
            mock_db_fn.return_value = MagicMock()

            mock_trait = MagicMock()
            mock_trait.get_traits_for_prompt.return_value = ""
            mock_trait_cls.return_value = mock_trait

            mock_ts = MagicMock()
            mock_ts.get_active_thread_id.return_value = None
            mock_ts_fn.return_value = mock_ts

            mock_er = MagicMock()
            mock_er.retrieve_episodes.return_value = []
            mock_er_cls.return_value = mock_er

            mock_sr = MagicMock()
            mock_sr.retrieve_concepts.return_value = []
            mock_sr_cls.return_value = mock_sr

            response = client.get('/memory/context')

            assert response.status_code == 200
            data = response.get_json()
            assert "traits" in data
            assert "facts" in data
            assert "significant_episodes" in data
            assert "concepts" in data

    # ------------------------------------------------------------------
    # POST /memory/forget — scope=topic
    # ------------------------------------------------------------------

    def test_forget_topic_missing_topic_returns_400(self, client):
        """POST /memory/forget scope=topic without topic field returns 400."""
        response = client.post(
            '/memory/forget',
            json={"scope": "topic"},
            content_type='application/json',
        )

        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
        assert "topic" in data["error"].lower()

    def test_forget_topic_clears_memory_stores(self, client):
        """POST /memory/forget scope=topic clears gists, facts, and working memory."""
        with patch('services.gist_storage_service.GistStorageService') as mock_gist_cls, \
             patch('services.fact_store_service.FactStoreService') as mock_fact_cls, \
             patch('services.working_memory_service.WorkingMemoryService') as mock_wm_cls:
            mock_gist = MagicMock()
            mock_gist_cls.return_value = mock_gist

            mock_fact = MagicMock()
            mock_fact_cls.return_value = mock_fact

            mock_wm = MagicMock()
            mock_wm_cls.return_value = mock_wm

            response = client.post(
                '/memory/forget',
                json={"scope": "topic", "topic": "test-topic"},
                content_type='application/json',
            )

            assert response.status_code == 200
            data = response.get_json()
            assert data["deleted"] is True
            assert data["scope"] == "topic"
            assert data["topic"] == "test-topic"

            mock_gist.clear_gists.assert_called_once_with("test-topic")
            mock_fact.clear_facts.assert_called_once_with("test-topic")
            mock_wm.clear.assert_called_once_with("test-topic")

    # ------------------------------------------------------------------
    # POST /memory/forget — scope=fact
    # ------------------------------------------------------------------

    def test_forget_fact_missing_fact_key_returns_400(self, client):
        """POST /memory/forget scope=fact without fact_key returns 400."""
        response = client.post(
            '/memory/forget',
            json={"scope": "fact", "topic": "some-topic"},
            content_type='application/json',
        )

        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
        assert "fact_key" in data["error"].lower()

    def test_forget_fact_missing_topic_returns_400(self, client):
        """POST /memory/forget scope=fact without topic returns 400."""
        response = client.post(
            '/memory/forget',
            json={"scope": "fact", "fact_key": "some-key"},
            content_type='application/json',
        )

        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data

    # ------------------------------------------------------------------
    # POST /memory/forget — scope=all
    # ------------------------------------------------------------------

    def test_forget_all_without_confirm_header_returns_400(self, client):
        """POST /memory/forget scope=all without X-Confirm-Delete header returns 400."""
        response = client.post(
            '/memory/forget',
            json={"scope": "all"},
            content_type='application/json',
        )

        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
        assert "X-Confirm-Delete" in data["error"]

    def test_forget_all_with_header_clears_data(self, client):
        """POST /memory/forget scope=all with header clears Redis and PostgreSQL."""
        with patch('services.redis_client.RedisClientService.create_connection') as mock_redis_fn, \
             patch('services.database_service.get_shared_db_service') as mock_db_fn, \
             patch('services.interaction_log_service.InteractionLogService') as mock_log_cls:
            mock_redis = MagicMock()
            mock_redis.keys.return_value = ["key1", "key2"]
            mock_redis_fn.return_value = mock_redis

            mock_conn = MagicMock()
            mock_conn_ctx = MagicMock()
            mock_conn_ctx.__enter__ = MagicMock(return_value=mock_conn)
            mock_conn_ctx.__exit__ = MagicMock(return_value=False)
            mock_db = MagicMock()
            mock_db.connection.return_value = mock_conn_ctx
            mock_db_fn.return_value = mock_db

            mock_log = MagicMock()
            mock_log_cls.return_value = mock_log

            response = client.post(
                '/memory/forget',
                json={"scope": "all"},
                headers={"X-Confirm-Delete": "yes"},
                content_type='application/json',
            )

            assert response.status_code == 200
            data = response.get_json()
            assert data["deleted"] is True
            assert data["scope"] == "all"

            # Redis keys should have been queried and deleted
            assert mock_redis.keys.call_count > 0
            assert mock_redis.delete.call_count > 0

    # ------------------------------------------------------------------
    # POST /memory/forget — invalid scope
    # ------------------------------------------------------------------

    def test_forget_invalid_scope_returns_400(self, client):
        """POST /memory/forget with invalid scope returns 400."""
        response = client.post(
            '/memory/forget',
            json={"scope": "invalid"},
            content_type='application/json',
        )

        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
        assert "scope" in data["error"].lower()

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
             patch('services.episodic_retrieval_service.EpisodicRetrievalService') as mock_er_cls, \
             patch('services.semantic_retrieval_service.SemanticRetrievalService') as mock_sr_cls, \
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
