"""
Tests for backend/api/conversation.py — conversation blueprint.

Covers /chat (SSE), /conversation/spark-status, /conversation/recent,
and /conversation/summary endpoints.
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from flask import Flask

from api.conversation import conversation_bp


@pytest.mark.unit
class TestConversationAPI:
    """Test conversation API endpoints."""

    @pytest.fixture
    def client(self):
        """Create Flask test client with conversation blueprint."""
        app = Flask(__name__)
        app.register_blueprint(conversation_bp)
        app.config['TESTING'] = True
        return app.test_client()

    @pytest.fixture(autouse=True)
    def bypass_auth(self):
        """Bypass session auth for all tests."""
        with patch('services.auth_session_service.validate_session', return_value=True):
            yield

    # ------------------------------------------------------------------
    # POST /chat
    # ------------------------------------------------------------------

    def test_chat_returns_sse_stream_with_request_id(self, client):
        """POST /chat returns text/event-stream content type and X-Request-ID header."""
        with patch('workers.digest_worker.digest_worker'), \
             patch('services.redis_client.RedisClientService.create_connection') as mock_redis:
            mock_r = MagicMock()
            mock_pubsub = MagicMock()
            mock_pubsub.get_message.return_value = None
            mock_r.pubsub.return_value = mock_pubsub
            mock_r.get.return_value = None
            mock_redis.return_value = mock_r

            response = client.post(
                '/chat',
                json={"text": "hello"},
                content_type='application/json',
            )

            assert response.status_code == 200
            assert 'text/event-stream' in response.content_type
            assert 'X-Request-ID' in response.headers
            assert len(response.headers['X-Request-ID']) > 0

    def test_chat_missing_text_returns_400(self, client):
        """POST /chat without text field returns 400."""
        response = client.post(
            '/chat',
            json={"source": "text"},
            content_type='application/json',
        )

        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
        assert "text" in data["error"].lower()

    def test_chat_empty_text_returns_400(self, client):
        """POST /chat with empty text returns 400."""
        response = client.post(
            '/chat',
            json={"text": "   "},
            content_type='application/json',
        )

        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data

    def test_chat_non_json_content_type_returns_400(self, client):
        """POST /chat with non-JSON content type returns 400."""
        response = client.post(
            '/chat',
            data="hello",
            content_type='text/plain',
        )

        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
        assert "json" in data["error"].lower()

    # ------------------------------------------------------------------
    # GET /conversation/spark-status
    # ------------------------------------------------------------------

    def test_spark_status_returns_needs_welcome_true(self, client):
        """GET /conversation/spark-status returns needs_welcome boolean."""
        with patch('services.spark_state_service.SparkStateService') as mock_cls:
            mock_svc = MagicMock()
            mock_svc.needs_welcome.return_value = True
            mock_cls.return_value = mock_svc

            response = client.get('/conversation/spark-status')

            assert response.status_code == 200
            data = response.get_json()
            assert data["needs_welcome"] is True

    def test_spark_status_returns_false_on_error(self, client):
        """GET /conversation/spark-status returns false when service raises."""
        with patch('services.spark_state_service.SparkStateService') as mock_cls:
            mock_cls.side_effect = RuntimeError("service unavailable")

            response = client.get('/conversation/spark-status')

            assert response.status_code == 200
            data = response.get_json()
            assert data["needs_welcome"] is False

    # ------------------------------------------------------------------
    # GET /conversation/recent
    # ------------------------------------------------------------------

    def test_recent_returns_exchanges(self, client):
        """GET /conversation/recent returns exchanges array."""
        with patch('services.thread_service.get_thread_service') as mock_get_ts, \
             patch('services.thread_conversation_service.ThreadConversationService') as mock_tcs_cls:
            mock_ts = MagicMock()
            mock_ts.get_active_thread_id.return_value = "thread-123"
            mock_get_ts.return_value = mock_ts

            mock_tcs = MagicMock()
            mock_tcs.get_conversation_history.return_value = [
                {
                    "id": "ex-1",
                    "prompt": {"message": "hello"},
                    "response": {"message": "hi there"},
                    "topic": "greetings",
                    "timestamp": "2026-01-01T00:00:00",
                },
            ]
            mock_tcs_cls.return_value = mock_tcs

            response = client.get('/conversation/recent')

            assert response.status_code == 200
            data = response.get_json()
            assert "exchanges" in data
            assert data["thread_id"] == "thread-123"
            assert len(data["exchanges"]) == 1
            assert data["exchanges"][0]["prompt"] == "hello"
            assert data["exchanges"][0]["response"] == "hi there"

    def test_recent_no_thread_returns_empty(self, client):
        """GET /conversation/recent with no active thread returns empty exchanges."""
        with patch('services.thread_service.get_thread_service') as mock_get_ts:
            mock_ts = MagicMock()
            mock_ts.get_active_thread_id.return_value = None
            mock_get_ts.return_value = mock_ts

            response = client.get('/conversation/recent')

            assert response.status_code == 200
            data = response.get_json()
            assert data["thread_id"] is None
            assert data["exchanges"] == []

    # ------------------------------------------------------------------
    # GET /conversation/summary
    # ------------------------------------------------------------------

    def test_summary_returns_time_range_keys(self, client):
        """GET /conversation/summary returns today/this_week/older_highlights keys."""
        with patch('services.thread_service.get_thread_service') as mock_get_ts, \
             patch('services.gist_storage_service.GistStorageService') as mock_gist_cls, \
             patch('services.database_service.get_shared_db_service') as mock_db_fn, \
             patch('services.episodic_retrieval_service.EpisodicRetrievalService') as mock_er_cls, \
             patch('services.config_service.ConfigService.resolve_agent_config', return_value={}), \
             patch('services.redis_client.RedisClientService.create_connection') as mock_redis:
            # No active thread
            mock_ts = MagicMock()
            mock_ts.get_active_thread_id.return_value = None
            mock_get_ts.return_value = mock_ts

            # Episodic retrieval returns empty
            mock_er = MagicMock()
            mock_er.retrieve_episodes.return_value = []
            mock_er_cls.return_value = mock_er

            mock_db_fn.return_value = MagicMock()

            response = client.get('/conversation/summary')

            assert response.status_code == 200
            data = response.get_json()
            assert "today" in data
            assert "this_week" in data
            assert "older_highlights" in data
            assert isinstance(data["today"], list)
            assert isinstance(data["this_week"], list)
            assert isinstance(data["older_highlights"], list)
