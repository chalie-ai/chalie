"""
Tests for backend/api/conversation.py — conversation blueprint.

Covers /conversation/recent and /conversation/summary endpoints.

Note: The /chat endpoint was replaced by the WebSocket handler in
api/websocket.py (Phase 4). WebSocket tests live separately.
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
            mock_tcs.store.llen.return_value = 1
            mock_tcs._conv_key.return_value = "thread_conv:thread-123"
            mock_tcs.get_paginated_history.return_value = {
                "exchanges": [
                    {
                        "id": "ex-1",
                        "prompt": {"message": "hello"},
                        "response": {"message": "hi there"},
                        "topic": "greetings",
                        "timestamp": "2026-01-01T00:00:00",
                    },
                ],
                "total": 1,
                "has_more": False,
            }
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
        with patch('services.thread_service.get_thread_service') as mock_get_ts, \
             patch('services.thread_conversation_service.ThreadConversationService') as mock_tcs_cls:
            mock_ts = MagicMock()
            mock_ts.get_active_thread_id.return_value = None
            mock_get_ts.return_value = mock_ts

            mock_tcs = MagicMock()
            mock_tcs.get_most_recent_expired_thread_id.return_value = None
            mock_tcs_cls.return_value = mock_tcs

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
        with patch('services.database_service.get_shared_db_service') as mock_db_fn, \
             patch('services.episodic_retrieval_service.EpisodicRetrievalService') as mock_er_cls, \
             patch('services.config_service.ConfigService.resolve_agent_config', return_value={}):
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
