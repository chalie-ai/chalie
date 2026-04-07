"""
Tests for backend/api/conversation.py — conversation blueprint.

Covers /conversation/recent endpoint.

Note: The /chat endpoint was replaced by the WebSocket handler in
api/websocket.py (Phase 4). WebSocket tests live separately.
"""

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
        mock_store = MagicMock()
        mock_store.get.return_value = 'web:default:1'
        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor
        # total count = 2 (one user + one assistant)
        mock_cursor.fetchone.return_value = (2,)
        mock_cursor.fetchall.return_value = [
            (1, 'user', 'hello', '2026-01-01T00:00:00'),
            (2, 'assistant', 'hi there', '2026-01-01T00:00:01'),
        ]
        mock_db.connection.return_value = mock_conn

        with patch('services.memory_client.MemoryClientService.create_connection', return_value=mock_store), \
             patch('services.database_service.get_shared_db_service', return_value=mock_db):
            response = client.get('/conversation/recent')

        assert response.status_code == 200
        data = response.get_json()
        assert "exchanges" in data
        assert data["thread_id"] == 'web:default:1'

    def test_recent_no_entries_returns_empty(self, client):
        """GET /conversation/recent with no transcript entries returns empty."""
        mock_store = MagicMock()
        mock_store.get.return_value = None
        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = (0,)
        mock_db.connection.return_value = mock_conn

        with patch('services.memory_client.MemoryClientService.create_connection', return_value=mock_store), \
             patch('services.database_service.get_shared_db_service', return_value=mock_db):
            response = client.get('/conversation/recent')

        assert response.status_code == 200
        data = response.get_json()
        assert data["exchanges"] == []
        assert data["total"] == 0
