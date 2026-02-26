"""
Tests for backend/api/privacy.py — privacy blueprint.

Covers /privacy/data-summary and /privacy/delete-all endpoints.
"""

import pytest
from unittest.mock import patch, MagicMock, call
from flask import Flask

from api.privacy import privacy_bp


@pytest.mark.unit
class TestPrivacyAPI:
    """Test privacy API endpoints."""

    @pytest.fixture
    def client(self):
        """Create Flask test client with privacy blueprint."""
        app = Flask(__name__)
        app.register_blueprint(privacy_bp)
        app.config['TESTING'] = True
        return app.test_client()

    @pytest.fixture(autouse=True)
    def bypass_auth(self):
        """Bypass session auth for all tests."""
        with patch('services.auth_session_service.validate_session', return_value=True):
            yield

    # ------------------------------------------------------------------
    # GET /privacy/data-summary
    # ------------------------------------------------------------------

    def test_data_summary_returns_counts(self, client):
        """GET /privacy/data-summary returns table counts and fact count."""
        mock_conn = MagicMock()

        # Each table COUNT(*) returns (5,); timestamps query returns row with dates
        table_row = MagicMock()
        table_row.__getitem__ = lambda self, idx: 5
        timestamp_row = MagicMock()
        timestamp_row.__getitem__ = lambda self, idx: "2026-01-01" if idx == 0 else "2026-02-01"
        timestamp_row.__bool__ = lambda self: True

        # conn.execute() is called for each table + timestamps
        # 4 tables + 1 timestamp query = 5 execute calls
        table_cursor = MagicMock()
        table_cursor.fetchone.return_value = table_row
        mock_conn.execute.return_value = table_cursor

        mock_conn_ctx = MagicMock()
        mock_conn_ctx.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn_ctx.__exit__ = MagicMock(return_value=False)

        mock_db = MagicMock()
        mock_db.connection.return_value = mock_conn_ctx

        mock_redis = MagicMock()
        mock_redis.keys.return_value = ["fact_index:topic1", "fact_index:topic2"]

        with patch('services.database_service.get_shared_db_service', return_value=mock_db), \
             patch('services.redis_client.RedisClientService.create_connection', return_value=mock_redis):
            response = client.get('/privacy/data-summary')

            assert response.status_code == 200
            data = response.get_json()
            # Table counts should be present
            assert "episodes" in data
            assert "semantic_concepts" in data
            assert "user_traits" in data
            assert "threads" in data
            # Redis fact count
            assert data["facts"] == 2

    # ------------------------------------------------------------------
    # DELETE /privacy/delete-all
    # ------------------------------------------------------------------

    def test_delete_all_without_confirm_header_returns_400(self, client):
        """DELETE /privacy/delete-all without X-Confirm-Delete returns 400."""
        response = client.delete('/privacy/delete-all')

        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
        assert "X-Confirm-Delete" in data["error"]

    def test_delete_all_with_header_clears_data(self, client):
        """DELETE /privacy/delete-all with header clears data and returns 200."""
        mock_redis = MagicMock()
        mock_redis.keys.return_value = ["key1", "key2"]

        mock_conn = MagicMock()
        mock_conn_ctx = MagicMock()
        mock_conn_ctx.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn_ctx.__exit__ = MagicMock(return_value=False)

        mock_db = MagicMock()
        mock_db.connection.return_value = mock_conn_ctx

        with patch('services.redis_client.RedisClientService.create_connection', return_value=mock_redis), \
             patch('services.database_service.get_shared_db_service', return_value=mock_db), \
             patch('services.interaction_log_service.InteractionLogService') as mock_log_cls:
            mock_log = MagicMock()
            mock_log_cls.return_value = mock_log

            response = client.delete(
                '/privacy/delete-all',
                headers={"X-Confirm-Delete": "yes"},
            )

            assert response.status_code == 200
            data = response.get_json()
            assert data["deleted"] is True
            assert "timestamp" in data

            # Redis patterns were scanned and deleted
            assert mock_redis.keys.call_count > 0
            assert mock_redis.delete.call_count > 0

            # PostgreSQL tables were truncated
            assert mock_conn.execute.call_count > 0
            mock_conn.commit.assert_called_once()

    def test_delete_all_logs_audit_event(self, client):
        """DELETE /privacy/delete-all logs a privacy_delete_all audit event."""
        mock_redis = MagicMock()
        mock_redis.keys.return_value = []

        mock_conn = MagicMock()
        mock_conn_ctx = MagicMock()
        mock_conn_ctx.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn_ctx.__exit__ = MagicMock(return_value=False)

        mock_db = MagicMock()
        mock_db.connection.return_value = mock_conn_ctx

        with patch('services.redis_client.RedisClientService.create_connection', return_value=mock_redis), \
             patch('services.database_service.get_shared_db_service', return_value=mock_db), \
             patch('services.interaction_log_service.InteractionLogService') as mock_log_cls:
            mock_log = MagicMock()
            mock_log_cls.return_value = mock_log

            response = client.delete(
                '/privacy/delete-all',
                headers={"X-Confirm-Delete": "yes"},
            )

            assert response.status_code == 200

            # Verify audit event was logged
            mock_log.log_event.assert_called_once()
            call_args = mock_log.log_event.call_args
            # event_type passed as keyword arg
            assert call_args.kwargs.get("event_type") == "privacy_delete_all"
