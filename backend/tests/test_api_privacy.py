"""
Tests for backend/api/privacy.py — privacy blueprint.

Covers /privacy/data-summary and /privacy/delete-all endpoints.
"""

import pytest
from unittest.mock import patch, MagicMock
from flask import Flask

from api.privacy import privacy_bp
from services.memory_store import MemoryStore


@pytest.mark.unit
class TestPrivacyAPI:
    """Test privacy API endpoints."""

    @pytest.fixture
    def client(self, db):
        """Create Flask test client with privacy blueprint.

        Requires the ``db`` fixture so that get_shared_db_service() returns
        the test database (patched at module level by conftest).
        """
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

    def test_data_summary_returns_counts(self, client, db):
        """GET /privacy/data-summary returns table counts."""
        store = MemoryStore()

        with patch('services.memory_client.MemoryClientService.create_connection', return_value=store):
            response = client.get('/privacy/data-summary')

            assert response.status_code == 200
            data = response.get_json()
            # Table counts should be present (all 0 from empty test DB)
            assert "episodes" in data
            assert "knowledge" in data
            assert "threads" in data
            assert "autobiography" in data
            assert "persistent_tasks" in data
            assert "place_fingerprints" in data

            # Verify counts are 0 for an empty database
            assert data["episodes"] == 0
            assert data["knowledge"] == 0

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

    def test_delete_all_with_header_clears_data(self, client, db):
        """DELETE /privacy/delete-all with header clears data and returns 200.

        Verifies that the endpoint scans all store key patterns and deletes
        matching keys, in addition to clearing all SQLite user-data tables.
        """
        store = MemoryStore()
        # Pre-populate a key that matches a pattern scanned by delete-all;
        # after the request, verifying the key is absent confirms that keys()
        # and delete() were both exercised on the real store.
        store.set("working_memory:privacy_test", "test_val")

        # Seed some data to verify it gets deleted
        db.execute(
            "INSERT INTO episodes (id, intent, context, action, emotion, outcome, gist, salience, freshness, topic) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("ep1", '{}', '{}', 'test', '{}', 'ok', 'test gist', 5, 5, 'test'),
        )
        db.commit()

        # Verify data is seeded
        row = db.execute("SELECT COUNT(*) FROM episodes").fetchone()
        assert row[0] == 1

        with patch('services.memory_client.MemoryClientService.create_connection', return_value=store), \
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

            # The pre-populated MemoryStore key should have been found and deleted
            # by the pattern scan — verifies keys() and delete() were both invoked.
            assert store.get("working_memory:privacy_test") is None

            # Verify episodes table was actually cleared
            row = db.execute("SELECT COUNT(*) FROM episodes").fetchone()
            assert row[0] == 0

    def test_delete_all_logs_audit_event(self, client, db):
        """DELETE /privacy/delete-all logs a privacy_delete_all audit event."""
        store = MemoryStore()

        with patch('services.memory_client.MemoryClientService.create_connection', return_value=store), \
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
