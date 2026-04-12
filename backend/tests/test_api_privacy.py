"""
Tests for backend/api/privacy.py — privacy blueprint.

Covers /privacy/data-summary and /privacy/delete-all endpoints.
"""

import pytest
from unittest.mock import patch
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
            assert "transcript" in data
            assert "autobiography" in data
            assert "scheduled_items" in data
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
        """DELETE /privacy/delete-all with header clears episodes, transcript, tool_calls."""
        # Seed data
        db.execute(
            "INSERT INTO episodes (id, intent, context, action, emotion, outcome, gist, salience, channel) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("ep1", '{}', '{}', 'test', '{}', 'ok', 'test gist', 5, 'test'),
        )
        db.execute(
            "INSERT INTO transcript (channel, role, content) VALUES (?, ?, ?)",
            ("user", "user", "hello"),
        )
        db.commit()

        assert db.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM transcript").fetchone()[0] == 1

        response = client.delete(
            '/privacy/delete-all',
            headers={"X-Confirm-Delete": "yes"},
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["deleted"] is True
        assert "timestamp" in data

        assert db.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM transcript").fetchone()[0] == 0
