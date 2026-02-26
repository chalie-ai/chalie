"""
Tests for backend/api/moments.py — moments blueprint.

Covers POST /moments, GET /moments, POST /moments/<id>/forget,
and GET /moments/search endpoints.
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone
from flask import Flask

from api.moments import moments_bp


@pytest.mark.unit
class TestMomentsAPI:
    """Test moments API endpoints."""

    @pytest.fixture
    def client(self):
        """Create Flask test client with moments blueprint."""
        app = Flask(__name__)
        app.register_blueprint(moments_bp)
        app.config['TESTING'] = True
        return app.test_client()

    @pytest.fixture(autouse=True)
    def bypass_auth(self):
        """Bypass session auth for all tests."""
        with patch('services.auth_session_service.validate_session', return_value=True):
            yield

    def _mock_moment_service(self):
        """Create a MagicMock for MomentService."""
        return MagicMock()

    # ------------------------------------------------------------------
    # POST /moments
    # ------------------------------------------------------------------

    def test_create_moment_returns_201(self, client):
        """POST /moments creates moment and returns 201."""
        mock_svc = self._mock_moment_service()
        mock_svc.create_moment.return_value = {
            "id": "moment-abc",
            "message_text": "Remember this meeting",
            "pinned_at": datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            "status": "active",
        }

        with patch('api.moments._get_moment_service', return_value=mock_svc):
            response = client.post(
                '/moments',
                json={"message_text": "Remember this meeting"},
                content_type='application/json',
            )

            assert response.status_code == 201
            data = response.get_json()
            assert "item" in data
            assert data["item"]["id"] == "moment-abc"
            mock_svc.create_moment.assert_called_once()

    def test_create_moment_missing_message_text_returns_400(self, client):
        """POST /moments without message_text returns 400."""
        with patch('api.moments._get_moment_service'):
            response = client.post(
                '/moments',
                json={"topic": "misc"},
                content_type='application/json',
            )

            assert response.status_code == 400
            data = response.get_json()
            assert "error" in data
            assert "message_text" in data["error"].lower()

    def test_create_moment_text_too_long_returns_400(self, client):
        """POST /moments with message_text > 10000 chars returns 400."""
        with patch('api.moments._get_moment_service'):
            long_text = "x" * 10001
            response = client.post(
                '/moments',
                json={"message_text": long_text},
                content_type='application/json',
            )

            assert response.status_code == 400
            data = response.get_json()
            assert "error" in data
            assert "10000" in data["error"]

    def test_create_moment_duplicate_returns_200(self, client):
        """POST /moments with duplicate detection returns 200 with duplicate flag."""
        mock_svc = self._mock_moment_service()
        mock_svc.create_moment.return_value = {
            "id": "moment-existing",
            "message_text": "Already pinned",
            "duplicate": True,
            "existing_id": "moment-abc",
            "pinned_at": datetime(2026, 1, 10, 8, 0, 0, tzinfo=timezone.utc),
        }

        with patch('api.moments._get_moment_service', return_value=mock_svc):
            response = client.post(
                '/moments',
                json={"message_text": "Already pinned"},
                content_type='application/json',
            )

            assert response.status_code == 200
            data = response.get_json()
            assert data["duplicate"] is True
            assert data["existing_id"] == "moment-abc"

    def test_create_moment_non_json_returns_400(self, client):
        """POST /moments with non-JSON content type returns 400."""
        with patch('api.moments._get_moment_service'):
            response = client.post(
                '/moments',
                data="plain text",
                content_type='text/plain',
            )

            assert response.status_code == 400

    # ------------------------------------------------------------------
    # GET /moments
    # ------------------------------------------------------------------

    def test_list_moments_returns_items(self, client):
        """GET /moments returns items list."""
        mock_svc = self._mock_moment_service()
        mock_svc.get_all_moments.return_value = [
            {
                "id": "moment-1",
                "message_text": "First moment",
                "pinned_at": datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            },
            {
                "id": "moment-2",
                "message_text": "Second moment",
                "pinned_at": datetime(2026, 1, 16, 12, 0, 0, tzinfo=timezone.utc),
            },
        ]

        with patch('api.moments._get_moment_service', return_value=mock_svc):
            response = client.get('/moments')

            assert response.status_code == 200
            data = response.get_json()
            assert "items" in data
            assert len(data["items"]) == 2
            assert data["items"][0]["id"] == "moment-1"

    # ------------------------------------------------------------------
    # POST /moments/<id>/forget
    # ------------------------------------------------------------------

    def test_forget_moment_succeeds(self, client):
        """POST /moments/<id>/forget returns ok when moment exists."""
        mock_svc = self._mock_moment_service()
        mock_svc.forget_moment.return_value = True

        with patch('api.moments._get_moment_service', return_value=mock_svc):
            response = client.post('/moments/moment-abc/forget')

            assert response.status_code == 200
            data = response.get_json()
            assert data["ok"] is True
            mock_svc.forget_moment.assert_called_once_with("moment-abc")

    def test_forget_moment_not_found_returns_404(self, client):
        """POST /moments/<id>/forget returns 404 when moment does not exist."""
        mock_svc = self._mock_moment_service()
        mock_svc.forget_moment.return_value = False

        with patch('api.moments._get_moment_service', return_value=mock_svc):
            response = client.post('/moments/nonexistent-id/forget')

            assert response.status_code == 404
            data = response.get_json()
            assert "error" in data

    # ------------------------------------------------------------------
    # GET /moments/search
    # ------------------------------------------------------------------

    def test_search_moments_missing_query_returns_400(self, client):
        """GET /moments/search without q returns 400."""
        with patch('api.moments._get_moment_service'):
            response = client.get('/moments/search')

            assert response.status_code == 400
            data = response.get_json()
            assert "error" in data
            assert "q" in data["error"].lower()

    def test_search_moments_returns_items(self, client):
        """GET /moments/search with q returns items."""
        mock_svc = self._mock_moment_service()
        mock_svc.search_moments.return_value = [
            {
                "id": "moment-hit",
                "message_text": "Coffee meeting notes",
                "pinned_at": datetime(2026, 2, 1, 9, 0, 0, tzinfo=timezone.utc),
            },
        ]

        with patch('api.moments._get_moment_service', return_value=mock_svc):
            response = client.get('/moments/search?q=coffee')

            assert response.status_code == 200
            data = response.get_json()
            assert "items" in data
            assert len(data["items"]) == 1
            assert data["items"][0]["id"] == "moment-hit"
            mock_svc.search_moments.assert_called_once_with("coffee", limit=3)
