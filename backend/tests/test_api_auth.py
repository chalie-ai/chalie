"""
Tests for backend/api/user_auth.py — authentication blueprint.

Covers /auth/status, /auth/register, /auth/login, and /auth/logout.
These tests do NOT bypass auth — they test the auth system itself.
"""

import pytest
from unittest.mock import patch, MagicMock
from flask import Flask
from werkzeug.security import generate_password_hash

from api.user_auth import user_auth_bp


@pytest.mark.unit
class TestAuthAPI:
    """Test user authentication API endpoints."""

    @pytest.fixture
    def client(self):
        """Create Flask test client with user_auth blueprint."""
        app = Flask(__name__)
        app.secret_key = 'test-secret-key'
        app.register_blueprint(user_auth_bp)
        app.config['TESTING'] = True
        return app.test_client()

    def _make_db_mock(self):
        """Build a mock DatabaseService with get_session() context manager wired up.

        Returns (mock_db, mock_session, mock_result) so tests can program
        session.execute().fetchone() etc.
        """
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_session.execute.return_value = mock_result

        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_session)
        mock_ctx.__exit__ = MagicMock(return_value=False)

        mock_db = MagicMock()
        mock_db.get_session.return_value = mock_ctx

        return mock_db, mock_session, mock_result

    # ------------------------------------------------------------------
    # GET /auth/status
    # ------------------------------------------------------------------

    def test_status_returns_expected_keys(self, client):
        """GET /auth/status returns has_master_account, has_providers, has_session."""
        mock_db, mock_session, mock_result = self._make_db_mock()

        # First call: account count = 1, second call: provider count = 0
        mock_result.fetchone.side_effect = [(1,), (0,)]

        with patch('services.database_service.get_shared_db_service', return_value=mock_db), \
             patch('services.auth_session_service.validate_session', return_value=False):
            response = client.get('/auth/status')

            assert response.status_code == 200
            data = response.get_json()
            assert data["has_master_account"] is True
            assert data["has_providers"] is False
            assert data["has_session"] is False

    def test_status_with_valid_session(self, client):
        """GET /auth/status with valid session returns has_session true."""
        mock_db, mock_session, mock_result = self._make_db_mock()
        mock_result.fetchone.side_effect = [(1,), (1,)]

        with patch('services.database_service.get_shared_db_service', return_value=mock_db), \
             patch('services.auth_session_service.validate_session', return_value=True):
            response = client.get('/auth/status')

            assert response.status_code == 200
            data = response.get_json()
            assert data["has_master_account"] is True
            assert data["has_providers"] is True
            assert data["has_session"] is True

    # ------------------------------------------------------------------
    # POST /auth/register
    # ------------------------------------------------------------------

    def test_register_short_password_returns_400(self, client):
        """POST /auth/register with short password returns 400."""
        response = client.post(
            '/auth/register',
            json={"username": "admin", "password": "short"},
            content_type='application/json',
        )

        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
        assert "8 characters" in data["error"].lower() or "password" in data["error"].lower()

    def test_register_missing_username_returns_400(self, client):
        """POST /auth/register with missing username returns 400."""
        response = client.post(
            '/auth/register',
            json={"password": "securepassword123"},
            content_type='application/json',
        )

        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
        assert "username" in data["error"].lower()

    def test_register_success_returns_201(self, client):
        """POST /auth/register with valid data creates account and returns 201."""
        mock_db, mock_session, mock_result = self._make_db_mock()

        # Existing count = 0 (no master account yet)
        mock_result.fetchone.return_value = (0,)

        with patch('services.database_service.get_shared_db_service', return_value=mock_db), \
             patch('services.auth_session_service.create_session') as mock_create_session:
            response = client.post(
                '/auth/register',
                json={"username": "admin", "password": "securepassword123"},
                content_type='application/json',
            )

            assert response.status_code == 201
            data = response.get_json()
            assert data["ok"] is True

            # Account was inserted via session.execute
            assert mock_session.execute.call_count >= 2  # SELECT COUNT + INSERT
            mock_session.commit.assert_called_once()
            mock_create_session.assert_called_once()

    def test_register_duplicate_returns_409(self, client):
        """POST /auth/register when account already exists returns 409."""
        mock_db, mock_session, mock_result = self._make_db_mock()

        # Existing count = 1 (account already exists)
        mock_result.fetchone.return_value = (1,)

        with patch('services.database_service.get_shared_db_service', return_value=mock_db):
            response = client.post(
                '/auth/register',
                json={"username": "admin", "password": "securepassword123"},
                content_type='application/json',
            )

            assert response.status_code == 409
            data = response.get_json()
            assert "error" in data
            assert "already exists" in data["error"].lower()

    # ------------------------------------------------------------------
    # POST /auth/login
    # ------------------------------------------------------------------

    def test_login_missing_credentials_returns_400(self, client):
        """POST /auth/login with missing credentials returns 400."""
        response = client.post(
            '/auth/login',
            json={"username": "admin"},
            content_type='application/json',
        )

        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data

    def test_login_invalid_credentials_returns_401(self, client):
        """POST /auth/login with invalid credentials returns 401."""
        mock_db, mock_session, mock_result = self._make_db_mock()

        # No matching user found
        mock_result.fetchone.return_value = None

        with patch('services.database_service.get_shared_db_service', return_value=mock_db):
            response = client.post(
                '/auth/login',
                json={"username": "admin", "password": "wrongpassword"},
                content_type='application/json',
            )

            assert response.status_code == 401
            data = response.get_json()
            assert "error" in data
            assert "invalid" in data["error"].lower()

    def test_login_success_returns_200(self, client):
        """POST /auth/login with valid credentials returns 200 and sets session."""
        mock_db, mock_session, mock_result = self._make_db_mock()

        # Return a matching password hash
        test_password = "securepassword123"
        stored_hash = generate_password_hash(test_password)
        mock_result.fetchone.return_value = (stored_hash,)

        with patch('services.database_service.get_shared_db_service', return_value=mock_db), \
             patch('services.auth_session_service.create_session') as mock_create_session:
            response = client.post(
                '/auth/login',
                json={"username": "admin", "password": test_password},
                content_type='application/json',
            )

            assert response.status_code == 200
            data = response.get_json()
            assert data["ok"] is True
            mock_create_session.assert_called_once()

    # ------------------------------------------------------------------
    # POST /auth/logout
    # ------------------------------------------------------------------

    def test_logout_returns_ok(self, client):
        """POST /auth/logout returns ok and destroys session."""
        with patch('services.auth_session_service.destroy_session') as mock_destroy:
            response = client.post('/auth/logout')

            assert response.status_code == 200
            data = response.get_json()
            assert data["ok"] is True
            mock_destroy.assert_called_once()
