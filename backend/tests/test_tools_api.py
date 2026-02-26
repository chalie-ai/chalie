"""
Tests for backend/api/tools.py
"""

import pytest
from unittest.mock import patch, MagicMock
from flask import Flask
from api.tools import tools_bp


@pytest.mark.unit
class TestToolsAPI:
    """Test tools API endpoints."""

    @pytest.fixture
    def client(self):
        """Create Flask test client."""
        app = Flask(__name__)
        app.register_blueprint(tools_bp)
        app.config['TESTING'] = True
        return app.test_client()

    @pytest.fixture(autouse=True)
    def bypass_auth(self):
        """Bypass session auth for all tests."""
        with patch('services.auth_session_service.validate_session', return_value=True):
            yield

    def test_install_no_source(self, client):
        """Install without source should return 400."""
        response = client.post('/tools/install', json={})

        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data or "ok" in data
