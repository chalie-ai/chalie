"""
Tests for App Update API endpoints.

Verifies that the update check and install endpoints are accessible
and return expected responses.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestAppUpdateAPI:
    """Test suite for /api/v1/update/ endpoints."""

    # ────────────────────────────────────────────────
    # GET /api/v1/update/check
    # ────────────────────────────────────────────────

    def test_get_update_check_returns_ok_and_version(self, client):
        """GET /api/v1/update/check returns status and version info."""
        with patch('consumer.APP_VERSION', '0.2.0'):
            resp = client.get('/api/v1/update/check')
        
        assert resp.status_code == 200
        data = resp.get_json()
        assert 'current_version' in data
        assert 'latest_version' in data
        assert 'update_available' in data
        assert 'deployment_mode' in data

    def test_get_update_check_dev_mode_skips_github(self, client):
        """In dev mode, update check returns current version without GitHub call."""
        with patch('os.environ.get') as mock_env:
            mock_env.return_value = None  # Simulate dev mode (no IS_DOCKER or APP_HOME)
            
            resp = client.get('/api/v1/update/check')
        
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['update_available'] is False
        assert data['deployment_mode'] == 'dev'

    def test_get_update_check_docker_mode_skips_github(self, client):
        """In docker mode, update check returns current version without GitHub call."""
        with patch('os.environ.get') as mock_env:
            def env_side_effect(key, default=None):
                if key == 'IS_DOCKER':
                    return 'true'
                return None
            
            mock_env.side_effect = env_side_effect
            
            resp = client.get('/api/v1/update/check')
        
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['update_available'] is False
        assert data['deployment_mode'] == 'docker'

    # ────────────────────────────────────────────────
    # POST /api/v1/update/install
    # ────────────────────────────────────────────────

    def test_post_update_install_no_update_available(self, client):
        """POST /api/v1/update/install returns no_update when current version is latest."""
        with patch('services.app_update_service.AppUpdateService.check_for_updates') as mock_check:
            mock_check.return_value = {
                'update_available': False,
                'latest_version': '0.2.0',
                'current_version': '0.2.0'
            }
            
            resp = client.post('/api/v1/update/install')
        
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['status'] == 'no_update'

    def test_post_update_install_triggers_background_thread(self, client):
        """POST /api/v1/update/install returns 202 when update is available."""
        with patch('services.app_update_service.AppUpdateService.check_for_updates') as mock_check:
            mock_check.return_value = {
                'update_available': True,
                'latest_version': '0.3.0',
                'current_version': '0.2.0'
            }
            
            resp = client.post('/api/v1/update/install')
        
        assert resp.status_code == 202
        data = resp.get_json()
        assert data['status'] == 'accepted'
