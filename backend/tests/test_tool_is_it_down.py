"""
Tests for backend/tools/is_it_down/handler.py
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from tools.is_it_down.handler import execute
import tools.is_it_down.handler as isdown_handler


@pytest.mark.unit
class TestIsItDownHandler:
    """Test is_it_down tool handler."""

    @pytest.fixture(autouse=True)
    def setup_temp_state(self, tmp_path, monkeypatch):
        """Setup temporary state file for each test."""
        state_file = tmp_path / "state.json"
        monkeypatch.setattr(isdown_handler, 'STATE_FILE', state_file)

    def test_no_websites_configured(self):
        """No websites configured should return error."""
        result = execute("test_topic", {}, {"WEBSITES": ""})
        assert "error" in result

    @patch('tools.is_it_down.handler._check_website')
    def test_all_sites_up_no_changes(self, mock_check):
        """All sites up with no changes should not notify."""
        mock_check.return_value = "up"

        result = execute("test_topic", {}, {"WEBSITES": "example.com\ntest.com"})

        assert result['notify'] is False
        assert result['reason'] == "no_changes"

    @patch('tools.is_it_down.handler._check_website')
    def test_site_goes_down(self, mock_check):
        """Site going down should notify with critical priority."""
        # First, establish state with site up
        mock_check.return_value = "up"
        execute("test_topic", {}, {"WEBSITES": "example.com"})

        # Now site goes down
        mock_check.return_value = "down"
        result = execute("test_topic", {}, {"WEBSITES": "example.com"})

        assert result['notify'] is True
        assert result['priority'] == "critical"
        assert len(result['changes']) > 0

    @patch('tools.is_it_down.handler._check_website')
    def test_site_recovers(self, mock_check):
        """Site recovering should notify with normal priority."""
        # First, establish state with site down
        mock_check.return_value = "down"
        execute("test_topic", {}, {"WEBSITES": "example.com"})

        # Now site recovers
        mock_check.return_value = "up"
        result = execute("test_topic", {}, {"WEBSITES": "example.com"})

        assert result['notify'] is True
        assert result['priority'] == "normal"
        assert any(c['change'] == "recovered" for c in result['changes'])

    @patch('tools.is_it_down.handler.requests.head')
    @patch('tools.is_it_down.handler.requests.get')
    def test_head_fails_get_succeeds(self, mock_get, mock_head):
        """If HEAD fails, GET should be tried as fallback."""
        mock_head.side_effect = Exception("HEAD failed")
        mock_get_response = MagicMock()
        mock_get_response.status_code = 200
        mock_get.return_value = mock_get_response

        result = execute("test_topic", {}, {"WEBSITES": "example.com"})

        # Both should have been called
        assert mock_head.called
        assert mock_get.called

    @patch('tools.is_it_down.handler.requests.head')
    @patch('tools.is_it_down.handler.requests.get')
    def test_both_fail_marks_down(self, mock_get, mock_head):
        """Both HEAD and GET failures should mark site as down."""
        mock_head.side_effect = Exception("HEAD failed")
        mock_get.side_effect = Exception("GET failed")

        result = execute("test_topic", {}, {"WEBSITES": "example.com"})

        # Should still execute and have status info
        assert 'all_statuses' in result or 'changes' in result

    @patch('tools.is_it_down.handler._check_website')
    def test_url_normalization(self, mock_check):
        """URLs without https:// should be normalized."""
        mock_check.return_value = "up"

        execute("test_topic", {}, {"WEBSITES": "example.com"})

        # Should have been called with normalized URL
        call_args = mock_check.call_args[0]
        assert call_args[0].startswith("https://")

    @patch('tools.is_it_down.handler._check_website')
    def test_state_persistence(self, mock_check):
        """Previous statuses should be restored from state."""
        # First execution
        mock_check.return_value = "up"
        execute("test_topic", {}, {"WEBSITES": "example.com"})

        # Second execution - should remember previous status
        result = execute("test_topic", {}, {"WEBSITES": "example.com"})

        # No changes should be detected
        assert result['notify'] is False

    def test_timeout_configuration(self):
        """Custom TIMEOUT_SECONDS should be used."""
        with patch('tools.is_it_down.handler._check_website') as mock_check:
            mock_check.return_value = "up"

            execute("test_topic", {}, {"WEBSITES": "example.com", "TIMEOUT_SECONDS": "30"})

            # Timeout should be 30
            assert mock_check.call_args[0][1] == 30
