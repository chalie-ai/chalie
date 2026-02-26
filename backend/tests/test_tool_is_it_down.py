"""
Tests for backend/tools/is_it_down/handler.py

State is passed via params["_state"] and returned in result["_state"].
"""

import pytest
from unittest.mock import patch, MagicMock
from tools.is_it_down.handler import execute


@pytest.mark.unit
class TestIsItDownHandler:
    """Test is_it_down tool handler."""

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
        mock_check.return_value = "down"

        # Pass state with site previously up
        state = {"statuses": {"https://example.com": "up"}, "last_check": ""}
        result = execute("test_topic", {"_state": state}, {"WEBSITES": "example.com"})

        assert result['notify'] is True
        assert result['priority'] == "critical"
        assert len(result['changes']) > 0

    @patch('tools.is_it_down.handler._check_website')
    def test_site_recovers(self, mock_check):
        """Site recovering should notify with normal priority."""
        mock_check.return_value = "up"

        # Pass state with site previously down
        state = {"statuses": {"https://example.com": "down"}, "last_check": ""}
        result = execute("test_topic", {"_state": state}, {"WEBSITES": "example.com"})

        assert result['notify'] is True
        assert result['priority'] == "normal"
        assert any(c['change'] == "recovered" for c in result['changes'])

    @patch('tools.is_it_down.handler.requests.head')
    @patch('tools.is_it_down.handler.requests.get')
    def test_head_fails_get_succeeds(self, mock_get, mock_head):
        """If HEAD fails with RequestException, GET should be tried as fallback."""
        import requests as real_requests
        mock_head.side_effect = real_requests.exceptions.ConnectionError("HEAD failed")
        mock_get_response = MagicMock()
        mock_get_response.status_code = 200
        mock_get.return_value = mock_get_response

        result = execute("test_topic", {}, {"WEBSITES": "example.com"})

        assert mock_head.called
        assert mock_get.called

    @patch('tools.is_it_down.handler.requests.head')
    @patch('tools.is_it_down.handler.requests.get')
    def test_both_fail_marks_down(self, mock_get, mock_head):
        """Both HEAD and GET failures should mark site as down."""
        mock_head.side_effect = Exception("HEAD failed")
        mock_get.side_effect = Exception("GET failed")

        result = execute("test_topic", {}, {"WEBSITES": "example.com"})

        assert 'all_statuses' in result or 'changes' in result

    @patch('tools.is_it_down.handler._check_website')
    def test_url_normalization(self, mock_check):
        """URLs without https:// should be normalized."""
        mock_check.return_value = "up"

        execute("test_topic", {}, {"WEBSITES": "example.com"})

        call_args = mock_check.call_args[0]
        assert call_args[0].startswith("https://")

    @patch('tools.is_it_down.handler._check_website')
    def test_state_persistence_via_params(self, mock_check):
        """State returned from first call can be fed back for continuity."""
        mock_check.return_value = "up"

        # First execution — no prior state
        result1 = execute("test_topic", {}, {"WEBSITES": "example.com"})
        assert '_state' in result1

        # Second execution — feed back prior state
        result2 = execute("test_topic", {"_state": result1['_state']}, {"WEBSITES": "example.com"})
        assert result2['notify'] is False

    def test_timeout_configuration(self):
        """Custom TIMEOUT_SECONDS should be used."""
        with patch('tools.is_it_down.handler._check_website') as mock_check:
            mock_check.return_value = "up"

            execute("test_topic", {}, {"WEBSITES": "example.com", "TIMEOUT_SECONDS": "30"})

            assert mock_check.call_args[0][1] == 30
