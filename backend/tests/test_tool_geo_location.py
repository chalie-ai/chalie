"""
Tests for backend/tools/geo_location/handler.py
"""

import pytest
from unittest.mock import patch, MagicMock
from tools.geo_location.handler import execute
import tools.geo_location.handler as geo_handler


@pytest.mark.unit
class TestGeoLocationHandler:
    """Test geolocation tool handler."""

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        """Clear module-level cache before/after each test."""
        geo_handler._cache = None
        geo_handler._cache_ts = 0.0
        yield
        geo_handler._cache = None
        geo_handler._cache_ts = 0.0

    def _make_geo_response(self, city="New York", region="NY", country="United States"):
        """Helper to create a mock geo response."""
        return {
            "status": "success",
            "city": city,
            "regionName": region,
            "country": country,
            "countryCode": "US",
            "lat": 40.7128,
            "lon": -74.0060,
            "timezone": "America/New_York",
            "isp": "Test ISP",
            "org": "Test Org",
        }

    @patch('requests.get')
    def test_successful_response(self, mock_get):
        """Successful response should return all fields."""
        mock_response = MagicMock()
        mock_response.json.return_value = self._make_geo_response()
        mock_get.return_value = mock_response

        result = execute("test_topic", {})

        assert result['city'] == "New York"
        assert result['region'] == "NY"
        assert result['country'] == "United States"
        assert result['country_code'] == "US"
        assert result['latitude'] == 40.7128
        assert result['longitude'] == -74.0060
        assert result['timezone'] == "America/New_York"
        assert result['source'] == "server_ip"
        assert 'retrieved_at' in result

    @patch('requests.get')
    def test_cache_hit_within_ttl(self, mock_get):
        """Second call within TTL should use cache (no HTTP)."""
        mock_response = MagicMock()
        mock_response.json.return_value = self._make_geo_response()
        mock_get.return_value = mock_response

        # First call
        result1 = execute("test_topic", {})

        # Second call - should use cache
        result2 = execute("test_topic", {})

        # Should only call once (cache hit on second)
        assert mock_get.call_count == 1
        assert result1['city'] == result2['city']

    @patch('requests.get')
    def test_api_failure_returns_stale_cache(self, mock_get):
        """API failure should return stale cache if available."""
        # Populate cache first
        mock_response = MagicMock()
        mock_response.json.return_value = self._make_geo_response(city="New York")
        mock_get.return_value = mock_response
        result1 = execute("test_topic", {})
        assert result1['city'] == "New York"

        # Now API fails
        mock_get.side_effect = Exception("Network error")
        result2 = execute("test_topic", {})

        # Should return stale cache
        assert result2['city'] == "New York"

    @patch('requests.get')
    def test_api_failure_no_cache(self, mock_get):
        """API failure without cache should return error dict."""
        mock_get.side_effect = Exception("Network error")
        result = execute("test_topic", {})

        assert "error" in result

    @patch('requests.get')
    def test_api_returns_failure_status(self, mock_get):
        """API response with status != 'success' should raise error."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "fail",
            "message": "invalid query"
        }
        mock_get.return_value = mock_response

        result = execute("test_topic", {})

        assert "error" in result
