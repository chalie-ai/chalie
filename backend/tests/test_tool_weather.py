"""
Tests for backend/tools/weather/handler.py
"""

import pytest
from unittest.mock import patch, MagicMock
from tools.weather.handler import execute
import tools.weather.handler as weather_handler


@pytest.mark.unit
class TestWeatherHandler:
    """Test weather tool handler."""

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        """Clear module-level cache before/after each test."""
        weather_handler._cache.clear()
        yield
        weather_handler._cache.clear()

    def _make_weather_response(self, location="London", condition="Sunny", temp_c=20,
                              feels_like_c=18, humidity=65, wind_kmh=10):
        """Helper to create a mock weather response."""
        return {
            "current_condition": [{
                "temp_C": str(temp_c),
                "temp_F": str(temp_c * 9/5 + 32),
                "FeelsLikeC": str(feels_like_c),
                "humidity": str(humidity),
                "windspeedKmph": str(wind_kmh),
                "winddir16Point": "N",
                "visibility": "10",
                "uvIndex": "5",
                "precipMM": "0",
                "localObsDateTime": "2024-01-15 10:00 AM",
                "weatherDesc": [{"value": condition}]
            }],
            "nearest_area": [{
                "areaName": [{"value": location}],
                "country": [{"value": "UK"}]
            }]
        }

    @patch('requests.get')
    def test_successful_weather_response(self, mock_get):
        """Successful response should return all fields."""
        mock_response = MagicMock()
        mock_response.json.return_value = self._make_weather_response()
        mock_get.return_value = mock_response

        result = execute("test_topic", {"location": "London"})

        assert result['location'] == "London, UK"
        assert result['condition'] == "Sunny"
        assert result['temperature_c'] == 20
        assert 'is_raining' in result
        assert 'is_daylight' in result

    @patch('requests.get')
    def test_cache_hit_within_ttl(self, mock_get):
        """Second call within TTL should use cache (no HTTP)."""
        mock_response = MagicMock()
        mock_response.json.return_value = self._make_weather_response()
        mock_get.return_value = mock_response

        # First call
        result1 = execute("test_topic", {"location": "London"})

        # Second call - should use cache
        result2 = execute("test_topic", {"location": "London"})

        # Should only call once (cache hit on second)
        assert mock_get.call_count == 1
        assert result1['condition'] == result2['condition']

    @patch('requests.get')
    @patch('tools.weather.handler.time.time')
    def test_cache_miss_after_ttl(self, mock_time, mock_get):
        """Expired cache should trigger fresh request."""
        mock_response = MagicMock()
        mock_response.json.return_value = self._make_weather_response()
        mock_get.return_value = mock_response

        # First call at t=0
        mock_time.return_value = 0
        execute("test_topic", {"location": "London"})

        # Second call at t=700 (> 600s TTL)
        mock_time.return_value = 700
        execute("test_topic", {"location": "London"})

        # Should call twice (cache expired)
        assert mock_get.call_count == 2

    @patch('requests.get')
    def test_api_failure_returns_stale_cache(self, mock_get):
        """API failure should return stale cache if available."""
        # Populate cache first
        mock_response = MagicMock()
        mock_response.json.return_value = self._make_weather_response(condition="Sunny")
        mock_get.return_value = mock_response
        result1 = execute("test_topic", {"location": "London"})
        assert result1['condition'] == "Sunny"

        # Now API fails
        mock_get.side_effect = Exception("Network error")
        result2 = execute("test_topic", {"location": "London"})

        # Should return stale cache
        assert result2['condition'] == "Sunny"

    @patch('requests.get')
    def test_api_failure_no_cache_returns_error(self, mock_get):
        """API failure without cache should return error."""
        mock_get.side_effect = Exception("Network error")
        result = execute("test_topic", {"location": "London"})

        assert "error" in result
        assert "unavailable" in result['error'].lower()

    @patch('requests.get')
    def test_is_raining_true(self, mock_get):
        """Condition with 'rain' should set is_raining=True."""
        mock_response = MagicMock()
        mock_response.json.return_value = self._make_weather_response(condition="Rain")
        mock_get.return_value = mock_response

        result = execute("test_topic", {"location": "London"})
        assert result['is_raining'] is True

    @patch('requests.get')
    def test_is_clear_true(self, mock_get):
        """Condition with 'sunny' should set is_clear=True."""
        mock_response = MagicMock()
        mock_response.json.return_value = self._make_weather_response(condition="Sunny")
        mock_get.return_value = mock_response

        result = execute("test_topic", {"location": "London"})
        assert result['is_clear'] is True

    @patch('requests.get')
    def test_is_hot_true(self, mock_get):
        """feels_like >= 30 should set is_hot=True."""
        mock_response = MagicMock()
        mock_response.json.return_value = self._make_weather_response(feels_like_c=32)
        mock_get.return_value = mock_response

        result = execute("test_topic", {"location": "London"})
        assert result['is_hot'] is True

    @patch('requests.get')
    def test_is_cold_true(self, mock_get):
        """feels_like <= 10 should set is_cold=True."""
        mock_response = MagicMock()
        mock_response.json.return_value = self._make_weather_response(feels_like_c=5)
        mock_get.return_value = mock_response

        result = execute("test_topic", {"location": "London"})
        assert result['is_cold'] is True

    @patch('requests.get')
    def test_is_windy_true(self, mock_get):
        """wind >= 30 kmh should set is_windy=True."""
        mock_response = MagicMock()
        mock_response.json.return_value = self._make_weather_response(wind_kmh=35)
        mock_get.return_value = mock_response

        result = execute("test_topic", {"location": "London"})
        assert result['is_windy'] is True

    @patch('requests.get')
    def test_invalid_json_response(self, mock_get):
        """Missing current_condition should return error."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"nearest_area": []}
        mock_get.return_value = mock_response

        result = execute("test_topic", {"location": "London"})
        assert "error" in result
        assert "unavailable" in result['error'].lower()

    @patch('requests.get')
    def test_estimate_daylight_daytime(self, mock_get):
        """Daylight between 6-20 should be True."""
        mock_response = MagicMock()
        mock_response.json.return_value = self._make_weather_response()
        mock_get.return_value = mock_response

        result = execute("test_topic", {"location": "London"})
        # Default obs time is "2024-01-15 10:00 AM" which is 10 hours - daylight
        assert result['is_daylight'] is True

    @patch('requests.get')
    def test_estimate_daylight_nighttime(self, mock_get):
        """Daylight outside 6-20 should be False."""
        mock_response = MagicMock()
        # 11:00 PM is 23:00
        response_data = self._make_weather_response()
        response_data['current_condition'][0]['localObsDateTime'] = "2024-01-15 11:00 PM"
        mock_response.json.return_value = response_data
        mock_get.return_value = mock_response

        result = execute("test_topic", {"location": "London"})
        assert result['is_daylight'] is False
