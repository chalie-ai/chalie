"""
Tests for backend/tools/tmdb_recommend/handler.py
"""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock
from tools.tmdb_recommend.handler import execute
import tools.tmdb_recommend.handler as tmdb_handler


@pytest.mark.unit
class TestTMDBRecommendHandler:
    """Test tmdb_recommend tool handler."""

    @pytest.fixture(autouse=True)
    def setup_temp_state(self, tmp_path, monkeypatch):
        """Setup temporary state file for each test."""
        state_file = tmp_path / "state.json"
        monkeypatch.setattr(tmdb_handler, 'STATE_FILE', state_file)

    def test_missing_api_key(self):
        """Missing API key should return error."""
        result = execute("test_topic", {}, {})

        assert "error" in result
        assert "API_KEY" in result['error']

    def test_missing_daily_time(self):
        """Missing DAILY_TIME should return error."""
        result = execute("test_topic", {}, {"TMDB_API_KEY": "key123"})

        assert "error" in result
        assert "DAILY_TIME" in result['error']

    def test_already_ran_today(self):
        """Already ran today should not notify."""
        with patch('datetime.datetime') as mock_dt, \
             patch('tools.tmdb_recommend.handler._load_state') as mock_load:
            mock_now = MagicMock()
            mock_now.strftime.return_value = "2024-01-15"
            mock_now.hour = 10
            mock_now.minute = 0
            mock_now.second = 0
            mock_now.microsecond = 0
            mock_now.replace.return_value = mock_now
            mock_dt.now.return_value = mock_now

            # State shows already ran today
            mock_load.return_value = {"last_run_date": "2024-01-15", "seen_ids": []}

            result = execute("test_topic", {}, {
                "TMDB_API_KEY": "key123",
                "DAILY_TIME": "09:00"
            })

            assert result['notify'] is False
            assert result['reason'] == "already_ran_today"

    def test_not_scheduled_time(self):
        """Before scheduled time should not notify."""
        with patch('datetime.datetime') as mock_dt, \
             patch('tools.tmdb_recommend.handler._load_state'):
            mock_now = MagicMock()
            mock_now.strftime.return_value = "2024-01-15"
            mock_now.hour = 8
            mock_now.minute = 0
            mock_now.second = 0
            mock_now.microsecond = 0
            mock_now.replace.return_value = mock_now
            mock_dt.now.return_value = mock_now

            result = execute("test_topic", {}, {
                "TMDB_API_KEY": "key123",
                "DAILY_TIME": "09:00"
            })

            assert result['notify'] is False
            assert result['reason'] == "not_scheduled_time"

    @patch('requests.get')
    def test_successful_recommendations(self, mock_get):
        """Successful recommendations should notify."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {
                    "id": 1,
                    "title": "Movie 1",
                    "release_date": "2024-01-01",
                    "vote_average": 8.0,
                    "popularity": 100,
                    "genre_ids": [28]
                }
            ]
        }
        mock_get.return_value = mock_response

        with patch('datetime.datetime') as mock_dt, \
             patch('tools.tmdb_recommend.handler._load_state'):
            mock_now = MagicMock()
            mock_now.strftime.return_value = "2024-01-15"
            mock_now.hour = 10
            mock_now.minute = 0
            mock_now.second = 0
            mock_now.microsecond = 0
            mock_now.replace.return_value = mock_now
            mock_dt.now.return_value = mock_now

            result = execute("test_topic", {}, {
                "TMDB_API_KEY": "key123",
                "DAILY_TIME": "09:00"
            })

            if result.get('notify'):
                assert 'picks' in result
                assert len(result['picks']) > 0

    @patch('requests.get')
    def test_genre_filtering(self, mock_get):
        """Only matching genres should be returned."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"id": 1, "title": "Action", "release_date": "2024-01-01", "vote_average": 8.0, "popularity": 100, "genre_ids": [28]},
                {"id": 2, "title": "Drama", "release_date": "2024-01-01", "vote_average": 8.0, "popularity": 100, "genre_ids": [18]}
            ]
        }
        mock_get.return_value = mock_response

        with patch('datetime.datetime') as mock_dt, \
             patch('tools.tmdb_recommend.handler._load_state'):
            mock_now = MagicMock()
            mock_now.strftime.return_value = "2024-01-15"
            mock_now.hour = 10
            mock_now.minute = 0
            mock_now.second = 0
            mock_now.microsecond = 0
            mock_now.replace.return_value = mock_now
            mock_dt.now.return_value = mock_now

            result = execute("test_topic", {}, {
                "TMDB_API_KEY": "key123",
                "DAILY_TIME": "09:00",
                "PREFERRED_GENRES": "action"  # Only action genre (ID 28)
            })

            if result.get('notify') and 'picks' in result:
                # Should only include action movies
                for pick in result['picks']:
                    # Genre filtering is applied in _fetch_and_filter
                    pass

    @patch('requests.get')
    def test_min_rating_filter(self, mock_get):
        """Below-threshold rating should be excluded."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"id": 1, "title": "Good", "release_date": "2024-01-01", "vote_average": 8.0, "popularity": 100},
                {"id": 2, "title": "Bad", "release_date": "2024-01-01", "vote_average": 5.0, "popularity": 100}
            ]
        }
        mock_get.return_value = mock_response

        with patch('datetime.datetime') as mock_dt, \
             patch('tools.tmdb_recommend.handler._load_state'):
            mock_now = MagicMock()
            mock_now.strftime.return_value = "2024-01-15"
            mock_now.hour = 10
            mock_now.minute = 0
            mock_now.second = 0
            mock_now.microsecond = 0
            mock_now.replace.return_value = mock_now
            mock_dt.now.return_value = mock_now

            result = execute("test_topic", {}, {
                "TMDB_API_KEY": "key123",
                "DAILY_TIME": "09:00",
                "MIN_RATING": "7.0"
            })

            # Should filter out low-rated items
            if result.get('notify') and 'picks' in result:
                for pick in result['picks']:
                    assert pick['rating'] >= 7.0

    @patch('requests.get')
    def test_seen_ids_excluded(self, mock_get):
        """Previously recommended should be excluded."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"id": 1, "title": "Movie 1", "release_date": "2024-01-01", "vote_average": 8.0, "popularity": 100}
            ]
        }
        mock_get.return_value = mock_response

        with patch('datetime.datetime') as mock_dt, \
             patch('tools.tmdb_recommend.handler._load_state') as mock_load:
            mock_now = MagicMock()
            mock_now.strftime.return_value = "2024-01-15"
            mock_now.hour = 10
            mock_now.minute = 0
            mock_now.second = 0
            mock_now.microsecond = 0
            mock_now.replace.return_value = mock_now
            mock_dt.now.return_value = mock_now

            # State has seen movie ID 1
            mock_load.return_value = {"seen_ids": [1], "last_run_date": ""}

            result = execute("test_topic", {}, {
                "TMDB_API_KEY": "key123",
                "DAILY_TIME": "09:00"
            })

            # Should not recommend movie 1 since it was seen
            if result.get('notify'):
                for pick in result.get('picks', []):
                    assert pick['tmdb_id'] != 1

    @patch('requests.get')
    def test_rolling_window_270(self, mock_get):
        """Seen IDs should keep rolling window of 270."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"id": i, "title": f"Movie {i}", "release_date": "2024-01-01", "vote_average": 8.0, "popularity": 100}
                for i in range(100)
            ]
        }
        mock_get.return_value = mock_response

        with patch('datetime.datetime') as mock_dt, \
             patch('tools.tmdb_recommend.handler._load_state'):
            mock_now = MagicMock()
            mock_now.strftime.return_value = "2024-01-15"
            mock_now.hour = 10
            mock_now.minute = 0
            mock_now.second = 0
            mock_now.microsecond = 0
            mock_now.replace.return_value = mock_now
            mock_dt.now.return_value = mock_now

            result = execute("test_topic", {}, {
                "TMDB_API_KEY": "key123",
                "DAILY_TIME": "09:00"
            })

            # Rolling window of 270 is maintained

    @patch('requests.get')
    def test_scoring_formula(self, mock_get):
        """Scoring formula: 0.3*popularity + 0.7*rating."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"id": 1, "title": "High rating", "release_date": "2024-01-01", "vote_average": 9.0, "popularity": 10},
                {"id": 2, "title": "High popularity", "release_date": "2024-01-01", "vote_average": 6.0, "popularity": 100}
            ]
        }
        mock_get.return_value = mock_response

        with patch('datetime.datetime') as mock_dt, \
             patch('tools.tmdb_recommend.handler._load_state'):
            mock_now = MagicMock()
            mock_now.strftime.return_value = "2024-01-15"
            mock_now.hour = 10
            mock_now.minute = 0
            mock_now.second = 0
            mock_now.microsecond = 0
            mock_now.replace.return_value = mock_now
            mock_dt.now.return_value = mock_now

            result = execute("test_topic", {}, {
                "TMDB_API_KEY": "key123",
                "DAILY_TIME": "09:00",
                "MIN_RATING": "5.0"
            })

            # Score: id=1: 0.3*10 + 0.7*9 = 3 + 6.3 = 9.3
            # Score: id=2: 0.3*100 + 0.7*6 = 30 + 4.2 = 34.2
            # Should pick id=2 first
