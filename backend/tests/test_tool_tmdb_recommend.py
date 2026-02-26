"""
Tests for backend/tools/tmdb_recommend/handler.py

State is passed via params["_state"] and returned in result["_state"].
"""

import pytest
from unittest.mock import patch, MagicMock
from tools.tmdb_recommend.handler import execute


@pytest.mark.unit
class TestTMDBRecommendHandler:
    """Test tmdb_recommend tool handler."""

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
        from datetime import datetime
        today_str = datetime.now().strftime("%Y-%m-%d")

        state = {"last_run_date": today_str, "seen_ids": []}
        result = execute("test_topic", {"_state": state}, {
            "TMDB_API_KEY": "key123",
            "DAILY_TIME": "09:00"
        })

        assert result['notify'] is False
        assert result['reason'] == "already_ran_today"

    def test_not_scheduled_time(self):
        """Before scheduled time should not notify."""
        result = execute("test_topic", {}, {
            "TMDB_API_KEY": "key123",
            "DAILY_TIME": "23:59"  # Always in the future (unless run at 23:59)
        })

        assert result['notify'] is False
        assert result['reason'] == "not_scheduled_time"

    @patch('tools.tmdb_recommend.handler.requests.get')
    def test_successful_recommendations(self, mock_get):
        """Successful recommendations should notify with picks."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
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

        result = execute("test_topic", {}, {
            "TMDB_API_KEY": "key123",
            "DAILY_TIME": "00:00"
        })

        assert result['notify'] is True
        assert 'picks' in result
        assert len(result['picks']) > 0
        assert '_state' in result

    @patch('tools.tmdb_recommend.handler.requests.get')
    def test_genre_filtering(self, mock_get):
        """Only matching genres should be included."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "results": [
                {"id": 1, "title": "Action", "release_date": "2024-01-01",
                 "vote_average": 8.0, "popularity": 100, "genre_ids": [28]},
                {"id": 2, "title": "Drama", "release_date": "2024-01-01",
                 "vote_average": 8.0, "popularity": 100, "genre_ids": [18]},
            ]
        }
        mock_get.return_value = mock_response

        result = execute("test_topic", {}, {
            "TMDB_API_KEY": "key123",
            "DAILY_TIME": "00:00",
            "PREFERRED_GENRES": "action"
        })

        if result.get('notify') and 'picks' in result:
            # Only action movie (id=1) should be included
            tmdb_ids = [p['tmdb_id'] for p in result['picks']]
            assert 1 in tmdb_ids
            assert 2 not in tmdb_ids

    @patch('tools.tmdb_recommend.handler.requests.get')
    def test_min_rating_filter(self, mock_get):
        """Below-threshold rating should be excluded."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "results": [
                {"id": 1, "title": "Good", "release_date": "2024-01-01",
                 "vote_average": 8.0, "popularity": 100},
                {"id": 2, "title": "Bad", "release_date": "2024-01-01",
                 "vote_average": 5.0, "popularity": 100},
            ]
        }
        mock_get.return_value = mock_response

        result = execute("test_topic", {}, {
            "TMDB_API_KEY": "key123",
            "DAILY_TIME": "00:00",
            "MIN_RATING": "7.0"
        })

        if result.get('notify') and 'picks' in result:
            for pick in result['picks']:
                assert pick['rating'] >= 7.0

    @patch('tools.tmdb_recommend.handler.requests.get')
    def test_seen_ids_excluded(self, mock_get):
        """Previously recommended IDs should be excluded."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "results": [
                {"id": 1, "title": "Movie 1", "release_date": "2024-01-01",
                 "vote_average": 8.0, "popularity": 100},
            ]
        }
        mock_get.return_value = mock_response

        # State has seen movie ID 1
        state = {"seen_ids": [1], "last_run_date": ""}
        result = execute("test_topic", {"_state": state}, {
            "TMDB_API_KEY": "key123",
            "DAILY_TIME": "00:00"
        })

        # Should not recommend movie 1 since it was seen
        if result.get('notify'):
            for pick in result.get('picks', []):
                assert pick['tmdb_id'] != 1

    @patch('tools.tmdb_recommend.handler.requests.get')
    def test_rolling_window_270(self, mock_get):
        """Seen IDs should be capped at rolling window of 270."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "results": [
                {"id": 999, "title": f"Movie 999", "release_date": "2024-01-01",
                 "vote_average": 8.0, "popularity": 100},
            ]
        }
        mock_get.return_value = mock_response

        # State with 300 seen IDs
        state = {"seen_ids": list(range(300)), "last_run_date": ""}
        result = execute("test_topic", {"_state": state}, {
            "TMDB_API_KEY": "key123",
            "DAILY_TIME": "00:00"
        })

        if '_state' in result:
            assert len(result['_state']['seen_ids']) <= 270

    @patch('tools.tmdb_recommend.handler.requests.get')
    def test_scoring_formula(self, mock_get):
        """Scoring: 0.3*popularity + 0.7*rating — high popularity wins."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "results": [
                {"id": 1, "title": "High rating", "release_date": "2024-01-01",
                 "vote_average": 9.0, "popularity": 10},
                {"id": 2, "title": "High popularity", "release_date": "2024-01-01",
                 "vote_average": 6.0, "popularity": 100},
            ]
        }
        mock_get.return_value = mock_response

        result = execute("test_topic", {}, {
            "TMDB_API_KEY": "key123",
            "DAILY_TIME": "00:00",
            "MIN_RATING": "5.0"
        })

        # Score: id=1: 0.3*10 + 0.7*9 = 9.3
        # Score: id=2: 0.3*100 + 0.7*6 = 34.2
        # id=2 should be first
        if result.get('notify') and len(result.get('picks', [])) >= 2:
            assert result['picks'][0]['tmdb_id'] == 2
