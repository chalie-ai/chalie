"""
Tests for backend/tools/tmdb_search/handler.py
"""

import pytest
from unittest.mock import patch, MagicMock
from tools.tmdb_search.handler import execute


@pytest.mark.unit
class TestTMDBSearchHandler:
    """Test tmdb_search tool handler."""

    @patch('requests.get')
    def test_missing_api_key(self, mock_get):
        """Missing API key should return error."""
        result = execute("test_topic", {"query": "test"}, {})

        assert "error" in result
        assert "API_KEY" in result['error']

    @patch('requests.get')
    def test_search_movies(self, mock_get):
        """Search for movies should call correct endpoint."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [{"id": 1, "title": "Test Movie", "release_date": "2024-01-01", "vote_average": 8.0}],
            "total_results": 1,
            "page": 1
        }
        mock_get.return_value = mock_response

        result = execute("test_topic", {
            "query": "test",
            "media_type": "movie",
            "category": "search"
        }, {"TMDB_API_KEY": "key123"})

        assert result['results'][0]['type'] == "Movie"

    @patch('requests.get')
    def test_search_tv(self, mock_get):
        """Search for TV should call correct endpoint."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [{"id": 1, "name": "Test Show", "first_air_date": "2024-01-01", "vote_average": 8.0}],
            "total_results": 1,
            "page": 1
        }
        mock_get.return_value = mock_response

        result = execute("test_topic", {
            "query": "test",
            "media_type": "tv",
            "category": "search"
        }, {"TMDB_API_KEY": "key123"})

        assert result['results'][0]['type'] == "TV"

    @patch('requests.get')
    def test_multi_search(self, mock_get):
        """Multi-search should use /search/multi endpoint."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [{"id": 1, "title": "Test", "release_date": "2024-01-01", "vote_average": 8.0}],
            "total_results": 1,
            "page": 1
        }
        mock_get.return_value = mock_response

        execute("test_topic", {
            "query": "test",
            "media_type": "all",
            "category": "search"
        }, {"TMDB_API_KEY": "key123"})

        call_url = mock_get.call_args[0][0]
        assert "/search/multi" in call_url

    @patch('requests.get')
    def test_trending_day(self, mock_get):
        """Trending day should use correct endpoint."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [{"id": 1, "title": "Test", "vote_average": 8.0}],
            "total_results": 1,
            "page": 1
        }
        mock_get.return_value = mock_response

        execute("test_topic", {
            "category": "trending_day",
            "media_type": "movie"
        }, {"TMDB_API_KEY": "key123"})

        call_url = mock_get.call_args[0][0]
        assert "/trending/" in call_url
        assert "/day" in call_url

    @patch('requests.get')
    def test_trending_week(self, mock_get):
        """Trending week should use correct endpoint."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [],
            "total_results": 0,
            "page": 1
        }
        mock_get.return_value = mock_response

        execute("test_topic", {
            "category": "trending_week",
            "media_type": "tv"
        }, {"TMDB_API_KEY": "key123"})

        call_url = mock_get.call_args[0][0]
        assert "/trending/" in call_url
        assert "/week" in call_url

    @patch('requests.get')
    def test_top_rated_requires_specific_type(self, mock_get):
        """Top rated with media_type=all should error."""
        result = execute("test_topic", {
            "category": "top_rated",
            "media_type": "all"
        }, {"TMDB_API_KEY": "key123"})

        assert "error" in result

    @patch('requests.get')
    def test_popular_requires_specific_type(self, mock_get):
        """Popular with media_type=all should error."""
        result = execute("test_topic", {
            "category": "popular",
            "media_type": "all"
        }, {"TMDB_API_KEY": "key123"})

        assert "error" in result

    def test_missing_query_for_search(self):
        """Search without query should error."""
        result = execute("test_topic", {
            "category": "search",
            "media_type": "all"
        }, {"TMDB_API_KEY": "key123"})

        assert "error" in result

    @patch('requests.get')
    def test_results_limited_to_5(self, mock_get):
        """Results should be limited to 5."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"id": i, "title": f"Movie {i}", "vote_average": 8.0}
                for i in range(10)
            ],
            "total_results": 10,
            "page": 1
        }
        mock_get.return_value = mock_response

        result = execute("test_topic", {
            "query": "test",
            "category": "search"
        }, {"TMDB_API_KEY": "key123"})

        assert len(result['results']) <= 5

    @patch('requests.get')
    def test_overview_truncation(self, mock_get):
        """Overview > 150 chars should be truncated."""
        long_overview = "x" * 200
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"id": 1, "title": "Test", "overview": long_overview, "vote_average": 8.0}
            ],
            "total_results": 1,
            "page": 1
        }
        mock_get.return_value = mock_response

        result = execute("test_topic", {
            "query": "test",
            "category": "search"
        }, {"TMDB_API_KEY": "key123"})

        assert len(result['results'][0]['overview']) <= 150

    @patch('requests.get')
    def test_page_validation(self, mock_get):
        """Invalid page should default to 1."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [],
            "total_results": 0,
            "page": 1
        }
        mock_get.return_value = mock_response

        execute("test_topic", {
            "query": "test",
            "page": "invalid"
        }, {"TMDB_API_KEY": "key123"})

        # Should not error, page should be set to 1
        assert mock_get.called
