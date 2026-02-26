"""
Tests for backend/tools/duckduckgo_search/handler.py
"""

import pytest
from unittest.mock import patch, MagicMock
from tools.duckduckgo_search.handler import execute


@pytest.mark.unit
class TestDuckDuckGoSearchHandler:
    """Test duckduckgo_search tool handler."""

    def test_empty_query_returns_empty(self):
        """Empty query should return empty results without searching."""
        result = execute("test_topic", {"query": ""})

        assert result['results'] == []
        assert result['count'] == 0

    @patch('tools.duckduckgo_search.handler._search_ddg')
    def test_successful_search(self, mock_search):
        """Successful search should return formatted results."""
        mock_search.return_value = [
            {"title": "Test Title", "snippet": "Test snippet", "url": "http://example.com"}
        ]

        result = execute("test_topic", {"query": "test search", "limit": 5})

        assert result['count'] == 1
        assert result['results'][0]['title'] == "Test Title"
        assert result['results'][0]['url'] == "http://example.com"

    @patch('tools.duckduckgo_search.handler._search_ddg')
    def test_snippet_truncation(self, mock_search):
        """Long snippets should be truncated to 200 chars."""
        long_snippet = "x" * 250
        mock_search.return_value = [
            {"title": "Test", "snippet": long_snippet, "url": "http://example.com"}
        ]

        result = execute("test_topic", {"query": "test", "limit": 5})

        assert len(result['results'][0]['snippet']) <= 200
        assert result['results'][0]['snippet'].endswith("...")

    @patch('tools.duckduckgo_search.handler._search_ddg')
    def test_search_error_returns_empty_with_error(self, mock_search):
        """Failed search should return empty results with error field."""
        mock_search.side_effect = Exception("Search error")

        result = execute("test_topic", {"query": "test search"})

        assert result['results'] == []
        assert result['count'] == 0
        assert 'error' in result

    @patch('tools.duckduckgo_search.handler._search_ddg')
    def test_whitespace_query_returns_empty(self, mock_search):
        """Whitespace-only query should return empty results."""
        result = execute("test_topic", {"query": "   "})

        assert result['results'] == []
        assert result['count'] == 0
        mock_search.assert_not_called()

    @patch('tools.duckduckgo_search.handler._search_ddg')
    def test_limit_clamped_min(self, mock_search):
        """Limit below 1 should be clamped to 1."""
        mock_search.return_value = []

        execute("test_topic", {"query": "test", "limit": 0})

        assert mock_search.call_args[0][1] == 1

    @patch('tools.duckduckgo_search.handler._search_ddg')
    def test_limit_clamped_max(self, mock_search):
        """Limit above 10 should be clamped to 10."""
        mock_search.return_value = []

        execute("test_topic", {"query": "test", "limit": 99})

        assert mock_search.call_args[0][1] == 10

    @patch('tools.duckduckgo_search.handler._search_ddg')
    def test_multiple_results_preserved(self, mock_search):
        """Multiple results should be returned with count."""
        mock_search.return_value = [
            {"title": f"Result {i}", "snippet": f"Snippet {i}", "url": f"http://example.com/{i}"}
            for i in range(3)
        ]

        result = execute("test_topic", {"query": "test", "limit": 5})

        assert result['count'] == 3
        assert len(result['results']) == 3

    @patch('tools.duckduckgo_search.handler._search_ddg')
    def test_config_timeout_used(self, mock_search):
        """Custom timeout from config should be passed to search."""
        mock_search.return_value = []

        execute("test_topic", {"query": "test"}, config={"DUCKDUCKGO_TIMEOUT": "15"})

        assert mock_search.call_args[0][2] == 15

    def test_missing_query_key_returns_empty(self):
        """Missing query key should return empty results."""
        result = execute("test_topic", {})

        assert result['results'] == []
        assert result['count'] == 0
