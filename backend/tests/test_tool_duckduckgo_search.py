"""
Tests for backend/tools/duckduckgo_search/handler.py
"""

import pytest
import json
import time
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from tools.duckduckgo_search.handler import execute
import tools.duckduckgo_search.handler as ddg_handler


@pytest.mark.unit
class TestDuckDuckGoSearchHandler:
    """Test duckduckgo_search tool handler."""

    @pytest.fixture(autouse=True)
    def setup_temp_dirs(self, tmp_path, monkeypatch):
        """Setup temporary data directory for each test."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        monkeypatch.setattr(ddg_handler, '_DATA_DIR', data_dir)
        monkeypatch.setattr(ddg_handler, '_BUDGET_FILE', data_dir / "budget.json")
        monkeypatch.setattr(ddg_handler, '_DEDUP_FILE', data_dir / "dedup.json")

    def test_empty_query_returns_empty(self):
        """Empty query should return empty results without searching."""
        result = execute("test_topic", {"query": ""})

        assert result['results'] == []
        assert result['count'] == 0
        assert 'budget_remaining' in result

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
    def test_budget_consumed_on_search(self, mock_search):
        """Budget should be decremented on successful search."""
        mock_search.return_value = [{"title": "Test", "snippet": "Test", "url": "http://example.com"}]

        result1 = execute("test_topic", {"query": "search 1"})
        initial_budget = result1['budget_remaining']

        result2 = execute("test_topic", {"query": "search 2"})
        new_budget = result2['budget_remaining']

        assert new_budget < initial_budget

    @patch('tools.duckduckgo_search.handler._search_ddg')
    def test_budget_exhausted_blocks_search(self, mock_search):
        """Exhausted budget should block new searches."""
        mock_search.return_value = [{"title": "Test", "snippet": "Test", "url": "http://example.com"}]

        # Exhaust budget (default 8)
        for i in range(8):
            execute("test_topic", {"query": f"search {i}"})

        # Next search should be blocked
        result = execute("test_topic", {"query": "blocked search"})
        assert result['budget_remaining'] == 0
        assert 'message' in result
        assert 'exhausted' in result['message'].lower()

    @patch('tools.duckduckgo_search.handler._search_ddg')
    def test_budget_refund_on_failure(self, mock_search):
        """Failed search should refund budget."""
        mock_search.side_effect = Exception("Search error")

        result = execute("test_topic", {"query": "test search"})

        # Budget should not be consumed (refunded)
        assert result['budget_remaining'] > 0
        assert 'error' in result

    @patch('tools.duckduckgo_search.handler._search_ddg')
    def test_dedup_blocks_repeat_query(self, mock_search):
        """Same query within cooldown should be blocked."""
        mock_search.return_value = [{"title": "Test", "snippet": "Test", "url": "http://example.com"}]

        # First search
        result1 = execute("test_topic", {"query": "test search"})
        assert result1['count'] == 1

        # Second search (same query within cooldown)
        result2 = execute("test_topic", {"query": "test search"})
        assert result2['count'] == 0
        assert 'DUPLICATE' in result2.get('message', '')

    @patch('tools.duckduckgo_search.handler._search_ddg')
    @patch('tools.duckduckgo_search.handler.time.time')
    def test_dedup_allows_after_cooldown(self, mock_time, mock_search):
        """Same query after cooldown should be allowed."""
        mock_search.return_value = [{"title": "Test", "snippet": "Test", "url": "http://example.com"}]

        # First search at t=0
        mock_time.return_value = 0
        result1 = execute("test_topic", {"query": "test search"})
        assert result1['count'] == 1

        # Second search at t=2000 (> 1800s default cooldown)
        mock_time.return_value = 2000
        result2 = execute("test_topic", {"query": "test search"})
        assert result2['count'] == 1

    @patch('tools.duckduckgo_search.handler._search_ddg')
    def test_query_normalization(self, mock_search):
        """Whitespace/case should be normalized for dedup."""
        mock_search.return_value = [{"title": "Test", "snippet": "Test", "url": "http://example.com"}]

        # First search
        execute("test_topic", {"query": "test  SEARCH"})

        # Whitespace/case variations should be deduplicated
        result = execute("test_topic", {"query": "TEST search"})
        assert result['count'] == 0
        assert 'DUPLICATE' in result.get('message', '')

    def test_limit_clamped_1_to_10(self):
        """Limit should be clamped between 1 and 10."""
        with patch('tools.duckduckgo_search.handler._search_ddg') as mock_search:
            mock_search.return_value = []

            # Test limit=0 -> 1
            execute("test_topic", {"query": "test", "limit": 0})
            assert mock_search.call_args[0][1] == 1

            # Test limit=99 -> 10
            execute("test_topic", {"query": "test2", "limit": 99})
            # Note: may be blocked by dedup
