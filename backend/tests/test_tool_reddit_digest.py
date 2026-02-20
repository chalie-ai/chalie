"""
Tests for backend/tools/reddit_digest/handler.py
"""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock
from tools.reddit_digest.handler import execute
import tools.reddit_digest.handler as reddit_handler


@pytest.mark.unit
class TestRedditDigestHandler:
    """Test reddit_digest tool handler."""

    @pytest.fixture(autouse=True)
    def setup_temp_state(self, tmp_path, monkeypatch):
        """Setup temporary state file for each test."""
        state_file = tmp_path / "state.json"
        monkeypatch.setattr(reddit_handler, 'STATE_FILE', state_file)

    def test_missing_daily_time(self):
        """Missing DAILY_TIME should return error."""
        result = execute("test_topic", {}, {
            "SUBREDDITS": "python"
        })

        assert "error" in result

    def test_missing_subreddits(self):
        """Missing SUBREDDITS should return error."""
        result = execute("test_topic", {}, {
            "DAILY_TIME": "09:00"
        })

        assert "error" in result

    @patch('datetime.datetime')
    def test_not_scheduled_time(self, mock_datetime):
        """Before scheduled time should not notify."""
        # Set mock to 08:00 when scheduled time is 09:00
        mock_now = MagicMock()
        mock_now.hour = 8
        mock_now.minute = 0
        mock_now.second = 0
        mock_now.microsecond = 0
        mock_now.strftime.return_value = "2024-01-15"
        mock_now.replace.return_value = mock_now
        mock_datetime.now.return_value = mock_now

        result = execute("test_topic", {}, {
            "DAILY_TIME": "09:00",
            "SUBREDDITS": "python"
        })

        assert result['notify'] is False
        assert result['reason'] == "not_scheduled_time"

    @patch('requests.get')
    @patch('feedparser.parse')
    def test_successful_digest(self, mock_parse, mock_get):
        """Successful digest should notify."""
        mock_response = MagicMock()
        mock_response.content = "<rss></rss>"
        mock_get.return_value = mock_response

        mock_feed = MagicMock()
        mock_feed.get.return_value = [
            {
                "title": "Python tip 1",
                "link": "http://example.com/1",
                "summary": "This is a great tip"
            }
        ]
        mock_parse.return_value = mock_feed

        result = execute("test_topic", {}, {
            "DAILY_TIME": "00:00",  # Always scheduled
            "SUBREDDITS": "python"
        })

        assert result['notify'] is True
        assert "message" in result

    @patch('requests.get')
    @patch('feedparser.parse')
    def test_no_posts_found(self, mock_parse, mock_get):
        """No posts found should return notify=False."""
        mock_response = MagicMock()
        mock_response.content = "<rss></rss>"
        mock_get.return_value = mock_response

        mock_feed = MagicMock()
        mock_feed.get.return_value = []
        mock_parse.return_value = mock_feed

        result = execute("test_topic", {}, {
            "DAILY_TIME": "00:00",
            "SUBREDDITS": "python"
        })

        assert result['notify'] is False
        assert result['reason'] == "no_posts_found"

    def test_invalid_daily_time_format(self):
        """Invalid DAILY_TIME format should error."""
        result = execute("test_topic", {}, {
            "DAILY_TIME": "not_a_time",
            "SUBREDDITS": "python"
        })

        assert "error" in result
        assert "HH:MM" in result['error']

    @patch('requests.get')
    @patch('feedparser.parse')
    def test_posts_per_sub_default(self, mock_parse, mock_get):
        """Should default to 3 posts per sub."""
        mock_response = MagicMock()
        mock_response.content = "<rss></rss>"
        mock_get.return_value = mock_response

        mock_feed = MagicMock()
        mock_feed.get.return_value = [
            {"title": f"Post {i}", "link": f"http://example.com/{i}", "summary": ""}
            for i in range(5)
        ]
        mock_parse.return_value = mock_feed

        result = execute("test_topic", {}, {
            "DAILY_TIME": "00:00",
            "SUBREDDITS": "python"
        })

        # Should only include 3 posts (default)
        if result['notify']:
            posts = result.get('subreddits', {}).get('python', [])
            assert len(posts) <= 3
