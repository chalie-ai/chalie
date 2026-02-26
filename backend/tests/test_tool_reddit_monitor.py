"""
Tests for backend/tools/reddit_monitor/handler.py
"""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

feedparser = pytest.importorskip('feedparser', reason='feedparser not installed')
from tools.reddit_monitor.handler import execute
import tools.reddit_monitor.handler as reddit_handler


@pytest.mark.unit
class TestRedditMonitorHandler:
    """Test reddit_monitor tool handler."""

    @pytest.fixture(autouse=True)
    def setup_temp_state(self, tmp_path, monkeypatch):
        """Setup temporary state file for each test."""
        state_file = tmp_path / "state.json"
        monkeypatch.setattr(reddit_handler, 'STATE_FILE', state_file)

    def test_missing_threads(self):
        """Missing THREADS_TO_FOLLOW should return error."""
        result = execute("test_topic", {}, {})

        assert "error" in result

    @patch('datetime.datetime')
    def test_quiet_hours_block(self, mock_datetime):
        """During quiet hours should not notify."""
        # Set mock to 23:00 when quiet hours are 22:00-08:00
        mock_now = MagicMock()
        mock_now.hour = 23
        mock_now.minute = 0
        mock_now.second = 0
        mock_now.microsecond = 0
        mock_now.replace.return_value = mock_now
        mock_datetime.now.return_value = mock_now

        result = execute("test_topic", {}, {
            "THREADS_TO_FOLLOW": "r/python",
            "QUIET_HOURS_START": "22:00",
            "QUIET_HOURS_END": "08:00"
        })

        assert result['notify'] is False
        assert result['reason'] == "quiet_hours"

    @patch('datetime.datetime')
    def test_overnight_quiet_hours(self, mock_datetime):
        """Overnight quiet hours (22:00-08:00) should work."""
        # Set mock to 06:00 (in quiet hours)
        mock_now = MagicMock()
        mock_now.hour = 6
        mock_now.minute = 0
        mock_now.second = 0
        mock_now.microsecond = 0
        mock_now.replace.return_value = mock_now
        mock_datetime.now.return_value = mock_now

        with patch('tools.reddit_monitor.handler.requests.get') as mock_get, \
             patch('tools.reddit_monitor.handler.feedparser.parse') as mock_parse:
            mock_get.return_value = MagicMock(content=b"<rss></rss>")
            mock_parse.return_value = MagicMock(get=lambda x: [])

            result = execute("test_topic", {}, {
                "THREADS_TO_FOLLOW": "r/python",
                "QUIET_HOURS_START": "22:00",
                "QUIET_HOURS_END": "08:00"
            })

            assert result['notify'] is False
            assert result['reason'] == "quiet_hours"

    @patch('requests.get')
    @patch('feedparser.parse')
    def test_new_items_detected(self, mock_parse, mock_get):
        """New items should trigger notification."""
        mock_get.return_value = MagicMock(content=b"<rss></rss>")
        mock_feed = MagicMock()
        mock_feed.get.return_value = [
            {
                "id": "entry1",
                "title": "New post",
                "link": "http://reddit.com/post1"
            }
        ]
        mock_parse.return_value = mock_feed

        result = execute("test_topic", {}, {
            "THREADS_TO_FOLLOW": "r/python"
        })

        assert result['notify'] is True
        assert len(result['new_items']) > 0

    @patch('requests.get')
    @patch('feedparser.parse')
    def test_seen_items_filtered(self, mock_parse, mock_get):
        """Previously seen items should be skipped."""
        mock_get.return_value = MagicMock(content=b"<rss></rss>")

        # First call with one item
        mock_feed = MagicMock()
        mock_feed.get.return_value = [
            {"id": "entry1", "title": "Post 1", "link": "http://reddit.com/1"}
        ]
        mock_parse.return_value = mock_feed

        result1 = execute("test_topic", {}, {
            "THREADS_TO_FOLLOW": "r/python"
        })

        # Second call with same item
        result2 = execute("test_topic", {}, {
            "THREADS_TO_FOLLOW": "r/python"
        })

        # Second call should not have new items
        assert result2['notify'] is False

    @patch('requests.get')
    @patch('feedparser.parse')
    def test_no_new_items(self, mock_parse, mock_get):
        """No new items should not notify."""
        mock_get.return_value = MagicMock(content=b"<rss></rss>")
        mock_feed = MagicMock()
        mock_feed.get.return_value = []
        mock_parse.return_value = mock_feed

        result = execute("test_topic", {}, {
            "THREADS_TO_FOLLOW": "r/python"
        })

        assert result['notify'] is False
        assert result['reason'] == "no_new_items"

    @patch('requests.get')
    @patch('feedparser.parse')
    def test_subreddit_vs_post_url(self, mock_parse, mock_get):
        """Subreddit vs post URL should use correct RSS format."""
        mock_get.return_value = MagicMock(content=b"<rss></rss>")
        mock_feed = MagicMock()
        mock_feed.get.return_value = []
        mock_parse.return_value = mock_feed

        # Test with subreddit URL
        execute("test_topic", {}, {
            "THREADS_TO_FOLLOW": "https://www.reddit.com/r/python"
        })

        # Check that correct RSS URL was used
        call_url = mock_get.call_args[0][0]
        assert "/new/.rss" in call_url or "/.rss" in call_url
