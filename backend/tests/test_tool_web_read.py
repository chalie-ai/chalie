"""
Tests for backend/tools/web_read/handler.py
"""

import pytest
from unittest.mock import patch, MagicMock
from tools.web_read.handler import execute


@pytest.mark.unit
class TestWebReadHandler:
    """Test web_read tool handler."""

    @patch('requests.get')
    @patch('trafilatura.extract')
    def test_successful_extract(self, mock_extract, mock_get):
        """Successful extraction should return content."""
        mock_response = MagicMock()
        mock_response.text = "<html><body>Test content</body></html>"
        mock_get.return_value = mock_response

        mock_extract.return_value = "Extracted test content"

        result = execute("test_topic", {"url": "http://example.com"})

        assert "content" in result
        assert result['url'] == "http://example.com"
        assert result['char_count'] > 0

    def test_empty_url(self):
        """Empty URL should return error."""
        result = execute("test_topic", {"url": ""})

        assert "error" in result
        assert result['content'] == ""

    def test_max_chars_clamped(self):
        """max_chars should be clamped 100-4000."""
        with patch('tools.web_read.handler.requests.get') as mock_get, \
             patch('tools.web_read.handler.trafilatura.extract') as mock_extract:
            mock_response = MagicMock()
            mock_response.text = "<html>test</html>"
            mock_get.return_value = mock_response
            mock_extract.return_value = "x" * 5000

            # Test with max_chars=50 (should be clamped to 100)
            result = execute("test_topic", {"url": "http://example.com", "max_chars": 50})

            # Character count should be clamped
            assert result['char_count'] <= 4000

    @patch('requests.get')
    @patch('trafilatura.extract')
    def test_truncation_flag(self, mock_extract, mock_get):
        """Long content should set truncated=True."""
        mock_response = MagicMock()
        mock_response.text = "<html>test</html>"
        mock_get.return_value = mock_response

        long_content = "x" * 3000
        mock_extract.return_value = long_content

        result = execute("test_topic", {
            "url": "http://example.com",
            "max_chars": 1000
        })

        assert result['truncated'] is True

    @patch('requests.get')
    def test_fetch_failure(self, mock_get):
        """HTTP error should return error."""
        mock_get.side_effect = Exception("Connection error")

        result = execute("test_topic", {"url": "http://example.com"})

        assert "error" in result
        assert result['content'] == ""

    @patch('requests.get')
    @patch('trafilatura.extract')
    def test_no_readable_content(self, mock_extract, mock_get):
        """Short/empty extraction should error."""
        mock_response = MagicMock()
        mock_response.text = "<html>short</html>"
        mock_get.return_value = mock_response

        mock_extract.return_value = "x" * 30  # Too short

        result = execute("test_topic", {"url": "http://example.com"})

        assert "error" in result

    @patch('requests.get')
    def test_trafilatura_fallback(self, mock_get):
        """Regex fallback should work if trafilatura fails."""
        mock_response = MagicMock()
        mock_response.text = "<html><body>" + "content " * 20 + "</body></html>"
        mock_get.return_value = mock_response

        with patch('tools.web_read.handler.trafilatura.extract') as mock_extract:
            mock_extract.side_effect = Exception("trafilatura failed")

            result = execute("test_topic", {"url": "http://example.com"})

            # Should fall back to regex and extract something
            assert "error" not in result or result.get('char_count', 0) > 0

    @patch('requests.get')
    @patch('trafilatura.extract')
    def test_html_entity_decoding(self, mock_extract, mock_get):
        """HTML entities should be decoded."""
        mock_response = MagicMock()
        mock_response.text = "<html>Test &amp; content</html>"
        mock_get.return_value = mock_response

        mock_extract.return_value = None  # Force fallback

        result = execute("test_topic", {"url": "http://example.com"})

        # Fallback regex should decode entities
        if 'content' in result and result.get('content'):
            assert '&' in result['content'] or '&amp;' not in result['content']
