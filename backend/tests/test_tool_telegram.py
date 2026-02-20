"""
Tests for backend/tools/telegram/handler.py
"""

import pytest
from unittest.mock import patch, MagicMock
from tools.telegram.handler import execute


@pytest.mark.unit
class TestTelegramHandler:
    """Test telegram tool handler."""

    @patch('requests.post')
    def test_successful_send(self, mock_post):
        """Successful send should return sent=True."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        result = execute("test_topic", {"message": "Test message"}, {
            "TELEGRAM_TOKEN": "token123",
            "TELEGRAM_CHAT_ID": "chat123"
        })

        assert result['sent'] is True

    def test_empty_message(self):
        """Empty message should return error."""
        result = execute("test_topic", {"message": ""}, {
            "TELEGRAM_TOKEN": "token123",
            "TELEGRAM_CHAT_ID": "chat123"
        })

        assert result['sent'] is False
        assert "error" in result
        assert "message" in result['error'].lower()

    @patch('requests.post')
    def test_message_truncation(self, mock_post):
        """Messages > 4000 chars should be truncated."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        long_message = "x" * 5000
        result = execute("test_topic", {"message": long_message}, {
            "TELEGRAM_TOKEN": "token123",
            "TELEGRAM_CHAT_ID": "chat123"
        })

        # Check that the message was truncated
        called_json = mock_post.call_args[1]['json']
        assert len(called_json['text']) <= 4000 + len("\n\n... (truncated)")

    def test_missing_token(self):
        """Missing token should error."""
        result = execute("test_topic", {"message": "Test"}, {
            "TELEGRAM_CHAT_ID": "chat123"
        })

        assert result['sent'] is False
        assert "error" in result

    def test_missing_chat_id(self):
        """Missing chat_id should error."""
        result = execute("test_topic", {"message": "Test"}, {
            "TELEGRAM_TOKEN": "token123"
        })

        assert result['sent'] is False
        assert "error" in result

    @patch('requests.post')
    def test_http_error(self, mock_post):
        """HTTP error should return sent=False."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad request"
        mock_post.return_value = mock_response

        result = execute("test_topic", {"message": "Test"}, {
            "TELEGRAM_TOKEN": "token123",
            "TELEGRAM_CHAT_ID": "chat123"
        })

        assert result['sent'] is False
        assert "error" in result

    @patch('requests.post')
    def test_config_precedence(self, mock_post):
        """DB config should take precedence over env vars."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        # When config is provided, it should be used
        with patch.dict('os.environ', {'TELEGRAM_TOKEN': 'env_token', 'TELEGRAM_CHAT_ID': 'env_chat'}):
            execute("test_topic", {"message": "Test"}, {
                "TELEGRAM_TOKEN": "config_token",
                "TELEGRAM_CHAT_ID": "config_chat"
            })

        # Should use config values
        call_url = mock_post.call_args[0][0]
        assert "config_token" in call_url
