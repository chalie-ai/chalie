"""
Tests for backend/services/tool_config_service.py
"""

import pytest
from unittest.mock import MagicMock, patch
from services.tool_config_service import ToolConfigService


@pytest.mark.unit
class TestToolConfigService:
    """Test tool configuration service."""

    @pytest.fixture
    def mock_cursor(self):
        """Create a mock cursor with controllable returns."""
        cursor = MagicMock()
        cursor.fetchall.return_value = []
        cursor.fetchone.return_value = None
        cursor.rowcount = 0
        return cursor

    @pytest.fixture
    def mock_db(self, mock_cursor):
        """Create mock database service with proper connection → cursor chain."""
        db = MagicMock()
        conn = MagicMock()
        conn.cursor.return_value = mock_cursor
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=conn)
        ctx.__exit__ = MagicMock(return_value=False)
        db.connection.return_value = ctx
        return db

    @pytest.fixture
    def service(self, mock_db):
        """Create service with mocked DB."""
        return ToolConfigService(mock_db)

    def test_get_empty_config(self, service, mock_cursor):
        """Get on empty config should return empty dict."""
        mock_cursor.fetchall.return_value = []
        result = service.get_tool_config("test_tool")
        assert result == {}

    def test_set_and_get_config(self, service, mock_cursor):
        """Set then get should return stored config."""
        success = service.set_tool_config("test_tool", {"key1": "value1"})
        assert success is True

        mock_cursor.fetchall.return_value = [("key1", "value1")]
        result = service.get_tool_config("test_tool")
        assert result == {"key1": "value1"}

    def test_upsert_existing_key(self, service, mock_cursor):
        """Updating existing key should execute SQL."""
        service.set_tool_config("test_tool", {"key1": "new_value"})
        assert mock_cursor.execute.called

    def test_delete_existing_key(self, service, mock_cursor):
        """Delete existing key should return True."""
        mock_cursor.rowcount = 1
        result = service.delete_tool_config_key("test_tool", "key1")
        assert result is True

    def test_delete_nonexistent_key(self, service, mock_cursor):
        """Delete nonexistent key should return False."""
        mock_cursor.rowcount = 0
        result = service.delete_tool_config_key("test_tool", "nonexistent")
        assert result is False

    def test_db_error_get(self, service, mock_db):
        """DB error on get should return empty dict gracefully."""
        mock_db.connection.side_effect = Exception("DB error")

        result = service.get_tool_config("test_tool")

        assert result == {}

    def test_db_error_set(self, service, mock_db):
        """DB error on set should return False gracefully."""
        mock_db.connection.side_effect = Exception("DB error")

        result = service.set_tool_config("test_tool", {"key": "value"})

        assert result is False
