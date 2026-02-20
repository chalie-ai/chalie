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
    def mock_db(self):
        """Create mock database service."""
        db = MagicMock()
        return db

    @pytest.fixture
    def service(self, mock_db):
        """Create service with mocked DB."""
        return ToolConfigService(mock_db)

    def test_get_empty_config(self, service, mock_db):
        """Get on empty config should return empty dict."""
        cursor = MagicMock()
        cursor.fetchall.return_value = []
        conn_ctx = MagicMock()
        conn_ctx.__enter__.return_value = cursor
        conn_ctx.__exit__.return_value = False
        mock_db.connection.return_value = conn_ctx

        result = service.get_tool_config("test_tool")

        assert result == {}

    def test_set_and_get_config(self, service, mock_db):
        """Set and get should work round-trip."""
        # Set
        cursor = MagicMock()
        conn_ctx = MagicMock()
        conn_ctx.__enter__.return_value = cursor
        conn_ctx.__exit__.return_value = False
        mock_db.connection.return_value = conn_ctx

        success = service.set_tool_config("test_tool", {"key1": "value1"})
        assert success is True

        # Get
        cursor.fetchall.return_value = [("key1", "value1")]
        result = service.get_tool_config("test_tool")

        assert result == {"key1": "value1"}

    def test_upsert_existing_key(self, service, mock_db):
        """Updating existing key should work."""
        cursor = MagicMock()
        conn_ctx = MagicMock()
        conn_ctx.__enter__.return_value = cursor
        conn_ctx.__exit__.return_value = False
        mock_db.connection.return_value = conn_ctx

        service.set_tool_config("test_tool", {"key1": "new_value"})

        # Should call execute
        assert cursor.execute.called

    def test_delete_existing_key(self, service, mock_db):
        """Delete existing key should return True."""
        cursor = MagicMock()
        cursor.rowcount = 1
        conn_ctx = MagicMock()
        conn_ctx.__enter__.return_value = cursor
        conn_ctx.__exit__.return_value = False
        mock_db.connection.return_value = conn_ctx

        result = service.delete_tool_config_key("test_tool", "key1")

        assert result is True

    def test_delete_nonexistent_key(self, service, mock_db):
        """Delete nonexistent key should return False."""
        cursor = MagicMock()
        cursor.rowcount = 0
        conn_ctx = MagicMock()
        conn_ctx.__enter__.return_value = cursor
        conn_ctx.__exit__.return_value = False
        mock_db.connection.return_value = conn_ctx

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
