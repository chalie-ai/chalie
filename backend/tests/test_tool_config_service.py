"""
Tests for backend/services/tool_config_service.py
"""

import pytest
from unittest.mock import patch
from services.tool_config_service import ToolConfigService
from services.database_service import get_shared_db_service


@pytest.mark.unit
class TestToolConfigService:
    """Test tool configuration service."""

    @pytest.fixture
    def service(self, db):
        """Create service with real DB."""
        return ToolConfigService(get_shared_db_service())

    def test_get_empty_config(self, db, service):
        """Get on empty config should return empty dict."""
        result = service.get_tool_config("test_tool")
        assert result == {}

    def test_set_and_get_config(self, db, service):
        """Set then get should return stored config."""
        success = service.set_tool_config("test_tool", {"key1": "value1"})
        assert success is True

        result = service.get_tool_config("test_tool")
        assert result == {"key1": "value1"}

    def test_upsert_existing_key(self, db, service):
        """Updating existing key should overwrite the value."""
        service.set_tool_config("test_tool", {"key1": "old_value"})
        service.set_tool_config("test_tool", {"key1": "new_value"})

        result = service.get_tool_config("test_tool")
        assert result["key1"] == "new_value"

    def test_delete_existing_key(self, db, service):
        """Delete existing key should return True."""
        service.set_tool_config("test_tool", {"key1": "value1"})

        result = service.delete_tool_config_key("test_tool", "key1")
        assert result is True

        # Verify it is gone
        config = service.get_tool_config("test_tool")
        assert "key1" not in config

    def test_delete_nonexistent_key(self, db, service):
        """Delete nonexistent key should return False."""
        result = service.delete_tool_config_key("test_tool", "nonexistent")
        assert result is False

    def test_db_error_get(self, db, service):
        """DB error on get should return empty dict gracefully."""
        with patch.object(service, 'db') as broken_db:
            broken_db.connection.side_effect = Exception("DB error")
            result = service.get_tool_config("test_tool")
        assert result == {}

    def test_db_error_set(self, db, service):
        """DB error on set should return False gracefully."""
        with patch.object(service, 'db') as broken_db:
            broken_db.connection.side_effect = Exception("DB error")
            result = service.set_tool_config("test_tool", {"key": "value"})
        assert result is False
