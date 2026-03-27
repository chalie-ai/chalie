"""
Tests for backend/services/settings_service.py

Covers: get/set settings, sensitive value masking, API key generation.
Uses real SQLite via the shared `db` fixture (conftest.py).
"""

import pytest
from unittest.mock import patch
from services.settings_service import SettingsService
from services.database_service import get_shared_db_service


@pytest.mark.unit
class TestSettingsService:

    @pytest.fixture(autouse=True)
    def _clean_settings(self, db):
        """Remove migration-seeded settings so each test starts with a blank slate."""
        db.execute("DELETE FROM settings")
        db.commit()

    @pytest.fixture
    def service(self, db):
        with patch('services.encryption_key_service.get_encryption_key', return_value='test-key'):
            return SettingsService(get_shared_db_service())

    # ── get ───────────────────────────────────────────────────────────

    def test_get_returns_value_when_found(self, db, service):
        """get should return the resolved value when a matching row exists."""
        db.execute(
            "INSERT INTO settings (key, value, is_sensitive, value_type) "
            "VALUES ('theme', 'my-setting-value', 0, 'string')"
        )
        db.commit()

        value = service.get('theme')

        assert value == 'my-setting-value'

    def test_get_returns_none_when_not_found(self, db, service):
        """get should return None when no row matches the key."""
        value = service.get('nonexistent_key')

        assert value is None

    # ── set ───────────────────────────────────────────────────────────

    def test_set_creates_new_setting(self, db, service):
        """set should INSERT when no existing row is found for the key."""
        returned = service.set('new_key', 'new_value', 'string', 'A new setting')

        assert returned == 'new_value'

        # Verify the row was actually inserted
        row = db.execute(
            "SELECT value, value_type, description FROM settings WHERE key = 'new_key'"
        ).fetchone()
        assert row is not None
        assert row[0] == 'new_value'
        assert row[1] == 'string'
        assert row[2] == 'A new setting'

    def test_set_updates_existing_setting(self, db, service):
        """set should UPDATE when an existing non-sensitive row is found."""
        # Seed an existing non-sensitive row
        db.execute(
            "INSERT INTO settings (key, value, is_sensitive, value_type) "
            "VALUES ('existing_key', 'old_value', 0, 'string')"
        )
        db.commit()

        returned = service.set('existing_key', 'updated_value')

        assert returned == 'updated_value'

        # Verify the row was actually updated
        row = db.execute(
            "SELECT value FROM settings WHERE key = 'existing_key'"
        ).fetchone()
        assert row[0] == 'updated_value'

    # ── get_all ──────────────────────────────────────────────────────

    def test_get_all_masks_sensitive_values(self, db, service):
        """get_all should return '***' for sensitive settings."""
        db.execute(
            "INSERT INTO settings (key, value, encrypted_value, is_sensitive, value_type) "
            "VALUES ('api_key', NULL, 'encrypted-blob', 1, 'string')"
        )
        db.execute(
            "INSERT INTO settings (key, value, is_sensitive, value_type) "
            "VALUES ('theme', 'dark', 0, 'string')"
        )
        db.execute(
            "INSERT INTO settings (key, value, encrypted_value, is_sensitive, value_type) "
            "VALUES ('db_password', NULL, 'another-blob', 1, 'string')"
        )
        db.commit()

        settings = service.get_all()

        assert settings['api_key'] == '***'
        assert settings['db_password'] == '***'

    def test_get_all_returns_non_sensitive_values(self, db, service):
        """get_all should return plain text values for non-sensitive settings."""
        db.execute(
            "INSERT INTO settings (key, value, is_sensitive, value_type) "
            "VALUES ('theme', 'dark', 0, 'string')"
        )
        db.execute(
            "INSERT INTO settings (key, value, is_sensitive, value_type) "
            "VALUES ('language', 'en', 0, 'string')"
        )
        db.commit()

        settings = service.get_all()

        assert settings['theme'] == 'dark'
        assert settings['language'] == 'en'
        assert len(settings) == 2

    # ── get_api_key_or_generate ──────────────────────────────────────

    def test_get_api_key_or_generate_returns_existing(self, db, service):
        """get_api_key_or_generate should return existing key without generating."""
        db.execute(
            "INSERT INTO settings (key, value, is_sensitive, value_type) "
            "VALUES ('api_key', 'existing-api-key-abc123', 0, 'string')"
        )
        db.commit()

        key = service.get_api_key_or_generate()

        assert key == 'existing-api-key-abc123'

    def test_get_api_key_or_generate_creates_new_when_none(self, db, service):
        """get_api_key_or_generate should generate and store a new key when absent."""
        with patch('services.settings_service.secrets.token_urlsafe', return_value='generated-key-xyz789'):
            key = service.get_api_key_or_generate()

        assert key == 'generated-key-xyz789'

        # Verify the key was persisted — read back through the service to
        # handle both plain and encrypted storage paths transparently.
        stored = service.get('api_key')
        assert stored == 'generated-key-xyz789'
