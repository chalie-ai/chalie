"""
Tests for backend/services/settings_service.py

Covers: get/set settings, sensitive value masking, API key generation.
Uses SQLAlchemy session mock pattern (db.get_session() context manager).
"""

import pytest
from unittest.mock import patch, MagicMock, call
from services.settings_service import SettingsService


@pytest.mark.unit
class TestSettingsService:

    @pytest.fixture
    def mock_db(self):
        """Provides (db, session, result) wired for db.get_session() context manager."""
        db = MagicMock()
        session = MagicMock()
        result = MagicMock()
        session.execute.return_value = result
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=session)
        ctx.__exit__ = MagicMock(return_value=False)
        db.get_session.return_value = ctx
        return db, session, result

    @pytest.fixture
    def service(self, mock_db):
        db, _, _ = mock_db
        with patch('services.encryption_key_service.get_encryption_key', return_value='test-key'):
            return SettingsService(db)

    # ── get ───────────────────────────────────────────────────────────

    def test_get_returns_value_when_found(self, service, mock_db):
        """get should return the resolved value when a matching row exists."""
        _, _, result = mock_db
        result.fetchone.return_value = ('my-setting-value',)

        value = service.get('theme')

        assert value == 'my-setting-value'

    def test_get_returns_none_when_not_found(self, service, mock_db):
        """get should return None when no row matches the key."""
        _, _, result = mock_db
        result.fetchone.return_value = None

        value = service.get('nonexistent_key')

        assert value is None

    # ── set ───────────────────────────────────────────────────────────

    def test_set_creates_new_setting(self, service, mock_db):
        """set should INSERT when no existing row is found for the key."""
        _, session, result = mock_db
        # First execute: SELECT to check existence returns None
        # Second execute: INSERT
        result.fetchone.return_value = None

        returned = service.set('new_key', 'new_value', 'string', 'A new setting')

        assert returned == 'new_value'
        # Should have been called at least twice: SELECT + INSERT
        assert session.execute.call_count >= 2
        session.commit.assert_called_once()

    def test_set_updates_existing_setting(self, service, mock_db):
        """set should UPDATE when an existing non-sensitive row is found."""
        _, session, result = mock_db
        # Existing row: (id=1, is_sensitive=False)
        result.fetchone.return_value = (1, False)

        returned = service.set('existing_key', 'updated_value')

        assert returned == 'updated_value'
        # SELECT + UPDATE
        assert session.execute.call_count >= 2
        session.commit.assert_called_once()

    # ── get_all ──────────────────────────────────────────────────────

    def test_get_all_masks_sensitive_values(self, service, mock_db):
        """get_all should return '***' for sensitive settings."""
        _, _, result = mock_db
        # The SQL CASE masks sensitive values at the query level,
        # so fetchall returns already-masked rows.
        result.fetchall.return_value = [
            ('api_key', '***'),
            ('theme', 'dark'),
            ('db_password', '***'),
        ]

        settings = service.get_all()

        assert settings['api_key'] == '***'
        assert settings['db_password'] == '***'

    def test_get_all_returns_non_sensitive_values(self, service, mock_db):
        """get_all should return plain text values for non-sensitive settings."""
        _, _, result = mock_db
        result.fetchall.return_value = [
            ('theme', 'dark'),
            ('language', 'en'),
        ]

        settings = service.get_all()

        assert settings['theme'] == 'dark'
        assert settings['language'] == 'en'
        assert len(settings) == 2

    # ── get_api_key_or_generate ──────────────────────────────────────

    def test_get_api_key_or_generate_returns_existing(self, service, mock_db):
        """get_api_key_or_generate should return existing key without generating."""
        _, _, result = mock_db
        # The get('api_key') call returns an existing key
        result.fetchone.return_value = ('existing-api-key-abc123',)

        key = service.get_api_key_or_generate()

        assert key == 'existing-api-key-abc123'

    def test_get_api_key_or_generate_creates_new_when_none(self, service, mock_db):
        """get_api_key_or_generate should generate and store a new key when absent."""
        _, session, result = mock_db

        # First call to get('api_key') returns None (no existing key).
        # Then set() is called, which does another SELECT (returns None) + INSERT.
        # Finally the generated key is returned.
        result.fetchone.return_value = None

        with patch('services.settings_service.secrets.token_urlsafe', return_value='generated-key-xyz789'):
            key = service.get_api_key_or_generate()

        assert key == 'generated-key-xyz789'
        # Verify that session.execute was called (for the INSERT inside set)
        assert session.execute.call_count >= 1
        session.commit.assert_called()
