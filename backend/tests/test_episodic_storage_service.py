"""Tests for EpisodicStorageService — reduced mock depth (mock exactly one layer: DatabaseService)."""

import json
import uuid
import pytest
from unittest.mock import MagicMock, patch

from services.episodic_storage_service import EpisodicStorageService


pytestmark = pytest.mark.unit


def _make_episode_data(**overrides):
    """Minimal valid episode data with all 9 required fields."""
    base = {
        'intent': {'type': 'exploration'},
        'context': {'topic': 'test'},
        'action': 'user asked a question',
        'emotion': {'valence': 0.5},
        'outcome': 'answered successfully',
        'gist': 'Test conversation about coding',
        'salience': 5.0,
        'freshness': 0.8,
        'topic': 'programming',
    }
    base.update(overrides)
    return base


@pytest.fixture
def storage_env():
    """EpisodicStorageService with mocked DatabaseService — one-layer mock.

    Returns (service, mock_cursor) so tests can set cursor behavior.
    """
    mock_db = MagicMock()
    mock_cursor = MagicMock()
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_db.connection.return_value.__exit__ = MagicMock(return_value=False)

    svc = EpisodicStorageService(mock_db)
    return svc, mock_cursor


# ── store_episode — field validation ─────────────────────────────────

class TestStoreEpisodeValidation:

    @pytest.mark.parametrize("missing_field", [
        'intent', 'context', 'action', 'emotion', 'outcome',
        'gist', 'salience', 'freshness', 'topic',
    ])
    def test_raises_value_error_for_missing_field(self, missing_field, storage_env):
        """Each of the 9 required fields triggers ValueError when absent."""
        svc, _ = storage_env
        data = _make_episode_data()
        del data[missing_field]

        with pytest.raises(ValueError, match=f"Missing required field: {missing_field}"):
            svc.store_episode(data)


class TestStoreEpisodeSuccess:

    def test_returns_uuid_string_on_success(self, storage_env):
        """Successful store returns the episode UUID as a string."""
        svc, mock_cursor = storage_env
        test_uuid = str(uuid.uuid4())
        mock_cursor.fetchone.return_value = (test_uuid,)

        result = svc.store_episode(_make_episode_data())

        assert result == test_uuid
        mock_cursor.execute.assert_called_once()

    def test_curiosity_pursuit_failure_non_fatal(self, storage_env):
        """CuriosityPursuitService failure doesn't prevent episode storage."""
        svc, mock_cursor = storage_env
        test_uuid = str(uuid.uuid4())
        mock_cursor.fetchone.return_value = (test_uuid,)

        # Even if curiosity service fails (ImportError, etc.), store still succeeds
        with patch('services.episodic_storage_service.CuriosityPursuitService',
                   side_effect=ImportError("not available"), create=True):
            result = svc.store_episode(_make_episode_data())

        assert result == test_uuid


# ── update_episode ───────────────────────────────────────────────────

class TestUpdateEpisode:

    def test_empty_updates_returns_true_immediately(self, storage_env):
        """Empty updates dict → returns True without touching DB."""
        svc, mock_cursor = storage_env
        result = svc.update_episode('some-id', {})
        assert result is True
        mock_cursor.execute.assert_not_called()

    def test_returns_true_when_row_updated(self, storage_env):
        """rowcount=1 → returns True."""
        svc, mock_cursor = storage_env
        mock_cursor.rowcount = 1

        result = svc.update_episode('some-id', {'gist': 'updated gist'})
        assert result is True

    def test_returns_false_when_no_row_found(self, storage_env):
        """rowcount=0 → returns False."""
        svc, mock_cursor = storage_env
        mock_cursor.rowcount = 0

        result = svc.update_episode('nonexistent-id', {'gist': 'updated'})
        assert result is False

    def test_json_fields_serialized(self, storage_env):
        """JSON-typed fields (intent, context, emotion, etc.) are serialized."""
        svc, mock_cursor = storage_env
        mock_cursor.rowcount = 1

        svc.update_episode('some-id', {'intent': {'type': 'new'}})

        # Verify the execute call passed json.dumps of the intent
        call_args = mock_cursor.execute.call_args[0]
        params = call_args[1]
        assert json.loads(params[0]) == {'type': 'new'}


# ── soft_delete_episode ──────────────────────────────────────────────

class TestSoftDeleteEpisode:

    def test_returns_true_when_row_deleted(self, storage_env):
        """rowcount>0 → returns True."""
        svc, mock_cursor = storage_env
        mock_cursor.rowcount = 1

        result = svc.soft_delete_episode('some-id')
        assert result is True

    def test_returns_false_when_not_found_or_already_deleted(self, storage_env):
        """rowcount=0 → returns False (episode doesn't exist or already deleted)."""
        svc, mock_cursor = storage_env
        mock_cursor.rowcount = 0

        result = svc.soft_delete_episode('nonexistent-id')
        assert result is False

    def test_returns_false_on_db_error(self, storage_env):
        """Database exception → returns False (doesn't propagate)."""
        svc, mock_cursor = storage_env
        mock_cursor.execute.side_effect = Exception("connection lost")

        result = svc.soft_delete_episode('some-id')
        assert result is False
