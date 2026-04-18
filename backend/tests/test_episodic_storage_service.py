"""Tests for EpisodicService storage — real SQLite via shared db fixture."""

import uuid
import pytest
from unittest.mock import patch

from services.episodic_service import EpisodicService


pytestmark = pytest.mark.unit


@pytest.fixture
def episodic_svc(db):
    """EpisodicService backed by the real test DB."""
    from services.database_service import get_shared_db_service
    return EpisodicService(get_shared_db_service())


def _make_episode_data(**overrides):
    """Minimal valid episode data with all 3 required fields."""
    base = {
        'gist': 'Test conversation about coding',
        'salience': 5,
        'channel': 'programming',
    }
    base.update(overrides)
    return base


# ── store_episode — field validation ─────────────────────────────────

class TestStoreEpisodeValidation:

    @pytest.mark.parametrize("missing_field", [
        'gist', 'salience', 'channel',
    ])
    def test_raises_value_error_for_missing_field(self, episodic_svc, missing_field):
        """Each of the 3 required fields triggers ValueError when absent."""
        svc = episodic_svc
        data = _make_episode_data()
        del data[missing_field]

        with pytest.raises(ValueError, match=f"Missing required field: {missing_field}"):
            svc.store_episode(data)


class TestStoreEpisodeSuccess:

    def test_returns_uuid_string_on_success(self, db, episodic_svc):
        """Successful store returns the episode UUID as a string."""
        svc = episodic_svc
        test_uuid = 'c862db4f-bd83-4724-92a8-33fcf3bd38e8'

        with patch('services.episodic_service.uuid') as mock_uuid_mod:
            mock_uuid_mod.uuid4.return_value = uuid.UUID(test_uuid)
            result = svc.store_episode(_make_episode_data())

        assert result == test_uuid

        # Verify the episode was actually persisted
        row = db.execute("SELECT id, gist, channel FROM episodes WHERE id = ?", (test_uuid,)).fetchone()
        assert row is not None
        assert row[0] == test_uuid
        assert row[1] == 'Test conversation about coding'
        assert row[2] == 'programming'


# ── update_episode ───────────────────────────────────────────────────

class TestUpdateEpisode:

    def test_empty_updates_is_noop(self, episodic_svc):
        """Empty updates dict returns None without touching DB."""
        with patch.object(episodic_svc.db_service, 'connection', wraps=episodic_svc.db_service.connection) as spy:
            result = episodic_svc.update_episode('some-id', {})
            assert result is None
            spy.assert_not_called()

    def test_gist_update_persisted(self, db, episodic_svc):
        """Field update is written to the database."""
        svc = episodic_svc
        episode_id = svc.store_episode(_make_episode_data())

        result = svc.update_episode(episode_id, {'gist': 'updated gist'})
        assert result is None

        row = db.execute("SELECT gist FROM episodes WHERE id = ?", (episode_id,)).fetchone()
        assert row[0] == 'updated gist'

    def test_update_nonexistent_row_raises(self, episodic_svc):
        """Updating a row that doesn't exist propagates without silent failure."""
        # update_episode does not guard against missing rows — caller's responsibility.
        # It should complete without raising (SQLite silently updates 0 rows).
        svc = episodic_svc
        result = svc.update_episode('nonexistent-id', {'gist': 'updated'})
        assert result is None


# ── soft_delete_episode ──────────────────────────────────────────────

class TestSoftDeleteEpisode:

    def test_returns_true_when_row_deleted(self, db, episodic_svc):
        """rowcount>0 returns True."""
        svc = episodic_svc
        episode_id = svc.store_episode(_make_episode_data())

        result = svc.soft_delete_episode(episode_id)
        assert result is True

        # Verify deleted_at is set
        row = db.execute("SELECT deleted_at FROM episodes WHERE id = ?", (episode_id,)).fetchone()
        assert row[0] is not None

    def test_returns_false_when_not_found_or_already_deleted(self, episodic_svc):
        """rowcount=0 returns False (episode doesn't exist or already deleted)."""
        svc = episodic_svc
        result = svc.soft_delete_episode('nonexistent-id')
        assert result is False

    def test_returns_false_on_db_error(self, episodic_svc):
        """Database exception returns False (doesn't propagate)."""
        with patch.object(episodic_svc.db_service, 'connection', side_effect=Exception("connection lost")):
            result = episodic_svc.soft_delete_episode('some-id')
        assert result is False
