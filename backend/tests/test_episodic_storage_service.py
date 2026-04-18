"""Tests for EpisodicService storage — real SQLite via shared db fixture.

All tests use the real production stack. No mocks.
"""

import uuid

import pytest

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
        """Successful store returns a UUID string and persists the row."""
        svc = episodic_svc
        result = svc.store_episode(_make_episode_data())

        # Result must be a well-formed UUID string
        assert isinstance(result, str)
        parsed = uuid.UUID(result)  # raises if not a valid UUID
        assert str(parsed) == result

        # Verify the episode was actually persisted with the right data
        row = db.execute(
            "SELECT id, gist, channel FROM episodes WHERE id = ?", (result,)
        ).fetchone()
        assert row is not None
        assert row[0] == result
        assert row[1] == 'Test conversation about coding'
        assert row[2] == 'programming'

    def test_each_store_produces_distinct_id(self, episodic_svc):
        """Two store_episode calls produce two different UUIDs."""
        svc = episodic_svc
        id1 = svc.store_episode(_make_episode_data())
        id2 = svc.store_episode(_make_episode_data())
        assert id1 != id2


# ── update_episode ───────────────────────────────────────────────────

class TestUpdateEpisode:

    def test_empty_updates_is_noop(self, episodic_svc):
        """Empty updates dict returns None without modifying any row."""
        result = episodic_svc.update_episode('some-id', {})
        assert result is None

    def test_gist_update_persisted(self, db, episodic_svc):
        """Field update is written to the database."""
        svc = episodic_svc
        episode_id = svc.store_episode(_make_episode_data())

        result = svc.update_episode(episode_id, {'gist': 'updated gist'})
        assert result is None

        row = db.execute("SELECT gist FROM episodes WHERE id = ?", (episode_id,)).fetchone()
        assert row[0] == 'updated gist'

    def test_update_nonexistent_row_is_silent(self, episodic_svc):
        """Updating a row that doesn't exist completes without raising
        (SQLite silently updates 0 rows)."""
        result = episodic_svc.update_episode('nonexistent-id', {'gist': 'updated'})
        assert result is None


# ── soft_delete ──────────────────────────────────────────────────────

class TestSoftDeleteEpisode:

    def test_sets_deleted_at_on_existing_episode(self, db, episodic_svc):
        """soft_delete() sets deleted_at on an existing live episode."""
        svc = episodic_svc
        episode_id = svc.store_episode(_make_episode_data())

        svc.soft_delete(episode_id)

        row = db.execute("SELECT deleted_at FROM episodes WHERE id = ?", (episode_id,)).fetchone()
        assert row[0] is not None

    def test_nonexistent_id_is_noop(self, episodic_svc):
        """soft_delete() on a nonexistent ID completes without raising."""
        episodic_svc.soft_delete('nonexistent-id')

    def test_already_deleted_episode_timestamp_unchanged(self, db, episodic_svc):
        """soft_delete() on an already-deleted episode is a no-op (WHERE deleted_at IS NULL)."""
        svc = episodic_svc
        episode_id = svc.store_episode(_make_episode_data())
        svc.soft_delete(episode_id)

        first_deleted_at = db.execute(
            "SELECT deleted_at FROM episodes WHERE id = ?", (episode_id,)
        ).fetchone()[0]

        # Second call should not change deleted_at (WHERE deleted_at IS NULL filters it out)
        svc.soft_delete(episode_id)

        second_deleted_at = db.execute(
            "SELECT deleted_at FROM episodes WHERE id = ?", (episode_id,)
        ).fetchone()[0]
        assert first_deleted_at == second_deleted_at
