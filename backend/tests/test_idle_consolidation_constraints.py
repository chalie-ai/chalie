"""
Tests for Phase 5: Constraint pattern consolidation in IdleConsolidationService.

Verifies that recurring gate rejection patterns get consolidated into
episodic memory as constraint_learning episodes.
"""

import time
import pytest
from unittest.mock import patch, MagicMock

from services.idle_consolidation_service import (
    IdleConsolidationService,
    _CONSTRAINT_CONSOLIDATION_KEY,
    _CONSTRAINT_CONSOLIDATION_COOLDOWN,
)
from services.memory_store import MemoryStore


@pytest.fixture
def idle_mock_store():
    """Real MemoryStore for idle consolidation tests.

    Uses a fresh in-memory store so tests interact with genuine
    MemoryStore semantics (get returns None for missing keys, llen
    returns 0 for non-existent lists) instead of a shapeless mock.
    """
    return MemoryStore()


@pytest.fixture
def service(idle_mock_store):
    """IdleConsolidationService with mocked dependencies."""
    with patch('services.idle_consolidation_service.MemoryClientService') as mock_mc, \
         patch('services.idle_consolidation_service.SemanticConsolidationTracker'), \
         patch('services.idle_consolidation_service.ConfigService') as mock_config:

        mock_mc.create_connection.return_value = idle_mock_store
        mock_config.connections.return_value = {
            "memory": {"topics": {}, "queues": {}}
        }

        svc = IdleConsolidationService(check_interval=60)
        return svc


@pytest.mark.unit
class TestConstraintConsolidation:
    """Tests for _consolidate_constraints()."""

    def test_skips_when_on_cooldown(self, service, idle_mock_store):
        """Should skip consolidation when cooldown flag exists.

        Pre-populates the cooldown key with no TTL (``set`` instead of
        ``setex``).  After the service returns early, ``ttl`` must still
        report -1 (no expiry), proving ``setex`` was never called by the
        service path that runs when NOT on cooldown.
        """
        idle_mock_store.set(_CONSTRAINT_CONSOLIDATION_KEY, str(int(time.time())))

        service._consolidate_constraints()

        # Service returned early — the key must still carry no expiry
        # (we used .set with no TTL; if setex had run it would be > 0)
        assert idle_mock_store.ttl(_CONSTRAINT_CONSOLIDATION_KEY) == -1

    def test_no_significant_patterns(self, service, idle_mock_store):
        """Should set cooldown when no patterns have 10+ rejections."""
        mock_cms = MagicMock()
        mock_cms.get_blocked_action_patterns.return_value = [
            {'action': 'suggest', 'total_rejections': 5, 'top_reason': 'timing'}
        ]
        mock_episodic = MagicMock()

        # The code path for below-threshold patterns never reaches
        # get_embedding_service, so no embedding_service patch is needed.
        with patch('services.constraint_memory_service.ConstraintMemoryService', return_value=mock_cms), \
             patch('services.database_service.get_shared_db_service'), \
             patch('services.episodic_service.EpisodicService', return_value=mock_episodic):
            service._consolidate_constraints()

        # Cooldown flag must be written to the store
        assert idle_mock_store.get(_CONSTRAINT_CONSOLIDATION_KEY) is not None
        # No episodes should be created
        mock_episodic.store_episode.assert_not_called()

    def test_creates_episode_for_significant_pattern(self, service, idle_mock_store):
        """Should create an episode for patterns with 10+ rejections."""
        mock_cms = MagicMock()
        mock_cms.get_blocked_action_patterns.return_value = [
            {
                'action': 'communicate',
                'total_rejections': 15,
                'top_reason': 'timing_gate',
                'reason_breakdown': {'timing_gate': 10, 'quality_gate': 5},
            }
        ]

        mock_emb_svc = MagicMock()
        mock_emb_svc.generate_embedding.return_value = [0.1] * 384

        # services.embedding_service uses numpy C-extensions that cannot be
        # loaded in this sandbox, so we inject a fake module into sys.modules
        # directly rather than using patch() which would trigger the import.
        mock_emb_module = MagicMock()
        mock_emb_module.get_embedding_service.return_value = mock_emb_svc

        mock_episodic = MagicMock()
        mock_episodic.store_episode.return_value = 'ep-123'

        with patch.dict('sys.modules', {'services.embedding_service': mock_emb_module}), \
             patch('services.constraint_memory_service.ConstraintMemoryService', return_value=mock_cms), \
             patch('services.database_service.get_shared_db_service'), \
             patch('services.episodic_service.EpisodicService', return_value=mock_episodic), \
             patch.object(IdleConsolidationService, '_find_similar_constraint_episode', return_value=None):
            service._consolidate_constraints()

        mock_episodic.store_episode.assert_called_once()
        episode_data = mock_episodic.store_episode.call_args[0][0]

        assert episode_data['intent']['type'] == 'constraint_learning'
        assert episode_data['intent']['action'] == 'communicate'
        assert episode_data['outcome'] == 'constraint_learned'
        assert episode_data['salience'] == 3
        assert 'communicate' in episode_data['gist']
        assert 'timing_gate' in episode_data['gist']
        assert episode_data['topic'] == 'self_reflection'
        assert episode_data['embedding'] == [0.1] * 384

    def test_dedup_boosts_existing_episode(self, service, idle_mock_store):
        """Should boost activation when similar constraint episode exists."""
        mock_cms = MagicMock()
        mock_cms.get_blocked_action_patterns.return_value = [
            {
                'action': 'communicate',
                'total_rejections': 12,
                'top_reason': 'timing_gate',
                'reason_breakdown': {'timing_gate': 12},
            }
        ]

        mock_emb_svc = MagicMock()
        mock_emb_svc.generate_embedding.return_value = [0.1] * 384

        mock_emb_module = MagicMock()
        mock_emb_module.get_embedding_service.return_value = mock_emb_svc

        mock_episodic = MagicMock()

        existing = {'id': 'ep-existing', 'gist': 'old constraint', 'similarity': 0.92}

        with patch.dict('sys.modules', {'services.embedding_service': mock_emb_module}), \
             patch('services.constraint_memory_service.ConstraintMemoryService', return_value=mock_cms), \
             patch('services.database_service.get_shared_db_service') as mock_get_db, \
             patch('services.episodic_service.EpisodicService', return_value=mock_episodic), \
             patch.object(IdleConsolidationService, '_find_similar_constraint_episode', return_value=existing), \
             patch.object(IdleConsolidationService, '_boost_episode_activation') as mock_boost:
            service._consolidate_constraints()

        mock_boost.assert_called_once_with(mock_get_db.return_value, 'ep-existing')
        mock_episodic.store_episode.assert_not_called()

    def test_multiple_patterns_mixed(self, service, idle_mock_store):
        """Should handle mix of new and duplicate patterns."""
        mock_cms = MagicMock()
        mock_cms.get_blocked_action_patterns.return_value = [
            {'action': 'suggest', 'total_rejections': 20, 'top_reason': 'quality',
             'reason_breakdown': {'quality': 20}},
            {'action': 'nurture', 'total_rejections': 15, 'top_reason': 'phase',
             'reason_breakdown': {'phase': 15}},
            {'action': 'seed_thread', 'total_rejections': 5, 'top_reason': 'timing',
             'reason_breakdown': {'timing': 5}},  # Below threshold
        ]

        mock_emb_svc = MagicMock()
        mock_emb_svc.generate_embedding.return_value = [0.1] * 384

        mock_episodic = MagicMock()
        mock_episodic.store_episode.return_value = 'ep-new'

        call_count = [0]

        def find_similar_side_effect(db, emb, threshold=0.85):
            call_count[0] += 1
            if call_count[0] == 1:
                return None  # suggest -> new
            return {'id': 'ep-old', 'gist': 'old', 'similarity': 0.90}  # nurture -> dup

        mock_emb_module = MagicMock()
        mock_emb_module.get_embedding_service.return_value = mock_emb_svc

        with patch.dict('sys.modules', {'services.embedding_service': mock_emb_module}), \
             patch('services.constraint_memory_service.ConstraintMemoryService', return_value=mock_cms), \
             patch('services.database_service.get_shared_db_service'), \
             patch('services.episodic_service.EpisodicService', return_value=mock_episodic), \
             patch.object(IdleConsolidationService, '_find_similar_constraint_episode', side_effect=find_similar_side_effect), \
             patch.object(IdleConsolidationService, '_boost_episode_activation') as mock_boost:
            service._consolidate_constraints()

        # 1 created (suggest), 1 boosted (nurture), seed_thread filtered out
        assert mock_episodic.store_episode.call_count == 1
        mock_boost.assert_called_once()

    def test_cooldown_set_after_consolidation(self, service, idle_mock_store):
        """Cooldown flag should be set after consolidation runs."""
        mock_cms = MagicMock()
        mock_cms.get_blocked_action_patterns.return_value = []

        # patterns=[] → significant=[] → code returns before embedding_service is used
        with patch('services.constraint_memory_service.ConstraintMemoryService', return_value=mock_cms), \
             patch('services.database_service.get_shared_db_service'), \
             patch('services.episodic_service.EpisodicService'):
            service._consolidate_constraints()

        # Cooldown key must exist and carry the expected TTL
        assert idle_mock_store.get(_CONSTRAINT_CONSOLIDATION_KEY) is not None
        assert 0 < idle_mock_store.ttl(_CONSTRAINT_CONSOLIDATION_KEY) <= _CONSTRAINT_CONSOLIDATION_COOLDOWN

    def test_trigger_consolidation_calls_constraint_consolidation(self, service, idle_mock_store):
        """_trigger_consolidation should call _consolidate_constraints.

        ``workers`` package imports numpy (via digest_worker) which cannot be
        loaded in this sandbox, so we inject fake worker modules into
        sys.modules before the patch() resolver runs.
        """
        fake_worker_module = MagicMock()
        with patch.dict('sys.modules', {
                'workers.semantic_consolidation_worker': fake_worker_module,
             }), \
             patch.object(service, '_consolidate_constraints') as mock_cc, \
             patch('services.prompt_queue.PromptQueue'):
            service._trigger_consolidation()

        mock_cc.assert_called_once()


@pytest.mark.unit
class TestFindSimilarConstraintEpisode:
    """Tests for _find_similar_constraint_episode().

    These tests use a mock DatabaseService because the method performs
    sqlite-vec MATCH queries that require real vector data to be meaningful.
    The mock approach tests the similarity threshold logic without needing
    actual embeddings seeded into the vec table.
    """

    def test_returns_none_when_no_episodes(self):
        """Should return None when no constraint_learned episodes exist."""
        db = MagicMock()
        conn = MagicMock()
        cursor = MagicMock()
        cursor.fetchone.return_value = None
        conn.cursor.return_value = cursor
        db.connection.return_value.__enter__ = MagicMock(return_value=conn)
        db.connection.return_value.__exit__ = MagicMock(return_value=False)

        result = IdleConsolidationService._find_similar_constraint_episode(
            db, [0.1] * 384, threshold=0.85
        )
        assert result is None

    def test_returns_match_above_threshold(self):
        """Should return episode when similarity >= threshold."""
        db = MagicMock()
        conn = MagicMock()
        cursor = MagicMock()
        # distance=0.1 -> similarity = 1 - 0.1/2 = 0.95
        cursor.fetchone.return_value = ('ep-1', 'constraint gist', 0.1)
        conn.cursor.return_value = cursor
        db.connection.return_value.__enter__ = MagicMock(return_value=conn)
        db.connection.return_value.__exit__ = MagicMock(return_value=False)

        result = IdleConsolidationService._find_similar_constraint_episode(
            db, [0.1] * 384, threshold=0.85
        )
        assert result is not None
        assert result['id'] == 'ep-1'
        assert result['similarity'] == pytest.approx(0.95)

    def test_returns_none_below_threshold(self):
        """Should return None when similarity < threshold."""
        db = MagicMock()
        conn = MagicMock()
        cursor = MagicMock()
        # distance=0.8 -> similarity = 1 - 0.8/2 = 0.60
        cursor.fetchone.return_value = ('ep-1', 'constraint gist', 0.8)
        conn.cursor.return_value = cursor
        db.connection.return_value.__enter__ = MagicMock(return_value=conn)
        db.connection.return_value.__exit__ = MagicMock(return_value=False)

        result = IdleConsolidationService._find_similar_constraint_episode(
            db, [0.1] * 384, threshold=0.85
        )
        assert result is None


@pytest.mark.unit
class TestBoostEpisodeActivation:
    """Tests for _boost_episode_activation()."""

    def test_increments_activation_score(self, db):
        """Should increment activation_score by 1 in the real database."""
        from services.database_service import get_shared_db_service
        db_service = get_shared_db_service()

        # Seed an episode row
        db.execute(
            "INSERT INTO episodes (id, intent, context, action, emotion, outcome, "
            "gist, salience, freshness, topic, activation_score) "
            "VALUES ('ep-123', '{\"type\":\"constraint_learning\"}', '{}', 'test', "
            "'{\"valence\":0}', 'constraint_learned', 'test gist', 3, 1, "
            "'self_reflection', 5.0)"
        )
        db.commit()

        IdleConsolidationService._boost_episode_activation(db_service, 'ep-123')

        row = db.execute(
            "SELECT activation_score FROM episodes WHERE id = 'ep-123'"
        ).fetchone()
        assert row['activation_score'] == 6.0

    def test_boost_sets_last_accessed_at(self, db):
        """Boosting should update last_accessed_at to current timestamp."""
        from services.database_service import get_shared_db_service
        db_service = get_shared_db_service()

        db.execute(
            "INSERT INTO episodes (id, intent, context, action, emotion, outcome, "
            "gist, salience, freshness, topic, activation_score, last_accessed_at) "
            "VALUES ('ep-456', '{\"type\":\"constraint_learning\"}', '{}', 'test', "
            "'{\"valence\":0}', 'constraint_learned', 'test gist', 3, 1, "
            "'self_reflection', 1.0, NULL)"
        )
        db.commit()

        IdleConsolidationService._boost_episode_activation(db_service, 'ep-456')

        row = db.execute(
            "SELECT last_accessed_at FROM episodes WHERE id = 'ep-456'"
        ).fetchone()
        assert row['last_accessed_at'] is not None
