"""
Tests for Phase 5: Constraint pattern consolidation in IdleConsolidationService.

Verifies that recurring gate rejection patterns get consolidated into
episodic memory as constraint_learning episodes.
"""

import json
import time
import pytest
from unittest.mock import patch, MagicMock

from services.idle_consolidation_service import (
    IdleConsolidationService,
    _CONSTRAINT_CONSOLIDATION_KEY,
    _CONSTRAINT_CONSOLIDATION_COOLDOWN,
)


@pytest.fixture
def mock_store():
    """Mock MemoryStore."""
    store = MagicMock()
    store.get.return_value = None
    store.llen.return_value = 0
    return store


@pytest.fixture
def service(mock_store):
    """IdleConsolidationService with mocked dependencies."""
    with patch('services.idle_consolidation_service.MemoryClientService') as mock_mc, \
         patch('services.idle_consolidation_service.SemanticConsolidationTracker'), \
         patch('services.idle_consolidation_service.ConfigService') as mock_config:

        mock_mc.create_connection.return_value = mock_store
        mock_config.connections.return_value = {
            "memory": {"topics": {}, "queues": {}}
        }

        svc = IdleConsolidationService(check_interval=60)
        return svc


def _patch_lazy_imports():
    """Context manager helper that patches the lazy imports inside _consolidate_constraints."""
    mock_cms_cls = MagicMock()
    mock_db = MagicMock()
    mock_episodic_cls = MagicMock()
    mock_emb_svc = MagicMock()

    return (
        patch('services.constraint_memory_service.ConstraintMemoryService', mock_cms_cls),
        patch('services.database_service.get_shared_db_service', mock_db),
        patch('services.episodic_storage_service.EpisodicStorageService', mock_episodic_cls),
        patch('services.embedding_service.get_embedding_service', mock_emb_svc),
        mock_cms_cls,
        mock_db,
        mock_episodic_cls,
        mock_emb_svc,
    )


@pytest.mark.unit
class TestConstraintConsolidation:
    """Tests for _consolidate_constraints()."""

    def test_skips_when_on_cooldown(self, service, mock_store):
        """Should skip consolidation when cooldown flag exists."""
        mock_store.get.return_value = str(int(time.time()))

        service._consolidate_constraints()

        # Should check cooldown key
        mock_store.get.assert_called_with(_CONSTRAINT_CONSOLIDATION_KEY)

    def test_no_significant_patterns(self, service, mock_store):
        """Should set cooldown when no patterns have 10+ rejections."""
        mock_cms = MagicMock()
        mock_cms.get_blocked_action_patterns.return_value = [
            {'action': 'suggest', 'total_rejections': 5, 'top_reason': 'timing'}
        ]
        mock_episodic = MagicMock()

        with patch('services.constraint_memory_service.ConstraintMemoryService', return_value=mock_cms), \
             patch('services.database_service.get_shared_db_service'), \
             patch('services.episodic_storage_service.EpisodicStorageService', return_value=mock_episodic), \
             patch('services.embedding_service.get_embedding_service'):
            service._consolidate_constraints()

        # Cooldown should be set
        mock_store.setex.assert_called()
        # No episodes should be created
        mock_episodic.store_episode.assert_not_called()

    def test_creates_episode_for_significant_pattern(self, service, mock_store):
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

        mock_episodic = MagicMock()
        mock_episodic.store_episode.return_value = 'ep-123'

        with patch('services.constraint_memory_service.ConstraintMemoryService', return_value=mock_cms), \
             patch('services.database_service.get_shared_db_service'), \
             patch('services.episodic_storage_service.EpisodicStorageService', return_value=mock_episodic), \
             patch('services.embedding_service.get_embedding_service', return_value=mock_emb_svc), \
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

    def test_dedup_boosts_existing_episode(self, service, mock_store):
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

        mock_episodic = MagicMock()
        mock_db = MagicMock()

        existing = {'id': 'ep-existing', 'gist': 'old constraint', 'similarity': 0.92}

        with patch('services.constraint_memory_service.ConstraintMemoryService', return_value=mock_cms), \
             patch('services.database_service.get_shared_db_service', return_value=mock_db), \
             patch('services.episodic_storage_service.EpisodicStorageService', return_value=mock_episodic), \
             patch('services.embedding_service.get_embedding_service', return_value=mock_emb_svc), \
             patch.object(IdleConsolidationService, '_find_similar_constraint_episode', return_value=existing), \
             patch.object(IdleConsolidationService, '_boost_episode_activation') as mock_boost:
            service._consolidate_constraints()

        mock_boost.assert_called_once_with(mock_db, 'ep-existing')
        mock_episodic.store_episode.assert_not_called()

    def test_multiple_patterns_mixed(self, service, mock_store):
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
                return None  # suggest → new
            return {'id': 'ep-old', 'gist': 'old', 'similarity': 0.90}  # nurture → dup

        with patch('services.constraint_memory_service.ConstraintMemoryService', return_value=mock_cms), \
             patch('services.database_service.get_shared_db_service'), \
             patch('services.episodic_storage_service.EpisodicStorageService', return_value=mock_episodic), \
             patch('services.embedding_service.get_embedding_service', return_value=mock_emb_svc), \
             patch.object(IdleConsolidationService, '_find_similar_constraint_episode', side_effect=find_similar_side_effect), \
             patch.object(IdleConsolidationService, '_boost_episode_activation') as mock_boost:
            service._consolidate_constraints()

        # 1 created (suggest), 1 boosted (nurture), seed_thread filtered out
        assert mock_episodic.store_episode.call_count == 1
        mock_boost.assert_called_once()

    def test_cooldown_set_after_consolidation(self, service, mock_store):
        """Cooldown flag should be set after consolidation runs."""
        mock_cms = MagicMock()
        mock_cms.get_blocked_action_patterns.return_value = []

        with patch('services.constraint_memory_service.ConstraintMemoryService', return_value=mock_cms), \
             patch('services.database_service.get_shared_db_service'), \
             patch('services.episodic_storage_service.EpisodicStorageService'), \
             patch('services.embedding_service.get_embedding_service'):
            service._consolidate_constraints()

        # setex should be called with the cooldown key and TTL
        calls = [
            c for c in mock_store.setex.call_args_list
            if c[0][0] == _CONSTRAINT_CONSOLIDATION_KEY
        ]
        assert len(calls) >= 1
        assert calls[0][0][1] == _CONSTRAINT_CONSOLIDATION_COOLDOWN

    def test_trigger_consolidation_calls_constraint_consolidation(self, service, mock_store):
        """_trigger_consolidation should call _consolidate_constraints."""
        with patch.object(service, '_consolidate_constraints') as mock_cc, \
             patch('services.prompt_queue.PromptQueue'), \
             patch('workers.semantic_consolidation_worker.semantic_consolidation_worker'):
            service._trigger_consolidation()

        mock_cc.assert_called_once()


@pytest.mark.unit
class TestFindSimilarConstraintEpisode:
    """Tests for _find_similar_constraint_episode()."""

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
        # distance=0.1 → similarity = 1 - 0.1/2 = 0.95
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
        # distance=0.8 → similarity = 1 - 0.8/2 = 0.60
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

    def test_increments_activation_score(self):
        """Should execute UPDATE with activation_score + 1."""
        db = MagicMock()
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor
        db.connection.return_value.__enter__ = MagicMock(return_value=conn)
        db.connection.return_value.__exit__ = MagicMock(return_value=False)

        IdleConsolidationService._boost_episode_activation(db, 'ep-123')

        cursor.execute.assert_called_once()
        sql = cursor.execute.call_args[0][0]
        assert 'activation_score = activation_score + 1' in sql
        assert cursor.execute.call_args[0][1] == ('ep-123',)
