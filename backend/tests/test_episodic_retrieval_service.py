"""Tests for EpisodicRetrievalService — episode retrieval and scoring."""

import math
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from services.episodic_retrieval_service import EpisodicRetrievalService


pytestmark = pytest.mark.unit


class TestEpisodicRetrievalService:
    """Tests for EpisodicRetrievalService retrieval, scoring, and configuration."""

    # ── Constructor / Configuration ───────────────────────────────────

    def test_default_embedding_dimensions(self, mock_db_rows):
        """Default embedding dimensions should be 256 when config is empty."""
        db, _ = mock_db_rows
        svc = EpisodicRetrievalService(db, config={})
        assert svc.embedding_dimensions == 256

    def test_custom_embedding_dimensions(self, mock_db_rows):
        """Config-provided embedding_dimensions should override the default."""
        db, _ = mock_db_rows
        svc = EpisodicRetrievalService(db, config={'embedding_dimensions': 512})
        assert svc.embedding_dimensions == 512

    def test_custom_weights_override_defaults(self, mock_db_rows):
        """Config-provided inference_weights should override DEFAULT weights."""
        db, _ = mock_db_rows
        custom_weights = {
            'vector_similarity': 10,
            'topic_overlap': 1,
            'intent_overlap': 1,
            'activation_score': 1,
            'outcome_relevance': 1,
        }
        svc = EpisodicRetrievalService(db, config={'inference_weights': custom_weights})
        assert svc.weights == custom_weights

    def test_default_weights_used_when_config_empty(self, mock_db_rows):
        """Default weights should be used when config has no inference_weights."""
        db, _ = mock_db_rows
        svc = EpisodicRetrievalService(db, config={})
        assert svc.weights['vector_similarity'] == 4
        assert svc.weights['topic_overlap'] == 2

    # ── Empty candidates ──────────────────────────────────────────────

    def test_empty_candidates_returns_empty_list(self, mock_db_rows):
        """When no candidates are found, retrieve_episodes should return []."""
        db, _ = mock_db_rows
        svc = EpisodicRetrievalService(db, config={})

        with patch.object(svc, '_generate_embedding', return_value=[0.0] * 256), \
             patch.object(svc, '_hybrid_retrieve', return_value=[]):
            result = svc.retrieve_episodes(query_text='test query')

        assert result == []

    # ── Limit respected ───────────────────────────────────────────────

    def test_limit_is_respected(self, mock_db_rows):
        """retrieve_episodes should return at most `limit` results."""
        db, _ = mock_db_rows
        svc = EpisodicRetrievalService(db, config={})

        candidates = [
            {
                'id': str(i),
                'intent': {},
                'context': '',
                'action': f'action-{i}',
                'emotion': '',
                'outcome': f'outcome-{i}',
                'gist': f'gist-{i}',
                'salience': 5,
                'freshness': 0.8,
                'topic': 'test',
                'created_at': datetime.now() - timedelta(hours=i),
                'activation_score': 5.0,
                'last_accessed_at': None,
                'salience_factors': {},
                'open_loops': [],
                'vector_distance': 0.1 * i,
                'text_rank': None,
                'rrf_score': 1.0 / (60 + i),
            }
            for i in range(10)
        ]

        with patch.object(svc, '_generate_embedding', return_value=[0.0] * 256), \
             patch.object(svc, '_hybrid_retrieve', return_value=candidates), \
             patch.object(svc, '_apply_reconsolidation'):
            result = svc.retrieve_episodes(query_text='test', limit=3)

        assert len(result) <= 3

    # ── Scoring functions ─────────────────────────────────────────────

    def test_vector_similarity_identical_distance_zero(self, mock_db_rows):
        """Distance 0 (identical vectors) should produce score 10."""
        db, _ = mock_db_rows
        svc = EpisodicRetrievalService(db, config={})
        score = svc._calculate_vector_similarity([0.1], 0.0)
        assert score == 10.0

    def test_vector_similarity_neutral_when_no_data(self, mock_db_rows):
        """Missing distance or embedding should produce neutral score 5.0."""
        db, _ = mock_db_rows
        svc = EpisodicRetrievalService(db, config={})
        assert svc._calculate_vector_similarity(None, 0.5) == 5.0
        assert svc._calculate_vector_similarity([0.1], None) == 5.0

    def test_topic_overlap_exact_match(self, mock_db_rows):
        """Exact topic match should produce score 10.0."""
        db, _ = mock_db_rows
        svc = EpisodicRetrievalService(db, config={})
        score = svc._calculate_topic_overlap('python', 'python')
        assert score == 10.0

    def test_topic_overlap_partial_match(self, mock_db_rows):
        """Partial topic match (substring) should produce score 7.0."""
        db, _ = mock_db_rows
        svc = EpisodicRetrievalService(db, config={})
        score = svc._calculate_topic_overlap('python', 'python programming')
        assert score == 7.0

    def test_topic_overlap_no_match(self, mock_db_rows):
        """No topic overlap should produce score 2.0."""
        db, _ = mock_db_rows
        svc = EpisodicRetrievalService(db, config={})
        score = svc._calculate_topic_overlap('python', 'cooking')
        assert score == 2.0

    def test_topic_overlap_neutral_when_no_query_topic(self, mock_db_rows):
        """Neutral score 5.0 when query topic is None."""
        db, _ = mock_db_rows
        svc = EpisodicRetrievalService(db, config={})
        score = svc._calculate_topic_overlap(None, 'anything')
        assert score == 5.0

    # ── Effective freshness ───────────────────────────────────────────

    def test_effective_freshness_recent_episode_is_fresh(self, mock_db_rows):
        """An episode created moments ago should have high freshness."""
        db, _ = mock_db_rows
        svc = EpisodicRetrievalService(db, config={})
        freshness = svc._calculate_effective_freshness(
            salience=0.5,
            created_at=datetime.now(),
        )
        assert freshness > 0.9

    def test_effective_freshness_high_salience_slows_decay(self, mock_db_rows):
        """High salience should slow decay, resulting in higher freshness."""
        db, _ = mock_db_rows
        svc = EpisodicRetrievalService(db, config={})
        created = datetime.now() - timedelta(hours=48)

        fresh_high = svc._calculate_effective_freshness(salience=0.9, created_at=created)
        fresh_low = svc._calculate_effective_freshness(salience=0.1, created_at=created)

        assert fresh_high > fresh_low

    # ── Semantic boost ────────────────────────────────────────────────

    def test_semantic_boost_with_matching_concepts(self, mock_db_rows):
        """Concept names appearing in episode text should produce a boost."""
        db, _ = mock_db_rows
        svc = EpisodicRetrievalService(db, config={})
        episode = {'gist': 'Learned about python decorators', 'outcome': 'understood decorators'}
        concepts = [{'name': 'decorators'}, {'name': 'metaclasses'}]
        boost = svc._calculate_semantic_boost(episode, concepts)
        assert boost > 0.0

    def test_semantic_boost_no_match_returns_zero(self, mock_db_rows):
        """No matching concepts should return 0.0 boost."""
        db, _ = mock_db_rows
        svc = EpisodicRetrievalService(db, config={})
        episode = {'gist': 'Went to the gym', 'outcome': 'felt good'}
        concepts = [{'name': 'quantum mechanics'}]
        boost = svc._calculate_semantic_boost(episode, concepts)
        assert boost == 0.0

    def test_semantic_boost_empty_concepts_returns_zero(self, mock_db_rows):
        """Empty concept list should return 0.0 boost."""
        db, _ = mock_db_rows
        svc = EpisodicRetrievalService(db, config={})
        episode = {'gist': 'test', 'outcome': 'test'}
        assert svc._calculate_semantic_boost(episode, []) == 0.0
        assert svc._calculate_semantic_boost(episode, None) == 0.0

    # ── Exception handling ────────────────────────────────────────────

    def test_retrieve_episodes_exception_returns_empty(self, mock_db_rows):
        """Any unhandled exception in retrieve_episodes should return []."""
        db, _ = mock_db_rows
        svc = EpisodicRetrievalService(db, config={})

        with patch.object(svc, '_generate_embedding', side_effect=Exception('embed fail')):
            result = svc.retrieve_episodes(query_text='test')

        assert result == []
