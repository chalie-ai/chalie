"""Tests for EpisodicRetrievalService — episode retrieval and scoring."""

import math
import pytest
from datetime import timedelta
from unittest.mock import patch, MagicMock

from services.episodic_retrieval_service import EpisodicRetrievalService
from services.time_utils import utc_now


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
                'created_at': utc_now() - timedelta(hours=i),
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
            created_at=utc_now(),
        )
        assert freshness > 0.9

    def test_effective_freshness_high_salience_slows_decay(self, mock_db_rows):
        """High salience should slow decay, resulting in higher freshness."""
        db, _ = mock_db_rows
        svc = EpisodicRetrievalService(db, config={})
        created = utc_now() - timedelta(hours=48)

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


# ── FTS5 alias regression ─────────────────────────────────────────────────────

class TestFts5AliasRegression:
    """
    Regression: the FTS query aliased episodes_fts as 'f' in the FROM clause, but
    SQLite FTS5 requires the MATCH operator in WHERE to reference the virtual table
    by its full unaliased name. Mixing an alias in FROM with the full name in WHERE
    causes empty results (not a syntax error), while using the alias in WHERE raises
    OperationalError('no such column').

    The fix removes the alias entirely — the table is referenced by its full name
    in SELECT (rank), FROM, JOIN ON, WHERE MATCH, and ORDER BY.

    These tests use a real in-memory SQLite FTS5 table to confirm the behaviour.
    """

    def _make_conn(self):
        import sqlite3
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE episodes (id INTEGER PRIMARY KEY, gist TEXT, deleted_at TEXT)")
        conn.execute("INSERT INTO episodes VALUES (1, 'watering plants reminder', NULL)")
        conn.execute(
            "CREATE VIRTUAL TABLE episodes_fts USING fts5(gist, content=episodes, content_rowid=id)"
        )
        conn.execute("INSERT INTO episodes_fts(episodes_fts) VALUES('rebuild')")
        return conn

    def test_fts5_no_alias_match_returns_results(self):
        """Fixed query — no alias, full table name in WHERE MATCH — returns rows."""
        conn = self._make_conn()
        query = """
            SELECT e.id, e.gist, episodes_fts.rank AS text_rank
            FROM episodes_fts
            JOIN episodes e ON e.rowid = episodes_fts.rowid
            WHERE episodes_fts MATCH ?
              AND e.deleted_at IS NULL
            ORDER BY episodes_fts.rank
        """
        rows = conn.execute(query, ("watering",)).fetchall()
        assert len(rows) == 1
        assert rows[0][1] == 'watering plants reminder'

    def test_fts5_alias_in_where_raises(self):
        """Alias in WHERE MATCH raises OperationalError — confirms alias form is invalid."""
        import sqlite3
        conn = self._make_conn()
        bad_query = """
            SELECT f.rank
            FROM episodes_fts f
            JOIN episodes e ON e.rowid = f.rowid
            WHERE f MATCH ?
        """
        with pytest.raises(sqlite3.OperationalError, match="no such column"):
            conn.execute(bad_query, ("watering",)).fetchall()

