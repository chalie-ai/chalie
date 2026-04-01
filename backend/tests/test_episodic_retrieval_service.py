"""Tests for EpisodicService — episode retrieval and scoring."""

import math
import pytest
from datetime import timedelta
from unittest.mock import patch, MagicMock

from services.episodic_service import EpisodicService
from services.database_service import get_shared_db_service
from services.time_utils import utc_now


pytestmark = pytest.mark.unit


class TestEpisodicRetrievalService:
    """Tests for EpisodicService retrieval, scoring, and configuration."""

    # ── Constructor / Configuration ───────────────────────────────────

    def test_default_embedding_dimensions(self, db):
        """Default embedding dimensions should be 256 when config is empty."""
        svc = EpisodicService(get_shared_db_service(), config={})
        assert svc.embedding_dimensions == 256

    def test_custom_embedding_dimensions(self, db):
        """Config-provided embedding_dimensions should override the default."""
        svc = EpisodicService(get_shared_db_service(), config={'embedding_dimensions': 512})
        assert svc.embedding_dimensions == 512

    def test_custom_weights_override_defaults(self, db):
        """Config-provided inference_weights should override DEFAULT weights."""
        custom_weights = {
            'vector_similarity': 10,
            'topic_overlap': 1,
            'intent_overlap': 1,
            'activation_score': 1,
            'outcome_relevance': 1,
        }
        svc = EpisodicService(get_shared_db_service(), config={'inference_weights': custom_weights})
        assert svc.weights == custom_weights

    def test_default_weights_used_when_config_empty(self, db):
        """Default weights should be used when config has no inference_weights."""
        svc = EpisodicService(get_shared_db_service(), config={})
        assert svc.weights['vector_similarity'] == 4
        assert svc.weights['topic_overlap'] == 2

    # ── Empty candidates ──────────────────────────────────────────────

    def test_empty_candidates_returns_empty_list(self, db):
        """When no candidates are found, retrieve_episodes should return []."""
        svc = EpisodicService(get_shared_db_service(), config={})

        with patch.object(svc, '_generate_embedding', return_value=[0.0] * 256), \
             patch.object(svc, '_hybrid_retrieve', return_value=[]):
            result = svc.retrieve_episodes(query_text='test query')

        assert result == []

    # ── Limit respected ───────────────────────────────────────────────

    def test_limit_is_respected(self, db):
        """retrieve_episodes should return at most `limit` results."""
        svc = EpisodicService(get_shared_db_service(), config={})

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

    def test_vector_similarity_identical_distance_zero(self, db):
        """Distance 0 (identical vectors) should produce score 10."""
        svc = EpisodicService(get_shared_db_service(), config={})
        score = svc._calculate_vector_similarity([0.1], 0.0)
        assert score == 10.0

    def test_vector_similarity_neutral_when_no_data(self, db):
        """Missing distance or embedding should produce neutral score 5.0."""
        svc = EpisodicService(get_shared_db_service(), config={})
        assert svc._calculate_vector_similarity(None, 0.5) == 5.0
        assert svc._calculate_vector_similarity([0.1], None) == 5.0

    def test_topic_overlap_exact_match(self, db):
        """Exact topic match should produce score 10.0."""
        svc = EpisodicService(get_shared_db_service(), config={})
        score = svc._calculate_topic_overlap('python', 'python')
        assert score == 10.0

    def test_topic_overlap_partial_match(self, db):
        """Partial topic match (substring) should produce score 7.0."""
        svc = EpisodicService(get_shared_db_service(), config={})
        score = svc._calculate_topic_overlap('python', 'python programming')
        assert score == 7.0

    def test_topic_overlap_no_match(self, db):
        """No topic overlap should produce score 2.0."""
        svc = EpisodicService(get_shared_db_service(), config={})
        score = svc._calculate_topic_overlap('python', 'cooking')
        assert score == 2.0

    def test_topic_overlap_neutral_when_no_query_topic(self, db):
        """Neutral score 5.0 when query topic is None."""
        svc = EpisodicService(get_shared_db_service(), config={})
        score = svc._calculate_topic_overlap(None, 'anything')
        assert score == 5.0

    # ── Effective freshness ───────────────────────────────────────────

    def test_effective_freshness_recent_episode_is_fresh(self, db):
        """An episode created moments ago should have high freshness."""
        svc = EpisodicService(get_shared_db_service(), config={})
        freshness = svc._calculate_effective_freshness(
            salience=0.5,
            created_at=utc_now(),
        )
        assert freshness > 0.9

    def test_effective_freshness_high_salience_slows_decay(self, db):
        """High salience should slow decay, resulting in higher freshness."""
        svc = EpisodicService(get_shared_db_service(), config={})
        created = utc_now() - timedelta(hours=48)

        fresh_high = svc._calculate_effective_freshness(salience=0.9, created_at=created)
        fresh_low = svc._calculate_effective_freshness(salience=0.1, created_at=created)

        assert fresh_high > fresh_low

    # ── Semantic boost ────────────────────────────────────────────────

    def test_semantic_boost_with_matching_concepts(self, db):
        """Concept names appearing in episode text should produce a boost."""
        svc = EpisodicService(get_shared_db_service(), config={})
        episode = {'gist': 'Learned about python decorators', 'outcome': 'understood decorators'}
        concepts = [{'name': 'decorators'}, {'name': 'metaclasses'}]
        boost = svc._calculate_semantic_boost(episode, concepts)
        assert boost > 0.0

    def test_semantic_boost_no_match_returns_zero(self, db):
        """No matching concepts should return 0.0 boost."""
        svc = EpisodicService(get_shared_db_service(), config={})
        episode = {'gist': 'Went to the gym', 'outcome': 'felt good'}
        concepts = [{'name': 'quantum mechanics'}]
        boost = svc._calculate_semantic_boost(episode, concepts)
        assert boost == 0.0

    def test_semantic_boost_empty_concepts_returns_zero(self, db):
        """Empty concept list should return 0.0 boost."""
        svc = EpisodicService(get_shared_db_service(), config={})
        episode = {'gist': 'test', 'outcome': 'test'}
        assert svc._calculate_semantic_boost(episode, []) == 0.0
        assert svc._calculate_semantic_boost(episode, None) == 0.0

    # ── Exception handling ────────────────────────────────────────────

    def test_retrieve_episodes_exception_returns_empty(self, db):
        """Any unhandled exception in retrieve_episodes should return []."""
        svc = EpisodicService(get_shared_db_service(), config={})

        with patch.object(svc, '_generate_embedding', side_effect=Exception('embed fail')):
            result = svc.retrieve_episodes(query_text='test')

        assert result == []

    # ── TopicContext integration ──────────────────────────────────────

    def test_retrieve_accepts_topic_context(self, db):
        """retrieve_episodes works when a TopicContext is passed."""
        from services.topic_context import TopicContext

        svc = EpisodicService(get_shared_db_service(), config={})
        ctx = TopicContext(topic='test')

        with patch.object(svc, '_generate_embedding', return_value=[0.0] * 256), \
             patch.object(svc, '_hybrid_retrieve', return_value=[]):
            result = svc.retrieve_episodes(query_text='hello', _context=ctx)

        assert result == []
        assert ctx.failed_sections == []

    def test_retrieve_records_failure_to_context(self, db):
        """When retrieval fails, failure is recorded on TopicContext."""
        from services.topic_context import TopicContext

        svc = EpisodicService(get_shared_db_service(), config={})
        ctx = TopicContext(topic='test')

        with patch.object(svc, '_generate_embedding', side_effect=Exception('embed boom')):
            result = svc.retrieve_episodes(query_text='test', _context=ctx)

        assert result == []
        assert len(ctx.failed_sections) == 1
        assert ctx.failed_sections[0][0] == 'episodic_retrieval'
        assert 'embed boom' in ctx.failed_sections[0][1]

    def test_retrieve_uses_context_embedding(self, db):
        """When TopicContext has message_embedding, it can be passed as query_embedding."""
        from services.topic_context import TopicContext

        svc = EpisodicService(get_shared_db_service(), config={})
        embedding = [0.5] * 256
        ctx = TopicContext(topic='test', message_embedding=embedding)

        with patch.object(svc, '_generate_embedding') as mock_gen, \
             patch.object(svc, '_hybrid_retrieve', return_value=[]):
            # Pass the context's embedding as query_embedding
            svc.retrieve_episodes(
                query_text='hello',
                query_embedding=ctx.message_embedding,
                _context=ctx,
            )

        # _generate_embedding should NOT have been called since we provided query_embedding
        mock_gen.assert_not_called()


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
