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
            'retrieval_weight': 1,
            'outcome_relevance': 1,
        }
        svc = EpisodicService(get_shared_db_service(), config={'inference_weights': custom_weights})
        assert svc.weights == custom_weights

    def test_default_weights_used_when_config_empty(self, db):
        """Default weights should be used when config has no inference_weights."""
        svc = EpisodicService(get_shared_db_service(), config={})
        assert svc.weights['vector_similarity'] == 4
        assert 'topic_overlap' not in svc.weights

    # ── Empty candidates ──────────────────────────────────────────────

    def test_empty_candidates_returns_empty_list(self, db):
        """When no candidates are found, retrieve_episodes should return []."""
        svc = EpisodicService(get_shared_db_service(), config={})

        with patch.object(svc, '_generate_embedding', return_value=[0.0] * 256), \
             patch.object(svc, '_hybrid_retrieve', return_value=[]):
            result = svc.retrieve_episodes(query_text='test query')

        assert result == []

    # ── Radius controls result count ────────────────────────────────

    def test_radius_filters_by_distance(self, db):
        """retrieve_episodes returns all episodes within the adaptive radius."""
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
                'topic': 'test',
                'created_at': utc_now() - timedelta(hours=i),
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
            result = svc.retrieve_episodes(query_text='test', radius=0.3)

        assert len(result) == 10  # all candidates pass reranking

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

    # ── Exception handling ────────────────────────────────────────────

    def test_retrieve_episodes_exception_returns_empty(self, db):
        """Any unhandled exception in retrieve_episodes should return []."""
        svc = EpisodicService(get_shared_db_service(), config={})

        with patch.object(svc, '_generate_embedding', side_effect=Exception('embed fail')):
            result = svc.retrieve_episodes(query_text='test')

        assert result == []

    # ── Simplified API ──────────────────────────────────────────────

    def test_retrieve_with_default_radius(self, db):
        """retrieve_episodes works with just query_text and default radius."""
        svc = EpisodicService(get_shared_db_service(), config={})

        with patch.object(svc, '_generate_embedding', return_value=[0.0] * 256), \
             patch.object(svc, '_hybrid_retrieve', return_value=[]):
            result = svc.retrieve_episodes(query_text='hello')

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


# ── _analyze_query ────────────────────────────────────────────────────────────

class TestAnalyzeQuery:
    """Tests for EpisodicService._analyze_query — NLTK POS + noun-phrase chunking."""

    def test_normal_sentence_extracts_proper_nouns_as_entities(self, db):
        """Proper nouns (NNP/NNPS) should appear in entities."""
        svc = EpisodicService(get_shared_db_service(), config={})
        result = svc._analyze_query("Alice went to London last Tuesday")
        # Alice and London are proper nouns — both should be entities
        assert any(e in result['entities'] for e in ['Alice', 'London', 'Tuesday'])

    def test_normal_sentence_extracts_content_words_as_keywords(self, db):
        """Content words (nouns, verbs, adjectives) should appear in keywords."""
        svc = EpisodicService(get_shared_db_service(), config={})
        result = svc._analyze_query("the big brown dog runs fast")
        # 'big', 'brown', 'dog', 'runs', 'fast' are content words; 'the' is a DT stop
        assert len(result['keywords']) > 0
        assert 'dog' in result['keywords'] or 'runs' in result['keywords']

    def test_empty_string_returns_empty_lists(self, db):
        """Empty query text should return empty entities, keywords, and noun_phrases."""
        svc = EpisodicService(get_shared_db_service(), config={})
        result = svc._analyze_query("")
        assert result['entities'] == []
        assert result['keywords'] == []
        assert result['noun_phrases'] == []

    def test_single_word_appears_in_keywords(self, db):
        """A single content word should appear in keywords."""
        svc = EpisodicService(get_shared_db_service(), config={})
        result = svc._analyze_query("coffee")
        assert 'coffee' in result['keywords'] or 'coffee' in result['entities']

    def test_nltk_unavailable_falls_back_to_whitespace_split(self, db):
        """When NLTK is unavailable the method should split on whitespace."""
        svc = EpisodicService(get_shared_db_service(), config={})
        import services.episodic_service as _mod
        with patch.object(_mod, '_NLTK_AVAILABLE', False):
            result = svc._analyze_query("hello world test")
        assert result['entities'] == ['hello', 'world', 'test']
        assert result['keywords'] == ['hello', 'world', 'test']
        assert result['noun_phrases'] == []

    def test_nltk_unavailable_empty_string_returns_empty(self, db):
        """Whitespace fallback with empty string still returns empty lists."""
        svc = EpisodicService(get_shared_db_service(), config={})
        import services.episodic_service as _mod
        with patch.object(_mod, '_NLTK_AVAILABLE', False):
            result = svc._analyze_query("")
        assert result['entities'] == []
        assert result['keywords'] == []
        assert result['noun_phrases'] == []

    def test_result_has_required_keys(self, db):
        """Return value always contains entities, keywords, and noun_phrases keys."""
        svc = EpisodicService(get_shared_db_service(), config={})
        result = svc._analyze_query("some query text")
        assert 'entities' in result
        assert 'keywords' in result
        assert 'noun_phrases' in result


# ── format_for_prompt ─────────────────────────────────────────────────────────

class TestFormatForPrompt:
    """Tests for EpisodicService.format_for_prompt."""

    def test_empty_list_returns_empty_string(self, db):
        """format_for_prompt([]) should return ''."""
        svc = EpisodicService(get_shared_db_service(), config={})
        assert svc.format_for_prompt([]) == ''

    def test_single_episode_formats_as_date_dash_gist(self, db):
        """Single episode with valid created_at and gist → 'YYYY-MM-DD — gist'."""
        svc = EpisodicService(get_shared_db_service(), config={})
        episodes = [{'created_at': '2026-03-15T10:00:00+00:00', 'gist': 'had coffee'}]
        result = svc.format_for_prompt(episodes)
        assert result == '2026-03-15 — had coffee'

    def test_episode_with_consolidated_from_returns_date_range(self, db):
        """Episode with consolidated_from and source episodes in DB → 'YYYY-MM-DD to YYYY-MM-DD — gist'."""
        svc = EpisodicService(get_shared_db_service(), config={})

        # Insert two source episodes with distinct dates
        import json, uuid
        src_id_a = str(uuid.uuid4())
        src_id_b = str(uuid.uuid4())
        db_conn = get_shared_db_service()
        with db_conn.connection() as conn:
            conn.execute(
                "INSERT INTO episodes (id, intent, context, action, emotion, outcome, gist, salience, channel, created_at) "
                "VALUES (?, '{}', '', '', '{}', '', 'old gist a', 5, 'test', ?)",
                (src_id_a, '2026-01-01T00:00:00+00:00')
            )
            conn.execute(
                "INSERT INTO episodes (id, intent, context, action, emotion, outcome, gist, salience, channel, created_at) "
                "VALUES (?, '{}', '', '', '{}', '', 'old gist b', 5, 'test', ?)",
                (src_id_b, '2026-01-10T00:00:00+00:00')
            )

        episode = {
            'created_at': '2026-01-10T12:00:00+00:00',
            'gist': 'consolidated memory',
            'consolidated_from': json.dumps([src_id_a, src_id_b]),
        }
        result = svc.format_for_prompt([episode])
        assert '2026-01-01 to 2026-01-10' in result
        assert 'consolidated memory' in result

    def test_episode_with_none_gist_is_skipped(self, db):
        """Episodes where gist is None or empty should not appear in output."""
        svc = EpisodicService(get_shared_db_service(), config={})
        episodes = [
            {'created_at': '2026-03-15T10:00:00+00:00', 'gist': None},
            {'created_at': '2026-03-16T10:00:00+00:00', 'gist': ''},
            {'created_at': '2026-03-17T10:00:00+00:00', 'gist': 'valid gist'},
        ]
        result = svc.format_for_prompt(episodes)
        assert result == '2026-03-17 — valid gist'

    def test_malformed_created_at_string_slice_fallback_on_parse_exception(self, db):
        """When parse_utc raises, the except branch falls back to str(created_at)[:10]."""
        svc = EpisodicService(get_shared_db_service(), config={})
        episodes = [{'created_at': '2026-04-07-extra', 'gist': 'fallback test'}]
        # Patch parse_utc to raise so the except branch executes
        with patch('services.episodic_service.parse_utc', side_effect=ValueError('bad date')):
            result = svc.format_for_prompt(episodes)
        # str('2026-04-07-extra')[:10] == '2026-04-07'
        assert result == '2026-04-07 — fallback test'

    def test_multiple_episodes_are_separated_by_newlines(self, db):
        """Multiple episodes should be joined by newlines, one per line."""
        svc = EpisodicService(get_shared_db_service(), config={})
        episodes = [
            {'created_at': '2026-03-01T00:00:00+00:00', 'gist': 'first'},
            {'created_at': '2026-03-02T00:00:00+00:00', 'gist': 'second'},
        ]
        result = svc.format_for_prompt(episodes)
        lines = result.split('\n')
        assert len(lines) == 2
        assert lines[0] == '2026-03-01 — first'
        assert lines[1] == '2026-03-02 — second'

    def test_consolidated_from_same_date_returns_single_date(self, db):
        """When source episodes share the same date, output is a single date, not a range."""
        svc = EpisodicService(get_shared_db_service(), config={})
        import json, uuid
        src_id = str(uuid.uuid4())
        db_conn = get_shared_db_service()
        with db_conn.connection() as conn:
            conn.execute(
                "INSERT INTO episodes (id, intent, context, action, emotion, outcome, gist, salience, channel, created_at) "
                "VALUES (?, '{}', '', '', '{}', '', 'same day', 5, 'test', ?)",
                (src_id, '2026-02-14T09:00:00+00:00')
            )

        episode = {
            'created_at': '2026-02-14T10:00:00+00:00',
            'gist': 'valentine summary',
            'consolidated_from': json.dumps([src_id]),
        }
        result = svc.format_for_prompt([episode])
        assert result == '2026-02-14 — valentine summary'


# ── _count_episodes ───────────────────────────────────────────────────────────

class TestCountEpisodes:
    """Tests for EpisodicService._count_episodes."""

    def test_returns_zero_on_empty_db(self, db):
        """Empty episodes table should yield count 0."""
        svc = EpisodicService(get_shared_db_service(), config={})
        assert svc._count_episodes() == 0

    def test_returns_correct_count_after_inserts(self, db):
        """Count should reflect the number of non-deleted episodes inserted."""
        svc = EpisodicService(get_shared_db_service(), config={})
        import uuid
        db_conn = get_shared_db_service()
        with db_conn.connection() as conn:
            for i in range(5):
                conn.execute(
                    "INSERT INTO episodes (id, intent, context, action, emotion, outcome, gist, salience, channel) "
                    "VALUES (?, '{}', '', '', '{}', '', ?, 5, 'test')",
                    (str(uuid.uuid4()), f'gist {i}')
                )
        assert svc._count_episodes() == 5

    def test_does_not_count_soft_deleted_episodes(self, db):
        """Soft-deleted episodes (deleted_at IS NOT NULL) must not be counted."""
        svc = EpisodicService(get_shared_db_service(), config={})
        import uuid
        ep_id = str(uuid.uuid4())
        db_conn = get_shared_db_service()
        with db_conn.connection() as conn:
            conn.execute(
                "INSERT INTO episodes (id, intent, context, action, emotion, outcome, gist, salience, channel, deleted_at) "
                "VALUES (?, '{}', '', '', '{}', '', 'deleted ep', 5, 'test', datetime('now'))",
                (ep_id,)
            )
            conn.execute(
                "INSERT INTO episodes (id, intent, context, action, emotion, outcome, gist, salience, channel) "
                "VALUES (?, '{}', '', '', '{}', '', 'live ep', 5, 'test')",
                (str(uuid.uuid4()),)
            )
        assert svc._count_episodes() == 1


# ── Adaptive radius math ──────────────────────────────────────────────────────

class TestAdaptiveRadius:
    """Tests for the adaptive radius formula: effective_radius = radius / (1 + 0.1 * log2(count + 2))."""

    def test_zero_episodes_radius_near_input(self, db):
        """With 0 episodes: effective_radius = 0.3 / (1 + 0.1*log2(2)) = 0.3 / 1.1 ≈ 0.273."""
        svc = EpisodicService(get_shared_db_service(), config={})
        # DB is empty, so _count_episodes() returns 0
        captured = {}

        def capture_radius(embedding, text, effective_radius):
            captured['effective_radius'] = effective_radius
            return []

        with patch.object(svc, '_generate_embedding', return_value=[0.0] * 256), \
             patch.object(svc, '_hybrid_retrieve', side_effect=capture_radius):
            svc.retrieve_episodes(query_text='test', radius=0.3)

        expected = 0.3 / (1 + 0.1 * math.log2(0 + 2))  # 0.3 / 1.1
        assert abs(captured['effective_radius'] - expected) < 1e-9

    def test_one_hundred_episodes_tightens_radius(self, db):
        """With 100 episodes: effective_radius = 0.3 / (1 + 0.1*log2(102)) ≈ 0.18."""
        svc = EpisodicService(get_shared_db_service(), config={})
        captured = {}

        def capture_radius(embedding, text, effective_radius):
            captured['effective_radius'] = effective_radius
            return []

        with patch.object(svc, '_generate_embedding', return_value=[0.0] * 256), \
             patch.object(svc, '_hybrid_retrieve', side_effect=capture_radius), \
             patch.object(svc, '_count_episodes', return_value=100):
            svc.retrieve_episodes(query_text='test', radius=0.3)

        expected = 0.3 / (1 + 0.1 * math.log2(102))
        assert abs(captured['effective_radius'] - expected) < 1e-9
        assert captured['effective_radius'] < 0.3  # tighter than input

    def test_one_thousand_episodes_radius_even_tighter(self, db):
        """With 1000 episodes: radius is tighter than with 100 episodes."""
        svc = EpisodicService(get_shared_db_service(), config={})
        captured_100 = {}
        captured_1000 = {}

        def make_capture(bucket):
            def _fn(embedding, text, effective_radius):
                bucket['effective_radius'] = effective_radius
                return []
            return _fn

        with patch.object(svc, '_generate_embedding', return_value=[0.0] * 256), \
             patch.object(svc, '_hybrid_retrieve', side_effect=make_capture(captured_100)), \
             patch.object(svc, '_count_episodes', return_value=100):
            svc.retrieve_episodes(query_text='test', radius=0.3)

        with patch.object(svc, '_generate_embedding', return_value=[0.0] * 256), \
             patch.object(svc, '_hybrid_retrieve', side_effect=make_capture(captured_1000)), \
             patch.object(svc, '_count_episodes', return_value=1000):
            svc.retrieve_episodes(query_text='test', radius=0.3)

        assert captured_1000['effective_radius'] < captured_100['effective_radius']

    def test_radius_formula_matches_expected_values(self, db):
        """Verify formula at several specific episode counts."""
        svc = EpisodicService(get_shared_db_service(), config={})
        test_cases = [
            (0,    0.3 / (1 + 0.1 * math.log2(2))),
            (1,    0.3 / (1 + 0.1 * math.log2(3))),
            (14,   0.3 / (1 + 0.1 * math.log2(16))),
            (510,  0.3 / (1 + 0.1 * math.log2(512))),
        ]
        for count, expected in test_cases:
            captured = {}

            def capture(embedding, text, er, _expected=expected):
                captured['er'] = er
                return []

            with patch.object(svc, '_generate_embedding', return_value=[0.0] * 256), \
                 patch.object(svc, '_hybrid_retrieve', side_effect=capture), \
                 patch.object(svc, '_count_episodes', return_value=count):
                svc.retrieve_episodes(query_text='test', radius=0.3)

            assert abs(captured['er'] - expected) < 1e-9, (
                f"count={count}: got {captured['er']}, expected {expected}"
            )


# ── retrieve_episodes integration ────────────────────────────────────────────

class TestRetrieveEpisodesIntegration:
    """Integration-level tests for retrieve_episodes — mocked embedding + hybrid."""

    def _make_candidate(self, idx: int) -> dict:
        """Build a minimal candidate dict as _hybrid_retrieve would return."""
        return {
            'id': str(idx),
            'intent': '{}',
            'context': '',
            'action': f'action-{idx}',
            'emotion': '{}',
            'outcome': f'outcome-{idx}',
            'gist': f'gist-{idx}',
            'salience': 5,
            'topic': 'test',
            'created_at': utc_now() - timedelta(hours=idx),
            'last_accessed_at': None,
            'salience_factors': {},
            'open_loops': [],
            'entities': '[]',
            'goal_tags': '[]',
            'emotional_valence': None,
            'emotional_arousal': None,
            'vector_distance': 0.05 * idx,
            'text_rank': None,
            'rrf_score': 1.0 / (60 + idx),
            'retrieval_weight': 1.0,
        }

    def test_candidates_from_hybrid_produce_scored_results(self, db):
        """When hybrid retrieve returns candidates they should be returned scored."""
        svc = EpisodicService(get_shared_db_service(), config={})
        candidates = [self._make_candidate(i) for i in range(3)]

        with patch.object(svc, '_generate_embedding', return_value=[0.0] * 256), \
             patch.object(svc, '_hybrid_retrieve', return_value=candidates), \
             patch.object(svc, '_apply_reconsolidation'):
            result = svc.retrieve_episodes(query_text='something happened')

        assert len(result) == 3
        # Every result should have a composite_score attached by the ranker
        for ep in result:
            assert 'composite_score' in ep
            assert isinstance(ep['composite_score'], float)

    def test_empty_query_does_not_crash(self, db):
        """Empty query string should not raise; returns [] when hybrid finds nothing."""
        svc = EpisodicService(get_shared_db_service(), config={})

        with patch.object(svc, '_generate_embedding', return_value=[0.0] * 256), \
             patch.object(svc, '_hybrid_retrieve', return_value=[]):
            result = svc.retrieve_episodes(query_text='')

        assert result == []

    def test_results_ordered_by_composite_score_descending(self, db):
        """retrieve_episodes should return episodes sorted highest composite_score first."""
        svc = EpisodicService(get_shared_db_service(), config={})
        candidates = [self._make_candidate(i) for i in range(5)]

        with patch.object(svc, '_generate_embedding', return_value=[0.0] * 256), \
             patch.object(svc, '_hybrid_retrieve', return_value=candidates), \
             patch.object(svc, '_apply_reconsolidation'):
            result = svc.retrieve_episodes(query_text='test query')

        scores = [ep['composite_score'] for ep in result]
        assert scores == sorted(scores, reverse=True)

    def test_embedding_exception_returns_empty_list(self, db):
        """Embedding failure should be caught gracefully and return []."""
        svc = EpisodicService(get_shared_db_service(), config={})

        with patch.object(svc, '_generate_embedding', side_effect=RuntimeError('gpu error')):
            result = svc.retrieve_episodes(query_text='anything')

        assert result == []

    def test_single_candidate_returned_as_single_item_list(self, db):
        """A single candidate should be wrapped in a list and returned."""
        svc = EpisodicService(get_shared_db_service(), config={})
        candidates = [self._make_candidate(0)]

        with patch.object(svc, '_generate_embedding', return_value=[0.0] * 256), \
             patch.object(svc, '_hybrid_retrieve', return_value=candidates), \
             patch.object(svc, '_apply_reconsolidation'):
            result = svc.retrieve_episodes(query_text='single')

        assert len(result) == 1
        assert result[0]['gist'] == 'gist-0'

    def test_query_analysis_entities_fed_into_scoring(self, db):
        """Entity overlap from _analyze_query should influence composite_score."""
        svc = EpisodicService(get_shared_db_service(), config={})

        # candidate with entity matching the query
        candidate_match = self._make_candidate(0)
        candidate_match['entities'] = '["Alice"]'
        candidate_match['gist'] = 'Alice did something'

        # candidate without matching entity
        candidate_no_match = self._make_candidate(1)
        candidate_no_match['entities'] = '["Bob"]'
        candidate_no_match['gist'] = 'Bob did something'

        mock_analysis = {
            'entities': ['Alice'],
            'keywords': ['Alice', 'did', 'something'],
            'noun_phrases': ['Alice'],
        }

        with patch.object(svc, '_generate_embedding', return_value=[0.0] * 256), \
             patch.object(svc, '_hybrid_retrieve', return_value=[candidate_match, candidate_no_match]), \
             patch.object(svc, '_analyze_query', return_value=mock_analysis), \
             patch.object(svc, '_apply_reconsolidation'):
            result = svc.retrieve_episodes(query_text='Alice did something')

        # Alice-matching episode should outscore Bob episode
        alice_score = next(ep['composite_score'] for ep in result if ep['gist'] == 'Alice did something')
        bob_score = next(ep['composite_score'] for ep in result if ep['gist'] == 'Bob did something')
        assert alice_score > bob_score


# ── _get_consolidated_date_range ──────────────────────────────────────────────

class TestGetConsolidatedDateRange:
    """Tests for EpisodicService._get_consolidated_date_range."""

    def test_empty_source_ids_returns_fallback_from_episode(self, db):
        """When source_ids is empty the episode's own created_at date is used."""
        svc = EpisodicService(get_shared_db_service(), config={})
        episode = {'created_at': '2026-03-10T08:00:00+00:00'}
        result = svc._get_consolidated_date_range([], episode)
        assert result == '2026-03-10'

    def test_source_ids_with_matching_rows_returns_range(self, db):
        """Real source IDs in DB → 'YYYY-MM-DD to YYYY-MM-DD'."""
        svc = EpisodicService(get_shared_db_service(), config={})
        import uuid
        id_a = str(uuid.uuid4())
        id_b = str(uuid.uuid4())
        db_conn = get_shared_db_service()
        with db_conn.connection() as conn:
            conn.execute(
                "INSERT INTO episodes (id, intent, context, action, emotion, outcome, gist, salience, channel, created_at) "
                "VALUES (?, '{}', '', '', '{}', '', '', 5, 'test', ?)",
                (id_a, '2026-02-01T00:00:00+00:00')
            )
            conn.execute(
                "INSERT INTO episodes (id, intent, context, action, emotion, outcome, gist, salience, channel, created_at) "
                "VALUES (?, '{}', '', '', '{}', '', '', 5, 'test', ?)",
                (id_b, '2026-02-28T00:00:00+00:00')
            )
        episode = {'created_at': '2026-02-28T00:00:00+00:00'}
        result = svc._get_consolidated_date_range([id_a, id_b], episode)
        assert result == '2026-02-01 to 2026-02-28'

    def test_source_ids_not_in_db_returns_fallback(self, db):
        """If source IDs don't exist in DB, fallback to episode created_at."""
        svc = EpisodicService(get_shared_db_service(), config={})
        episode = {'created_at': '2026-01-05T00:00:00+00:00'}
        result = svc._get_consolidated_date_range(['nonexistent-id-1', 'nonexistent-id-2'], episode)
        assert result == '2026-01-05'

    def test_malformed_episode_created_at_falls_back_to_string_slice_on_parse_exception(self, db):
        """When parse_utc raises, the except branch falls back to str(created_at)[:10]."""
        svc = EpisodicService(get_shared_db_service(), config={})
        episode = {'created_at': '2026-06-20-extra-stuff'}
        with patch('services.episodic_service.parse_utc', side_effect=ValueError('bad')):
            result = svc._get_consolidated_date_range([], episode)
        # str('2026-06-20-extra-stuff')[:10] == '2026-06-20'
        assert result == '2026-06-20'
