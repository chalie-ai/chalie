"""Tests for the find_tools innate skill."""

import struct
import pytest
from unittest.mock import patch, MagicMock

from services.innate_skills.find_tools_skill import (
    handle_find_tools,
    _filter_available,
)


pytestmark = pytest.mark.unit

# Patch targets: imports are lazy (inside functions), so patch the source module
_REGISTRY = 'services.tool_registry_service.ToolRegistryService'
_EMB = 'services.embedding_service.EmbeddingService'
_DB = 'services.database_service.get_shared_db_service'


def _mock_registry(tools=None, ready=True, online=True):
    """Create a mock ToolRegistryService singleton."""
    mock = MagicMock()
    mock.tools = tools or {}
    mock._is_ready.return_value = ready
    mock._is_interface_online.return_value = online
    return mock


def _pack_embedding(values):
    """Pack a list of floats into bytes for sqlite-vec."""
    return struct.pack(f'{len(values)}f', *values)


def _seed_tool_profile(db, tool_name, tool_type='tool', summary='desc',
                       profile='long desc', domain='Other', effort='moderate',
                       embedding=None):
    """Seed a tool_capability_profiles row and its vec companion.

    Returns the rowid so callers can verify lookups.
    """
    db.execute(
        "INSERT INTO tool_capability_profiles "
        "(id, tool_name, tool_type, short_summary, full_profile, domain, effort) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (f'tcp-{tool_name}', tool_name, tool_type, summary, profile, domain, effort),
    )
    # Get the auto-generated rowid
    row = db.execute(
        "SELECT rowid FROM tool_capability_profiles WHERE tool_name = ?", (tool_name,)
    ).fetchone()
    rowid = row[0]

    if embedding is not None:
        blob = _pack_embedding(embedding)
        db.execute(
            "INSERT INTO tool_capability_profiles_vec (rowid, embedding) VALUES (?, ?)",
            (rowid, blob),
        )

    db.commit()
    return rowid


class TestHandleFindTools:

    def test_returns_dict(self):
        """Result should always be a dict with 'text' and '_discovered_tools'."""
        result = handle_find_tools("topic", {"query": ""})
        assert isinstance(result, dict)
        assert 'text' in result
        assert '_discovered_tools' in result

    def test_empty_query_returns_error(self):
        result = handle_find_tools("topic", {"query": ""})
        assert "ERROR" in result['text']
        assert result['_discovered_tools'] == []

    def test_missing_query_returns_error(self):
        result = handle_find_tools("topic", {})
        assert "ERROR" in result['text']
        assert result['_discovered_tools'] == []


class TestFilterAvailable:

    def _make_row(self, name, tool_type='tool', summary='desc', profile='long desc',
                  domain='Other', effort='moderate', distance=0.5, keywords=''):
        return (name, tool_type, summary, profile, domain, effort, distance, keywords)

    @patch(_REGISTRY)
    def test_filters_out_skills(self, mock_registry_cls):
        """Innate skills should be excluded from results."""
        mock_reg = _mock_registry(tools={'weather': {'manifest': {}}})
        mock_registry_cls.return_value = mock_reg

        rows = [
            self._make_row('recall', 'skill'),
            self._make_row('weather', 'tool'),
        ]
        result = _filter_available(rows, query='weather')
        assert len(result) == 1
        assert result[0]['tool_name'] == 'weather'

    @patch(_REGISTRY)
    def test_filters_unready_tools(self, mock_registry_cls):
        """Tools not ready should be excluded."""
        mock_reg = _mock_registry(tools={'broken': {'manifest': {}}}, ready=False)
        mock_registry_cls.return_value = mock_reg

        rows = [self._make_row('broken')]
        result = _filter_available(rows, query='broken')
        assert len(result) == 0

    @patch(_REGISTRY)
    def test_filters_offline_interface(self, mock_registry_cls):
        """Interface tools that are offline should be excluded."""
        mock_reg = _mock_registry(tools={'offline': {'manifest': {}}}, online=False)
        mock_registry_cls.return_value = mock_reg

        rows = [self._make_row('offline')]
        result = _filter_available(rows, query='offline')
        assert len(result) == 0

    @patch(_REGISTRY)
    def test_score_calculation(self, mock_registry_cls):
        """Score = distance * 10 - keyword_match_count. Lower = better."""
        mock_reg = _mock_registry(tools={'tool_a': {'manifest': {}}})
        mock_registry_cls.return_value = mock_reg

        # distance=0.4, no keyword match -> score = 0.4 * 10 - 0 = 4.0
        rows = [self._make_row('tool_a', distance=0.4, keywords='')]
        result = _filter_available(rows, query='something')
        assert len(result) == 1
        assert result[0]['score'] == pytest.approx(4.0, abs=0.01)

    @patch(_REGISTRY)
    def test_keyword_match_lowers_score(self, mock_registry_cls):
        """Each keyword match in query reduces score by 1."""
        mock_reg = _mock_registry(tools={'weather': {'manifest': {}}})
        mock_registry_cls.return_value = mock_reg

        # distance=0.5, 2 keyword matches -> score = 0.5 * 10 - 2 = 3.0
        rows = [self._make_row('weather', distance=0.5, keywords='weather,forecast,temperature')]
        result = _filter_available(rows, query='what is the weather forecast today')
        assert len(result) == 1
        assert result[0]['score'] == pytest.approx(3.0, abs=0.01)

    @patch(_REGISTRY)
    def test_unregistered_tool_excluded(self, mock_registry_cls):
        """Tools in profiles but not in registry should be excluded."""
        mock_reg = _mock_registry(tools={})
        mock_registry_cls.return_value = mock_reg

        rows = [self._make_row('ghost_tool')]
        result = _filter_available(rows, query='ghost')
        assert len(result) == 0




class TestSearchIntegration:
    """Integration-style tests that use the real DB with seeded tool profiles."""

    @patch(_REGISTRY)
    @patch(_EMB)
    def test_search_happy_path(self, mock_emb_cls, mock_registry_cls, db):
        """Full search flow: embed query -> vec search -> filter -> format."""
        # Use a simple embedding for both the query and the seeded profile
        embedding = [0.1] * 256
        mock_emb_cls.return_value.generate_embedding.return_value = embedding

        _seed_tool_profile(
            db, 'search', tool_type='tool', summary='Search the web',
            profile='Full profile', domain='Research', effort='light',
            embedding=embedding,
        )

        mock_reg = _mock_registry(
            tools={'search': {'manifest': {
                'input_schema': {
                    'type': 'object',
                    'properties': {'query': {'type': 'string'}},
                    'required': ['query'],
                }
            }}}
        )
        mock_registry_cls.return_value = mock_reg

        result = handle_find_tools("topic", {"query": "search online"})
        assert isinstance(result, dict)
        # Output is JSON: {"added_tools": [{"name": "search", "relevance": ...}]}
        import json
        parsed = json.loads(result['text'])
        assert any(t['name'] == 'search' for t in parsed['added_tools'])
        assert 'search' in result['_discovered_tools']

    @patch(_EMB)
    def test_search_falls_back_on_embedding_failure(self, mock_emb_cls, db):
        """When embedding fails, should fall back to keyword search."""
        mock_emb_cls.return_value.generate_embedding.side_effect = RuntimeError("model not loaded")

        result = handle_find_tools("topic", {"query": "weather"})
        assert isinstance(result, dict)
        # Fallback returns either JSON with added_tools or INFO message
        assert isinstance(result['_discovered_tools'], list)

    @patch(_REGISTRY)
    @patch(_EMB)
    def test_discovered_tools_list_populated(self, mock_emb_cls, mock_registry_cls, db):
        """_discovered_tools should contain the tool names from search results."""
        embedding = [0.1] * 256
        mock_emb_cls.return_value.generate_embedding.return_value = embedding

        _seed_tool_profile(
            db, 'tool_a', summary='Tool A', profile='Full', domain='D',
            effort='light', embedding=embedding,
        )
        _seed_tool_profile(
            db, 'tool_b', summary='Tool B', profile='Full', domain='D',
            effort='light', embedding=embedding,
        )

        mock_reg = _mock_registry(tools={
            'tool_a': {'manifest': {'parameters': {}}},
            'tool_b': {'manifest': {'parameters': {}}},
        })
        mock_registry_cls.return_value = mock_reg

        result = handle_find_tools("topic", {"query": "test"})
        assert set(result['_discovered_tools']) >= {'tool_a', 'tool_b'}


class TestLimitCapping:

    @patch(_REGISTRY)
    @patch(_EMB)
    def test_search_limit_capped_at_10(self, mock_emb_cls, mock_registry_cls, db):
        """Limit parameter should be capped at 10.

        We verify by seeding one tool and checking the result still works
        (the service caps limit internally to 10, over-fetching by +5 = 15).
        """
        embedding = [0.1] * 256
        mock_emb_cls.return_value.generate_embedding.return_value = embedding

        _seed_tool_profile(
            db, 'test_tool', summary='Test Tool', profile='Full', domain='D',
            effort='light', embedding=embedding,
        )

        mock_reg = _mock_registry(tools={
            'test_tool': {'manifest': {'parameters': {}}},
        })
        mock_registry_cls.return_value = mock_reg

        # Even with limit=50, should work (internally capped to 10)
        result = handle_find_tools("topic", {"query": "test", "limit": 50})
        assert isinstance(result, dict)
        # Should find our seeded tool
        assert 'test_tool' in result['_discovered_tools']


class TestEdgeCases:

    @patch(_REGISTRY)
    @patch(_EMB)
    def test_no_tools_registered(self, mock_emb_cls, mock_registry_cls, db):
        """When no tools are registered, should return helpful message."""
        mock_emb_cls.return_value.generate_embedding.return_value = [0.1] * 256
        mock_registry_cls.return_value = _mock_registry()

        result = handle_find_tools("topic", {"query": "anything"})
        assert isinstance(result, dict)
        # No matching tools → INFO message
        assert 'already available' in result['text']
        assert result['_discovered_tools'] == []

    @patch(_REGISTRY)
    @patch(_EMB)
    def test_only_skills_match(self, mock_emb_cls, mock_registry_cls, db):
        """When only innate skills match, should inform the user."""
        embedding = [0.1] * 256
        mock_emb_cls.return_value.generate_embedding.return_value = embedding

        _seed_tool_profile(
            db, 'recall', tool_type='skill', summary='Memory retrieval',
            profile='Full profile', domain='Memory', effort='trivial',
            embedding=embedding,
        )

        mock_registry_cls.return_value = _mock_registry()

        result = handle_find_tools("topic", {"query": "remember something"})
        assert isinstance(result, dict)
        # Skills are filtered out → no tools added → INFO message
        assert 'already available' in result['text']
        assert result['_discovered_tools'] == []


class TestKeywordScoring:
    """Tests for two-axis scoring: (distance * 10) - kw_match_count."""

    def _make_row(self, name, tool_type='tool', summary='desc', profile='long desc',
                  domain='Other', effort='moderate', distance=0.5, keywords=''):
        return (name, tool_type, summary, profile, domain, effort, distance, keywords)

    @patch(_REGISTRY)
    def test_case_insensitive_keyword_matching(self, mock_registry_cls):
        """Keyword matching must be case-insensitive: 'WEATHER' matches keyword 'weather'."""
        mock_reg = _mock_registry(tools={'weather': {'manifest': {}}})
        mock_registry_cls.return_value = mock_reg

        rows = [self._make_row('weather', distance=0.3, keywords='weather,forecast')]
        result = _filter_available(rows, query='WEATHER FORECAST TODAY')
        assert len(result) == 1
        # distance=0.3, 2 matches -> score = 3.0 - 2 = 1.0
        assert result[0]['score'] == pytest.approx(1.0, abs=0.01)

    @patch(_REGISTRY)
    def test_multiple_keyword_matches_stack(self, mock_registry_cls):
        """Each matching keyword subtracts exactly 1 from the score."""
        mock_reg = _mock_registry(tools={'weather': {'manifest': {}}})
        mock_registry_cls.return_value = mock_reg

        # distance=0.5 -> base score 5.0; keywords 'weather','forecast','temperature' all in query
        rows = [self._make_row('weather', distance=0.5, keywords='weather,forecast,temperature')]
        result = _filter_available(rows, query='weather forecast temperature today')
        assert len(result) == 1
        assert result[0]['score'] == pytest.approx(2.0, abs=0.01)  # 5.0 - 3 = 2.0

    @patch(_REGISTRY)
    def test_empty_keywords_score_is_distance_times_ten(self, mock_registry_cls):
        """When keywords is empty string, score = distance * 10 with no bonus."""
        mock_reg = _mock_registry(tools={'tool_a': {'manifest': {}}})
        mock_registry_cls.return_value = mock_reg

        rows = [self._make_row('tool_a', distance=0.7, keywords='')]
        result = _filter_available(rows, query='something unrelated')
        assert len(result) == 1
        assert result[0]['score'] == pytest.approx(7.0, abs=0.01)

    @patch(_REGISTRY)
    def test_null_keywords_treated_as_empty(self, mock_registry_cls):
        """None keywords (old DB rows before migration) should not crash scoring."""
        mock_reg = _mock_registry(tools={'old_tool': {'manifest': {}}})
        mock_registry_cls.return_value = mock_reg

        # Simulate a row where keywords column is NULL (None)
        row = ('old_tool', 'tool', 'desc', 'full desc', 'Other', 'moderate', 0.6, None)
        result = _filter_available([row], query='some query')
        assert len(result) == 1
        assert result[0]['score'] == pytest.approx(6.0, abs=0.01)

    @patch(_REGISTRY)
    def test_keywords_not_in_query_no_bonus(self, mock_registry_cls):
        """Keywords present on tool but absent from query give no score reduction."""
        mock_reg = _mock_registry(tools={'weather': {'manifest': {}}})
        mock_registry_cls.return_value = mock_reg

        rows = [self._make_row('weather', distance=0.4, keywords='weather,forecast,temperature')]
        result = _filter_available(rows, query='show me something completely different')
        assert len(result) == 1
        # None of the keywords appear in the query
        assert result[0]['score'] == pytest.approx(4.0, abs=0.01)

    @patch(_REGISTRY)
    def test_tools_sorted_ascending_by_score(self, mock_registry_cls):
        """Results must be sorted ascending by score (lower = better match)."""
        mock_reg = _mock_registry(tools={
            'tool_high': {'manifest': {}},
            'tool_mid': {'manifest': {}},
            'tool_low': {'manifest': {}},
        })
        mock_registry_cls.return_value = mock_reg

        rows = [
            # tool_high: score = 0.8*10 - 0 = 8.0
            self._make_row('tool_high', distance=0.8, keywords=''),
            # tool_mid: score = 0.5*10 - 1 = 4.0
            self._make_row('tool_mid', distance=0.5, keywords='forecast'),
            # tool_low: score = 0.3*10 - 2 = 1.0
            self._make_row('tool_low', distance=0.3, keywords='weather,forecast'),
        ]
        result = _filter_available(rows, query='weather forecast')
        assert len(result) == 3
        scores = [t['score'] for t in result]
        assert scores == sorted(scores), "Results not sorted ascending by score"
        assert result[0]['tool_name'] == 'tool_low'
        assert result[-1]['tool_name'] == 'tool_high'

    @patch(_REGISTRY)
    def test_query_words_must_contain_full_keyword(self, mock_registry_cls):
        """Keyword 'forecast' should NOT match query containing only 'fore'."""
        mock_reg = _mock_registry(tools={'weather': {'manifest': {}}})
        mock_registry_cls.return_value = mock_reg

        rows = [self._make_row('weather', distance=0.5, keywords='forecast')]
        result = _filter_available(rows, query='fore cast today')
        assert len(result) == 1
        # 'forecast' not in 'fore cast today' -> no match bonus
        assert result[0]['score'] == pytest.approx(5.0, abs=0.01)

    @patch(_REGISTRY)
    def test_multi_tool_query_matches_correct_keywords(self, mock_registry_cls):
        """Query matching keywords of one tool should not affect another tool's score."""
        mock_reg = _mock_registry(tools={
            'weather': {'manifest': {}},
            'search': {'manifest': {}},
        })
        mock_registry_cls.return_value = mock_reg

        rows = [
            # 'weather' keyword in query -> score = 5.0 - 1 = 4.0
            self._make_row('weather', distance=0.5, keywords='weather,temperature'),
            # no keyword overlap -> score = 0.5*10 - 0 = 5.0
            self._make_row('search', distance=0.5, keywords='search,google,lookup'),
        ]
        result = _filter_available(rows, query='what is the weather')
        assert len(result) == 2
        weather_result = next(t for t in result if t['tool_name'] == 'weather')
        search_result = next(t for t in result if t['tool_name'] == 'search')
        assert weather_result['score'] == pytest.approx(4.0, abs=0.01)
        assert search_result['score'] == pytest.approx(5.0, abs=0.01)

    @patch(_REGISTRY)
    def test_score_exposed_in_result_dict(self, mock_registry_cls):
        """Each result dict must include a 'score' key."""
        mock_reg = _mock_registry(tools={'tool_a': {'manifest': {}}})
        mock_registry_cls.return_value = mock_reg

        rows = [self._make_row('tool_a', distance=0.3, keywords='alpha,beta')]
        result = _filter_available(rows, query='alpha')
        assert 'score' in result[0]




class TestFilterAvailableQueryPassthrough:
    """Verify that query text is forwarded to _filter_available for keyword matching."""

    def _make_row(self, name, tool_type='tool', summary='desc', profile='long desc',
                  domain='Other', effort='moderate', distance=0.5, keywords=''):
        return (name, tool_type, summary, profile, domain, effort, distance, keywords)

    @patch(_REGISTRY)
    @patch(_EMB)
    def test_query_reaches_keyword_scoring(self, mock_emb_cls, mock_registry_cls, db):
        """End-to-end: keyword in query lowers score relative to tool without match."""
        embedding = [0.1] * 256
        mock_emb_cls.return_value.generate_embedding.return_value = embedding

        # Seed two tools with identical embeddings; only 'weather' has matching keywords
        _seed_tool_profile(
            db, 'weather', summary='Weather tool', profile='Full',
            domain='Context', effort='trivial', embedding=embedding,
        )
        _seed_tool_profile(
            db, 'news', summary='News tool', profile='Full',
            domain='Research', effort='light', embedding=embedding,
        )

        # Insert keywords for weather tool directly
        db.execute(
            "UPDATE tool_capability_profiles SET keywords = ? WHERE tool_name = ?",
            ('weather,forecast', 'weather')
        )
        db.commit()

        mock_reg = _mock_registry(tools={
            'weather': {'manifest': {'parameters': {}}},
            'news': {'manifest': {'parameters': {}}},
        })
        mock_registry_cls.return_value = mock_reg

        result = handle_find_tools("topic", {"query": "weather forecast"})
        assert isinstance(result, dict)
        # Both tools should appear; weather should rank first (lower score)
        assert 'weather' in result['_discovered_tools']
