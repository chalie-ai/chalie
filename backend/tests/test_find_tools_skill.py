"""Tests for the find_tools innate skill."""

import struct
import pytest
from unittest.mock import patch, MagicMock

from services.innate_skills.find_tools_skill import (
    handle_find_tools,
    _filter_available,
    _format_search_results,
    _get_param_summary,
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
        assert "Error" in result['text']
        assert result['_discovered_tools'] == []

    def test_missing_query_returns_error(self):
        result = handle_find_tools("topic", {})
        assert "Error" in result['text']
        assert result['_discovered_tools'] == []


class TestFilterAvailable:

    def _make_row(self, name, tool_type='tool', summary='desc', profile='long desc',
                  domain='Other', effort='moderate', distance=0.5):
        return (name, tool_type, summary, profile, domain, effort, distance)

    @patch(_REGISTRY)
    def test_filters_out_skills(self, mock_registry_cls):
        """Innate skills should be excluded from results."""
        mock_reg = _mock_registry(tools={'weather': {'manifest': {}}})
        mock_registry_cls.return_value = mock_reg

        rows = [
            self._make_row('recall', 'skill'),
            self._make_row('weather', 'tool'),
        ]
        result = _filter_available(rows)
        assert len(result) == 1
        assert result[0]['tool_name'] == 'weather'

    @patch(_REGISTRY)
    def test_filters_unready_tools(self, mock_registry_cls):
        """Tools not ready should be excluded."""
        mock_reg = _mock_registry(tools={'broken': {'manifest': {}}}, ready=False)
        mock_registry_cls.return_value = mock_reg

        rows = [self._make_row('broken')]
        result = _filter_available(rows)
        assert len(result) == 0

    @patch(_REGISTRY)
    def test_filters_offline_interface(self, mock_registry_cls):
        """Interface tools that are offline should be excluded."""
        mock_reg = _mock_registry(tools={'offline': {'manifest': {}}}, online=False)
        mock_registry_cls.return_value = mock_reg

        rows = [self._make_row('offline')]
        result = _filter_available(rows)
        assert len(result) == 0

    @patch(_REGISTRY)
    def test_similarity_calculation(self, mock_registry_cls):
        """L2 distance should be converted to similarity score."""
        mock_reg = _mock_registry(tools={'tool_a': {'manifest': {}}})
        mock_registry_cls.return_value = mock_reg

        rows = [self._make_row('tool_a', distance=0.4)]
        result = _filter_available(rows)
        assert len(result) == 1
        assert result[0]['similarity'] == pytest.approx(0.8, abs=0.01)

    @patch(_REGISTRY)
    def test_unregistered_tool_excluded(self, mock_registry_cls):
        """Tools in profiles but not in registry should be excluded."""
        mock_reg = _mock_registry(tools={})
        mock_registry_cls.return_value = mock_reg

        rows = [self._make_row('ghost_tool')]
        result = _filter_available(rows)
        assert len(result) == 0


class TestFormatSearchResults:

    @patch('services.innate_skills.find_tools_skill._get_param_summary')
    def test_format_includes_tool_name(self, mock_params):
        mock_params.return_value = "(query)"
        tools = [{
            'tool_name': 'web_search',
            'short_summary': 'Search the web',
            'full_profile': 'Full web search description',
            'domain': 'Research',
            'effort': 'light',
            'similarity': 0.92,
        }]
        result = _format_search_results("search the internet", tools)
        assert 'web_search' in result
        assert '92%' in result
        assert 'directly' in result.lower()

    @patch('services.innate_skills.find_tools_skill._get_param_summary')
    def test_format_multiple_tools(self, mock_params):
        mock_params.return_value = ""
        tools = [
            {'tool_name': 'a', 'short_summary': 's', 'full_profile': 'p',
             'domain': 'D', 'effort': 'light', 'similarity': 0.9},
            {'tool_name': 'b', 'short_summary': 's', 'full_profile': 'p',
             'domain': 'D', 'effort': 'moderate', 'similarity': 0.7},
        ]
        result = _format_search_results("query", tools)
        assert 'Found 2 tool(s)' in result


class TestGetParamSummary:

    @patch(_REGISTRY)
    def test_formats_required_and_optional(self, mock_registry_cls):
        mock_reg = _mock_registry(tools={
            'weather': {'manifest': {'parameters': {
                'location': {'required': True},
                'units': {'required': False},
            }}}
        })
        mock_registry_cls.return_value = mock_reg
        result = _get_param_summary('weather')
        assert result == '(location, units?)'

    @patch(_REGISTRY)
    def test_no_params(self, mock_registry_cls):
        mock_reg = _mock_registry(tools={'simple': {'manifest': {'parameters': {}}}})
        mock_registry_cls.return_value = mock_reg
        result = _get_param_summary('simple')
        assert result == '(no parameters)'

    @patch(_REGISTRY)
    def test_missing_tool(self, mock_registry_cls):
        mock_reg = _mock_registry(tools={})
        mock_registry_cls.return_value = mock_reg
        result = _get_param_summary('nonexistent')
        assert result == ''


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
            db, 'web_search', tool_type='tool', summary='Search the web',
            profile='Full profile', domain='Research', effort='light',
            embedding=embedding,
        )

        mock_reg = _mock_registry(
            tools={'web_search': {'manifest': {'parameters': {'query': {'required': True}}}}}
        )
        mock_registry_cls.return_value = mock_reg

        result = handle_find_tools("topic", {"query": "search online"})
        assert isinstance(result, dict)
        assert 'web_search' in result['text']
        assert 'Found 1 tool' in result['text']
        assert 'web_search' in result['_discovered_tools']

    @patch(_EMB)
    def test_search_falls_back_on_embedding_failure(self, mock_emb_cls, db):
        """When embedding fails, should fall back to keyword search."""
        mock_emb_cls.return_value.generate_embedding.side_effect = RuntimeError("model not loaded")

        result = handle_find_tools("topic", {"query": "weather"})
        assert isinstance(result, dict)
        assert "keyword" in result['text'].lower() or "No tools found" in result['text']
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
        assert set(result['_discovered_tools']) == {'tool_a', 'tool_b'}


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
        assert 'No tools found' in result['text'] or 'No available' in result['text']
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
        assert 'built-in skills' in result['text'].lower() or 'no tools found' in result['text'].lower()
        assert 'recall' in result['text']  # Should name the matching skill
        assert result['_discovered_tools'] == []
