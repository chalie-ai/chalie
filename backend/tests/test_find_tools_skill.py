"""Tests for the find_tools innate skill."""

import json
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


def _mock_db_with_rows(rows):
    """Create a mock DB service that returns rows from a cursor.

    Attaches mock_cursor as ._test_cursor for easy assertion access.
    """
    mock_db = MagicMock()
    mock_conn = MagicMock()
    mock_db.connection.return_value.__enter__ = lambda s: mock_conn
    mock_db.connection.return_value.__exit__ = MagicMock(return_value=False)
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.fetchall.return_value = rows
    mock_db._test_cursor = mock_cursor
    return mock_db


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
    """Integration-style tests that mock the embedding and DB layers."""

    @patch(_REGISTRY)
    @patch(_DB)
    @patch(_EMB)
    def test_search_happy_path(self, mock_emb_cls, mock_db_fn, mock_registry_cls):
        """Full search flow: embed query -> vec search -> filter -> format."""
        mock_emb_cls.return_value.generate_embedding.return_value = [0.1] * 768

        mock_db_fn.return_value = _mock_db_with_rows([
            ('web_search', 'tool', 'Search the web', 'Full profile', 'Research', 'light', 0.3),
        ])

        mock_reg = _mock_registry(
            tools={'web_search': {'manifest': {'parameters': {'query': {'required': True}}}}}
        )
        mock_registry_cls.return_value = mock_reg

        result = handle_find_tools("topic", {"query": "search online"})
        assert isinstance(result, dict)
        assert 'web_search' in result['text']
        assert 'Found 1 tool' in result['text']
        assert 'web_search' in result['_discovered_tools']

    @patch(_DB)
    @patch(_EMB)
    def test_search_falls_back_on_embedding_failure(self, mock_emb_cls, mock_db_fn):
        """When embedding fails, should fall back to keyword search."""
        mock_emb_cls.return_value.generate_embedding.side_effect = RuntimeError("model not loaded")
        mock_db_fn.return_value = _mock_db_with_rows([])

        result = handle_find_tools("topic", {"query": "weather"})
        assert isinstance(result, dict)
        assert "keyword" in result['text'].lower() or "No tools found" in result['text']
        assert isinstance(result['_discovered_tools'], list)

    @patch(_REGISTRY)
    @patch(_DB)
    @patch(_EMB)
    def test_discovered_tools_list_populated(self, mock_emb_cls, mock_db_fn, mock_registry_cls):
        """_discovered_tools should contain the tool names from search results."""
        mock_emb_cls.return_value.generate_embedding.return_value = [0.1] * 768
        mock_db_fn.return_value = _mock_db_with_rows([
            ('tool_a', 'tool', 'Tool A', 'Full', 'D', 'light', 0.2),
            ('tool_b', 'tool', 'Tool B', 'Full', 'D', 'light', 0.4),
        ])
        mock_reg = _mock_registry(tools={
            'tool_a': {'manifest': {'parameters': {}}},
            'tool_b': {'manifest': {'parameters': {}}},
        })
        mock_registry_cls.return_value = mock_reg

        result = handle_find_tools("topic", {"query": "test"})
        assert set(result['_discovered_tools']) == {'tool_a', 'tool_b'}


class TestLimitCapping:

    @patch(_REGISTRY)
    @patch(_DB)
    @patch(_EMB)
    def test_search_limit_capped_at_10(self, mock_emb_cls, mock_db_fn, mock_registry_cls):
        """Limit parameter should be capped at 10."""
        mock_emb_cls.return_value.generate_embedding.return_value = [0.1] * 768
        mock_db = _mock_db_with_rows([])
        mock_db_fn.return_value = mock_db
        mock_registry_cls.return_value = _mock_registry()

        handle_find_tools("topic", {"query": "test", "limit": 50})

        # k parameter should be 10 + 5 = 15 (not 50 + 5 = 55)
        call_args = mock_db._test_cursor.execute.call_args
        assert call_args[0][1][1] == 15


class TestEdgeCases:

    @patch(_REGISTRY)
    @patch(_DB)
    @patch(_EMB)
    def test_no_tools_registered(self, mock_emb_cls, mock_db_fn, mock_registry_cls):
        """When no tools are registered, should return helpful message."""
        mock_emb_cls.return_value.generate_embedding.return_value = [0.1] * 768
        mock_db_fn.return_value = _mock_db_with_rows([])
        mock_registry_cls.return_value = _mock_registry()

        result = handle_find_tools("topic", {"query": "anything"})
        assert isinstance(result, dict)
        assert 'No tools found' in result['text'] or 'No available' in result['text']
        assert result['_discovered_tools'] == []

    @patch(_REGISTRY)
    @patch(_DB)
    @patch(_EMB)
    def test_only_skills_match(self, mock_emb_cls, mock_db_fn, mock_registry_cls):
        """When only innate skills match, should inform the user."""
        mock_emb_cls.return_value.generate_embedding.return_value = [0.1] * 768
        mock_db_fn.return_value = _mock_db_with_rows([
            ('recall', 'skill', 'Memory retrieval', 'Full profile', 'Memory', 'trivial', 0.2),
        ])
        mock_registry_cls.return_value = _mock_registry()

        result = handle_find_tools("topic", {"query": "remember something"})
        assert isinstance(result, dict)
        assert 'innate skills' in result['text'].lower() or 'No available' in result['text']
        assert result['_discovered_tools'] == []
