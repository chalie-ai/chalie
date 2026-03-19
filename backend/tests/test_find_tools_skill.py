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

    def test_default_action_is_search(self):
        """With no action specified, should default to search."""
        result = handle_find_tools("topic", {"query": ""})
        assert "[FIND_TOOLS]" in result
        assert "Error" in result

    def test_unknown_action(self):
        result = handle_find_tools("topic", {"action": "bogus"})
        assert "Unknown action" in result

    def test_search_requires_query(self):
        result = handle_find_tools("topic", {"action": "search", "query": ""})
        assert "Error" in result
        assert "query" in result.lower()

    def test_details_requires_tool_name(self):
        result = handle_find_tools("topic", {"action": "details", "tool_name": ""})
        assert "Error" in result
        assert "tool_name" in result


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
        assert 'find_tools' in result

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
        """Full search flow: embed query → vec search → filter → format."""
        mock_emb_cls.return_value.generate_embedding.return_value = [0.1] * 768

        mock_db_fn.return_value = _mock_db_with_rows([
            ('web_search', 'tool', 'Search the web', 'Full profile', 'Research', 'light', 0.3),
        ])

        mock_reg = _mock_registry(
            tools={'web_search': {'manifest': {'parameters': {'query': {'required': True}}}}}
        )
        mock_registry_cls.return_value = mock_reg

        result = handle_find_tools("topic", {"action": "search", "query": "search online"})
        assert 'web_search' in result
        assert 'Found 1 tool' in result

    @patch(_DB)
    @patch(_EMB)
    def test_search_falls_back_on_embedding_failure(self, mock_emb_cls, mock_db_fn):
        """When embedding fails, should fall back to keyword search."""
        mock_emb_cls.return_value.generate_embedding.side_effect = RuntimeError("model not loaded")
        mock_db_fn.return_value = _mock_db_with_rows([])

        result = handle_find_tools("topic", {"action": "search", "query": "weather"})
        assert "keyword" in result.lower() or "No tools found" in result

    @patch('services.innate_skills.find_tools_skill._fetch_profile')
    @patch('services.innate_skills.find_tools_skill._fetch_manifest')
    def test_details_with_profile_and_manifest(self, mock_manifest, mock_profile):
        """Details should combine profile and manifest info."""
        mock_profile.return_value = {
            'full_profile': 'Searches the web using multiple engines',
            'short_summary': 'Web search',
            'usage_scenarios': json.dumps(['Finding news', 'Research']),
            'anti_scenarios': json.dumps(['Local file search']),
        }
        mock_manifest.return_value = {
            'parameters': {
                'query': {'required': True, 'type': 'string', 'description': 'Search query'},
                'region': {'required': False, 'type': 'string', 'description': 'Region code'},
            }
        }

        result = handle_find_tools("topic", {"action": "details", "tool_name": "web_search"})
        assert 'web_search' in result
        assert 'query' in result
        assert 'region' in result
        assert 'required' in result

    @patch('services.innate_skills.find_tools_skill._fetch_profile')
    @patch('services.innate_skills.find_tools_skill._fetch_manifest')
    def test_details_not_found(self, mock_manifest, mock_profile):
        """When tool doesn't exist, should return helpful error."""
        mock_profile.return_value = None
        mock_manifest.return_value = None

        result = handle_find_tools("topic", {"action": "details", "tool_name": "nonexistent"})
        assert 'not found' in result.lower()


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

        handle_find_tools("topic", {"action": "search", "query": "test", "limit": 50})

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

        result = handle_find_tools("topic", {"action": "search", "query": "anything"})
        assert 'No tools found' in result or 'No available' in result

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

        result = handle_find_tools("topic", {"action": "search", "query": "remember something"})
        assert 'innate skills' in result.lower() or 'No available' in result
