"""Extended tests for SemanticConsolidationService — pure string parsing only (no LLM/DB)."""

import json
import pytest
from unittest.mock import MagicMock, patch

from services.semantic_consolidation_service import SemanticConsolidationService


pytestmark = pytest.mark.unit


@pytest.fixture
def consolidation_service():
    """SemanticConsolidationService with mocked dependencies."""
    mock_llm = MagicMock()
    mock_storage = MagicMock()
    mock_config = MagicMock()
    mock_config.get_agent_config.return_value = {
        'similarity_threshold': 0.85,
    }
    with patch.object(SemanticConsolidationService, '_load_prompt_template', return_value='test prompt'):
        svc = SemanticConsolidationService(mock_llm, mock_storage, mock_config)
    return svc


# ── _extract_json_from_response ──────────────────────────────────────

class TestExtractJsonFromResponse:

    def test_extracts_from_json_fence(self, consolidation_service):
        response = 'Some preamble\n```json\n{"concepts": []}\n```\nAfterword'
        result = consolidation_service._extract_json_from_response(response)
        assert result == '{"concepts": []}'

    def test_extracts_from_generic_fence(self, consolidation_service):
        response = '```\n{"concepts": [], "relationships": []}\n```'
        result = consolidation_service._extract_json_from_response(response)
        assert result == '{"concepts": [], "relationships": []}'

    def test_returns_stripped_when_no_fence(self, consolidation_service):
        response = '  {"concepts": []}  '
        result = consolidation_service._extract_json_from_response(response)
        assert result == '{"concepts": []}'

    def test_takes_first_fence_when_multiple(self, consolidation_service):
        response = '```json\n{"first": true}\n```\n\n```json\n{"second": true}\n```'
        result = consolidation_service._extract_json_from_response(response)
        parsed = json.loads(result)
        assert parsed.get('first') is True


# ── _build_episode_content ───────────────────────────────────────────

class TestBuildEpisodeContent:

    def test_includes_gist(self, consolidation_service):
        episode = {'gist': 'User asked about Python async'}
        result = consolidation_service._build_episode_content(episode)
        assert 'Summary: User asked about Python async' in result

    def test_includes_intent_as_json(self, consolidation_service):
        episode = {'intent': {'type': 'question', 'direction': 'inbound'}}
        result = consolidation_service._build_episode_content(episode)
        assert 'Intent:' in result
        # The intent is JSON-serialized
        assert '"type"' in result
        assert '"question"' in result

    def test_skips_missing_fields(self, consolidation_service):
        episode = {'gist': 'Only gist present'}
        result = consolidation_service._build_episode_content(episode)
        assert 'Intent:' not in result
        assert 'Context:' not in result
        assert 'Action:' not in result
        assert 'Outcome:' not in result

    def test_double_newline_separator(self, consolidation_service):
        episode = {
            'gist': 'A summary',
            'action': 'searched web',
            'outcome': 'found results',
        }
        result = consolidation_service._build_episode_content(episode)
        # Parts joined by double newline
        assert '\n\n' in result
        parts = result.split('\n\n')
        assert len(parts) == 3
