"""Unit tests for ToolProfileService."""
import json
import pytest
from unittest.mock import MagicMock, patch, call

pytestmark = pytest.mark.unit


def _make_manifest(name="test_tool", has_documentation=True):
    m = {
        "name": name,
        "description": f"A {name} tool for testing",
        "trigger": {"type": "on_demand"},
        "parameters": {"query": {"type": "string"}},
        "returns": {"result": {"type": "string"}},
        "examples": [
            {"params": {"query": "test"}, "description": "Run a test query"}
        ],
    }
    if has_documentation:
        m["documentation"] = f"The {name} tool searches for information. Use it when the user asks to search, find, or look up information online. It returns titles, URLs, and snippets."
    return m


class TestManifestHash:
    def test_hash_is_deterministic(self):
        from services.tool_profile_service import _compute_manifest_hash
        m = _make_manifest()
        assert _compute_manifest_hash(m) == _compute_manifest_hash(m)

    def test_hash_changes_with_content(self):
        from services.tool_profile_service import _compute_manifest_hash
        m1 = _make_manifest("tool_a")
        m2 = _make_manifest("tool_b")
        assert _compute_manifest_hash(m1) != _compute_manifest_hash(m2)

    def test_hash_is_string(self):
        from services.tool_profile_service import _compute_manifest_hash
        h = _compute_manifest_hash(_make_manifest())
        assert isinstance(h, str)
        assert len(h) == 32  # MD5 hex


class TestCheckStaleness:
    def test_no_profile_returns_stale(self):
        from services.tool_profile_service import ToolProfileService
        svc = ToolProfileService()
        mock_db = MagicMock()
        mock_db.fetch_all.return_value = []  # No rows
        svc._db = mock_db
        assert svc.check_staleness("unknown_tool") is True

    def test_matching_hash_returns_not_stale(self):
        from services.tool_profile_service import ToolProfileService, _compute_manifest_hash
        svc = ToolProfileService()
        manifest = _make_manifest("test_tool")
        current_hash = _compute_manifest_hash(manifest)
        mock_db = MagicMock()
        mock_db.fetch_all.return_value = [{'manifest_hash': current_hash}]
        svc._db = mock_db
        assert svc.check_staleness("test_tool", current_hash) is False

    def test_changed_hash_returns_stale(self):
        from services.tool_profile_service import ToolProfileService
        svc = ToolProfileService()
        mock_db = MagicMock()
        mock_db.fetch_all.return_value = [{'manifest_hash': 'old_hash'}]
        svc._db = mock_db
        assert svc.check_staleness("test_tool", "new_hash") is True

    def test_db_error_returns_stale(self):
        from services.tool_profile_service import ToolProfileService
        svc = ToolProfileService()
        mock_db = MagicMock()
        mock_db.fetch_all.side_effect = Exception("DB error")
        svc._db = mock_db
        assert svc.check_staleness("test_tool") is True


class TestGetFullProfile:
    def test_returns_none_for_missing_tool(self):
        from services.tool_profile_service import ToolProfileService
        svc = ToolProfileService()
        mock_db = MagicMock()
        mock_db.fetch_all.return_value = []
        svc._db = mock_db
        assert svc.get_full_profile("nonexistent") is None

    def test_returns_dict_for_existing_tool(self):
        from services.tool_profile_service import ToolProfileService
        svc = ToolProfileService()
        mock_db = MagicMock()
        mock_db.fetch_all.return_value = [{
            'tool_name': 'test_tool',
            'short_summary': 'A test tool',
            'full_profile': 'This is the full profile',
            'usage_scenarios': '["scenario1", "scenario2"]',
            'anti_scenarios': '[]',
            'complementary_skills': '["recall"]',
            'enrichment_episode_ids': '[]',
            'enrichment_count': 0,
        }]
        svc._db = mock_db
        profile = svc.get_full_profile("test_tool")
        assert profile is not None
        assert profile['tool_name'] == 'test_tool'
        assert isinstance(profile['usage_scenarios'], list)
        assert 'scenario1' in profile['usage_scenarios']


class TestGetTriageSummaries:
    @patch('services.tool_profile_service.ToolProfileService._get_redis')
    def test_returns_cached_value(self, mock_get_redis):
        from services.tool_profile_service import ToolProfileService
        mock_redis = MagicMock()
        mock_redis.get.return_value = "## Cached Summaries\n- tool: does stuff"
        mock_get_redis.return_value = mock_redis

        svc = ToolProfileService()
        result = svc.get_triage_summaries()
        assert result == "## Cached Summaries\n- tool: does stuff"

    @patch('services.tool_profile_service.ToolProfileService._get_redis')
    def test_builds_from_db_when_cache_miss(self, mock_get_redis):
        from services.tool_profile_service import ToolProfileService
        mock_redis = MagicMock()
        mock_redis.get.return_value = None  # Cache miss
        mock_get_redis.return_value = mock_redis

        svc = ToolProfileService()
        mock_db = MagicMock()
        mock_db.fetch_all.return_value = [
            {'tool_name': 'duckduckgo_search', 'tool_type': 'tool', 'short_summary': 'Search the web'},
            {'tool_name': 'weather', 'tool_type': 'tool', 'short_summary': 'Check weather'},
        ]
        svc._db = mock_db

        result = svc.get_triage_summaries()
        assert 'duckduckgo_search' in result
        assert 'weather' in result
        assert '## Information Retrieval' in result or '## Environment' in result

    @patch('services.tool_profile_service.ToolProfileService._get_redis')
    def test_skills_not_in_triage_summaries(self, mock_get_redis):
        """Skills should not appear in triage summaries — they're always available."""
        from services.tool_profile_service import ToolProfileService
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        mock_get_redis.return_value = mock_redis

        svc = ToolProfileService()
        mock_db = MagicMock()
        mock_db.fetch_all.return_value = [
            {'tool_name': 'recall', 'tool_type': 'skill', 'short_summary': 'Search memory'},
            {'tool_name': 'duckduckgo_search', 'tool_type': 'tool', 'short_summary': 'Search web'},
        ]
        svc._db = mock_db
        result = svc.get_triage_summaries()
        assert 'recall' not in result  # Skills excluded from triage prompt


class TestScenarioCap:
    """Test that scenario count is capped at MAX_SCENARIOS."""

    def test_scenarios_capped_at_50_in_build_profile(self):
        from services.tool_profile_service import ToolProfileService, MAX_SCENARIOS
        svc = ToolProfileService()

        # Mock everything needed
        mock_db = MagicMock()
        mock_db.fetch_all.side_effect = [
            [],   # check_staleness → no rows (stale)
            [],   # _get_related_episodes → no episodes
        ]
        svc._db = mock_db

        # LLM returns 60 scenarios — should be capped to 50
        with patch.object(svc, '_get_llm') as mock_get_llm, \
             patch.object(svc, '_get_embedding_service') as mock_get_emb, \
             patch.object(svc, '_get_related_episodes', return_value="No episodes"), \
             patch.object(svc, 'check_staleness', return_value=True):

            mock_llm = MagicMock()
            import json
            mock_llm.send_message.return_value = MagicMock(text=json.dumps({
                'short_summary': 'Test tool',
                'full_profile': 'A test tool profile',
                'usage_scenarios': [f"scenario {i}" for i in range(60)],
                'anti_scenarios': [],
                'complementary_skills': [],
            }))
            mock_get_llm.return_value = mock_llm

            mock_emb = MagicMock()
            mock_emb.generate_embedding.return_value = [0.1] * 768
            mock_get_emb.return_value = mock_emb

            svc.build_profile("test_tool", _make_manifest())

            # Check what was inserted
            call_args = mock_db.execute.call_args
            if call_args:
                args = call_args[0]
                # Find the usage_scenarios argument
                for arg in args:
                    if isinstance(arg, str) and arg.startswith('[') and 'scenario' in arg:
                        scenarios = json.loads(arg)
                        assert len(scenarios) <= MAX_SCENARIOS
                        break
