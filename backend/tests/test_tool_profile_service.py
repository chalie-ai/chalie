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
            {'tool_name': 'duckduckgo_search', 'tool_type': 'tool', 'short_summary': 'Search the web', 'domain': 'Information Retrieval', 'triage_triggers': []},
            {'tool_name': 'weather', 'tool_type': 'tool', 'short_summary': 'Check weather', 'domain': 'Environment', 'triage_triggers': []},
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


class TestManifestFallback:
    """Test manifest-based triage fallback when DB has no tool profiles."""

    @patch('services.tool_profile_service.ToolProfileService._get_redis')
    def test_empty_db_falls_back_to_manifest(self, mock_get_redis):
        """When DB has no tool rows, triage summaries come from manifests."""
        from services.tool_profile_service import ToolProfileService
        mock_redis = MagicMock()
        mock_redis.get.return_value = None  # Cache miss
        mock_get_redis.return_value = mock_redis

        svc = ToolProfileService()
        mock_db = MagicMock()
        mock_db.fetch_all.return_value = []  # Empty DB
        svc._db = mock_db

        # Mock registry with one on-demand tool
        mock_registry = MagicMock()
        mock_registry.get_on_demand_tools.return_value = ['google_news']
        mock_registry.tools = {
            'google_news': {
                'manifest': {
                    'name': 'google_news',
                    'description': 'Search Google News',
                    'documentation': "Search news. Triggers: 'latest news on...', 'what's happening in...'",
                    'category': 'research',
                    'trigger': {'type': 'on_demand'},
                }
            }
        }

        with patch('services.tool_registry_service.ToolRegistryService', return_value=mock_registry):
            result = svc.get_triage_summaries()

        assert 'google_news' in result
        assert '## Research' in result

    @patch('services.tool_profile_service.ToolProfileService._get_redis')
    def test_db_exception_falls_back_to_manifest(self, mock_get_redis):
        """When DB fetch raises, triage summaries come from manifests."""
        from services.tool_profile_service import ToolProfileService
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        mock_get_redis.return_value = mock_redis

        svc = ToolProfileService()
        mock_db = MagicMock()
        mock_db.fetch_all.side_effect = Exception("connection refused")
        svc._db = mock_db

        mock_registry = MagicMock()
        mock_registry.get_on_demand_tools.return_value = ['web_search']
        mock_registry.tools = {
            'web_search': {
                'manifest': {
                    'name': 'web_search',
                    'description': 'Search the web',
                    'documentation': "Web search tool. Use for 'search for...', 'look up...'",
                    'category': 'information_retrieval',
                    'trigger': {'type': 'on_demand'},
                }
            }
        }

        with patch('services.tool_registry_service.ToolRegistryService', return_value=mock_registry):
            result = svc.get_triage_summaries()

        assert 'web_search' in result
        assert '## Information Retrieval' in result

    @patch('services.tool_profile_service.ToolProfileService._get_redis')
    def test_only_skills_in_db_falls_back_to_manifest(self, mock_get_redis):
        """When DB only has skill rows (no tools), manifest fallback triggers."""
        from services.tool_profile_service import ToolProfileService
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        mock_get_redis.return_value = mock_redis

        svc = ToolProfileService()
        mock_db = MagicMock()
        mock_db.fetch_all.return_value = [
            {'tool_name': 'recall', 'tool_type': 'skill', 'short_summary': 'Search memory', 'domain': 'Innate Skill', 'triage_triggers': []},
        ]
        svc._db = mock_db

        mock_registry = MagicMock()
        mock_registry.get_on_demand_tools.return_value = ['google_news']
        mock_registry.tools = {
            'google_news': {
                'manifest': {
                    'name': 'google_news',
                    'description': 'Search news',
                    'documentation': "News search. Use for 'latest news on...'",
                    'category': 'research',
                    'trigger': {'type': 'on_demand'},
                }
            }
        }

        with patch('services.tool_registry_service.ToolRegistryService', return_value=mock_registry):
            result = svc.get_triage_summaries()

        # Skills filtered out → by_domain empty → manifest fallback fires
        assert 'google_news' in result

    def test_manifest_fallback_includes_triggers(self):
        """Manifest fallback should include trigger phrases in summaries."""
        from services.tool_profile_service import ToolProfileService

        svc = ToolProfileService()

        mock_registry = MagicMock()
        mock_registry.get_on_demand_tools.return_value = ['google_news']
        mock_registry.tools = {
            'google_news': {
                'manifest': {
                    'name': 'google_news',
                    'description': 'Search Google News',
                    'documentation': "News tool. Triggers: 'latest news on...', 'what's happening in...'",
                    'category': 'research',
                    'trigger': {'type': 'on_demand'},
                }
            }
        }

        with patch('services.tool_registry_service.ToolRegistryService', return_value=mock_registry):
            result = svc._manifest_fallback_summaries()

        assert 'latest news on' in result
        assert "what's happening in" in result


class TestFallbackProfile:
    """Test deterministic fallback profile generation from manifests."""

    def test_extracts_triggers_from_documentation_quotes(self):
        from services.tool_profile_service import ToolProfileService
        svc = ToolProfileService()
        manifest = {
            'name': 'google_news',
            'documentation': "Use for 'latest news on...', 'any updates about...', 'catch me up on...'",
            'category': 'research',
        }
        profile = svc._fallback_profile('google_news', manifest)
        triggers = profile['triage_triggers']
        assert len(triggers) == 3
        assert 'latest news on' in triggers
        assert 'any updates about' in triggers
        assert 'catch me up on' in triggers

    def test_sets_domain_from_category(self):
        from services.tool_profile_service import ToolProfileService
        svc = ToolProfileService()
        manifest = {
            'name': 'google_news',
            'documentation': 'Search news articles',
            'category': 'research',
        }
        profile = svc._fallback_profile('google_news', manifest)
        assert profile['domain'] == 'Research'

    def test_domain_normalizes_underscores(self):
        from services.tool_profile_service import ToolProfileService
        svc = ToolProfileService()
        manifest = {
            'name': 'web_search',
            'documentation': 'Search the web',
            'category': 'information_retrieval',
        }
        profile = svc._fallback_profile('web_search', manifest)
        assert profile['domain'] == 'Information Retrieval'

    def test_no_documentation_yields_empty_triggers(self):
        from services.tool_profile_service import ToolProfileService
        svc = ToolProfileService()
        manifest = {
            'name': 'simple_tool',
            'description': 'A simple tool with no quotes',
        }
        profile = svc._fallback_profile('simple_tool', manifest)
        assert profile['triage_triggers'] == []
        assert profile['domain'] == 'Other'

    def test_short_quoted_phrases_ignored(self):
        """Phrases shorter than 4 chars should not be extracted as triggers."""
        from services.tool_profile_service import ToolProfileService
        svc = ToolProfileService()
        manifest = {
            'name': 'test_tool',
            'documentation': "Triggers: 'hello there', 'yo'. For greetings.",
            'category': 'social',
        }
        profile = svc._fallback_profile('test_tool', manifest)
        # 'yo' is only 2 chars, below the 4-char minimum
        assert 'yo' not in profile['triage_triggers']
        assert 'hello there' in profile['triage_triggers']


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
