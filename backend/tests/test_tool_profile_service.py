"""Unit tests for ToolProfileService."""
import json
import pytest
from unittest.mock import MagicMock, patch, call

from services.tool_profile_service import (
    ToolProfileService,
    _compute_manifest_hash,
    MAX_SCENARIOS,
    TRIAGE_SUMMARIES_CACHE_KEY,
)
from services.database_service import get_shared_db_service
from services.memory_store import MemoryStore

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


def _seed_profile(db, tool_name="test_tool", **overrides):
    """Insert a tool_capability_profiles row for tests that need one."""
    defaults = {
        'tool_type': 'tool',
        'short_summary': f'A {tool_name} tool',
        'full_profile': f'This is the full profile for {tool_name}',
        'usage_scenarios': '["scenario1", "scenario2"]',
        'anti_scenarios': '[]',
        'complementary_skills': '["recall"]',
        'manifest_hash': 'test_hash',
        'domain': 'Other',
        'triage_triggers': '[]',
        'effort': 'moderate',
        'descriptor': tool_name,
        'enrichment_episode_ids': '[]',
        'enrichment_count': 0,
    }
    defaults.update(overrides)
    db.execute("""
        INSERT OR REPLACE INTO tool_capability_profiles
            (tool_name, tool_type, short_summary, full_profile, usage_scenarios,
             anti_scenarios, complementary_skills, manifest_hash, domain,
             triage_triggers, effort, descriptor, enrichment_episode_ids,
             enrichment_count, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
    """, (
        tool_name,
        defaults['tool_type'],
        defaults['short_summary'],
        defaults['full_profile'],
        defaults['usage_scenarios'],
        defaults['anti_scenarios'],
        defaults['complementary_skills'],
        defaults['manifest_hash'],
        defaults['domain'],
        defaults['triage_triggers'],
        defaults['effort'],
        defaults['descriptor'],
        defaults['enrichment_episode_ids'],
        defaults['enrichment_count'],
    ))
    db.commit()


class TestManifestHash:
    def test_hash_is_deterministic(self):
        m = _make_manifest()
        assert _compute_manifest_hash(m) == _compute_manifest_hash(m)

    def test_hash_changes_with_content(self):
        m1 = _make_manifest("tool_a")
        m2 = _make_manifest("tool_b")
        assert _compute_manifest_hash(m1) != _compute_manifest_hash(m2)

    def test_hash_is_string(self):
        h = _compute_manifest_hash(_make_manifest())
        assert isinstance(h, str)
        assert len(h) == 32  # MD5 hex


class TestCheckStaleness:
    def test_no_profile_returns_stale(self, db):
        svc = ToolProfileService(get_shared_db_service())
        assert svc.check_staleness("unknown_tool") is True

    def test_matching_hash_returns_not_stale(self, db):
        manifest = _make_manifest("test_tool")
        current_hash = _compute_manifest_hash(manifest)
        _seed_profile(db, "test_tool", manifest_hash=current_hash)

        svc = ToolProfileService(get_shared_db_service())
        assert svc.check_staleness("test_tool", current_hash) is False

    def test_changed_hash_returns_stale(self, db):
        _seed_profile(db, "test_tool", manifest_hash='old_hash')

        svc = ToolProfileService(get_shared_db_service())
        assert svc.check_staleness("test_tool", "new_hash") is True

    def test_db_error_returns_stale(self, db):
        svc = ToolProfileService(get_shared_db_service())
        with patch.object(svc, '_get_db') as patched_get_db:
            broken_db = MagicMock()
            broken_db.fetch_all.side_effect = Exception("DB error")
            patched_get_db.return_value = broken_db
            assert svc.check_staleness("test_tool") is True


class TestGetFullProfile:
    def test_returns_none_for_missing_tool(self, db):
        svc = ToolProfileService(get_shared_db_service())
        assert svc.get_full_profile("nonexistent") is None

    def test_returns_dict_for_existing_tool(self, db):
        _seed_profile(db, "test_tool",
                      short_summary='A test tool',
                      full_profile='This is the full profile',
                      usage_scenarios='["scenario1", "scenario2"]',
                      anti_scenarios='[]',
                      complementary_skills='["recall"]',
                      enrichment_episode_ids='[]',
                      enrichment_count=0)

        svc = ToolProfileService(get_shared_db_service())
        profile = svc.get_full_profile("test_tool")
        assert profile is not None
        assert profile['tool_name'] == 'test_tool'
        assert isinstance(profile['usage_scenarios'], list)
        assert 'scenario1' in profile['usage_scenarios']


class TestGetTriageSummaries:
    @patch('services.tool_profile_service.ToolProfileService._get_store')
    def test_returns_cached_value(self, mock_get_store, db):
        store = MemoryStore()
        store.set(TRIAGE_SUMMARIES_CACHE_KEY, "## Cached Summaries\n- tool: does stuff")
        mock_get_store.return_value = store

        svc = ToolProfileService(get_shared_db_service())
        result = svc.get_triage_summaries()
        assert result == "## Cached Summaries\n- tool: does stuff"

    @patch('services.tool_profile_service.ToolProfileService._get_store')
    def test_builds_from_db_when_cache_miss(self, mock_get_store, db):
        store = MemoryStore()  # empty store → cache miss
        mock_get_store.return_value = store

        _seed_profile(db, 'duckduckgo_search',
                      tool_type='tool',
                      short_summary='Search the web',
                      domain='Information Retrieval',
                      triage_triggers='[]',
                      effort='moderate')
        _seed_profile(db, 'weather',
                      tool_type='tool',
                      short_summary='Check weather',
                      domain='Environment',
                      triage_triggers='[]',
                      effort='moderate')

        svc = ToolProfileService(get_shared_db_service())
        result = svc.get_triage_summaries()
        assert 'duckduckgo_search' in result
        assert 'weather' in result
        assert '## Information Retrieval' in result or '## Environment' in result

    @patch('services.tool_profile_service.ToolProfileService._get_store')
    def test_skills_not_in_triage_summaries(self, mock_get_store, db):
        """Skills should not appear in triage summaries -- they're always available."""
        store = MemoryStore()  # empty store → cache miss
        mock_get_store.return_value = store

        _seed_profile(db, 'recall', tool_type='skill',
                      short_summary='Search memory', domain='Innate Skill')
        _seed_profile(db, 'duckduckgo_search', tool_type='tool',
                      short_summary='Search web', domain='Information Retrieval')

        svc = ToolProfileService(get_shared_db_service())
        result = svc.get_triage_summaries()
        assert 'recall' not in result  # Skills excluded from triage prompt


class TestManifestFallback:
    """Test manifest-based triage fallback when DB has no tool profiles."""

    @patch('services.tool_profile_service.ToolProfileService._get_store')
    def test_empty_db_falls_back_to_manifest(self, mock_get_store, db):
        """When DB has no tool rows, triage summaries come from manifests."""
        store = MemoryStore()  # empty store → cache miss
        mock_get_store.return_value = store

        svc = ToolProfileService(get_shared_db_service())

        # Mock registry with one on-demand tool
        mock_registry = MagicMock()
        mock_registry.get_on_demand_tools.return_value = ['news_tool']
        mock_registry.tools = {
            'news_tool': {
                'manifest': {
                    'name': 'news_tool',
                    'description': 'Search news',
                    'documentation': "Search news. Triggers: 'latest news on...', 'what's happening in...'",
                    'category': 'research',
                    'trigger': {'type': 'on_demand'},
                }
            }
        }

        with patch('services.tool_registry_service.ToolRegistryService', return_value=mock_registry):
            result = svc.get_triage_summaries()

        assert 'news_tool' in result
        assert '## Research' in result

    @patch('services.tool_profile_service.ToolProfileService._get_store')
    def test_db_exception_falls_back_to_manifest(self, mock_get_store, db):
        """When DB fetch raises, triage summaries come from manifests."""
        store = MemoryStore()  # empty store → cache miss
        mock_get_store.return_value = store

        svc = ToolProfileService(get_shared_db_service())

        # Patch _get_db to return a broken service that raises on fetch_all
        broken_db = MagicMock()
        broken_db.fetch_all.side_effect = Exception("connection refused")

        mock_registry = MagicMock()
        mock_registry.get_on_demand_tools.return_value = ['weather']
        mock_registry.tools = {
            'weather': {
                'manifest': {
                    'name': 'weather',
                    'description': 'Search the web',
                    'documentation': "Web search tool. Use for 'search for...', 'look up...'",
                    'category': 'information_retrieval',
                    'trigger': {'type': 'on_demand'},
                }
            }
        }

        with patch.object(svc, '_get_db', return_value=broken_db), \
             patch('services.tool_registry_service.ToolRegistryService', return_value=mock_registry):
            result = svc.get_triage_summaries()

        assert 'weather' in result
        assert '## Information Retrieval' in result

    @patch('services.tool_profile_service.ToolProfileService._get_store')
    def test_only_skills_in_db_falls_back_to_manifest(self, mock_get_store, db):
        """When DB only has skill rows (no tools), manifest fallback triggers."""
        store = MemoryStore()  # empty store → cache miss
        mock_get_store.return_value = store

        _seed_profile(db, 'recall', tool_type='skill',
                      short_summary='Search memory', domain='Innate Skill')

        svc = ToolProfileService(get_shared_db_service())

        mock_registry = MagicMock()
        mock_registry.get_on_demand_tools.return_value = ['news_tool']
        mock_registry.tools = {
            'news_tool': {
                'manifest': {
                    'name': 'news_tool',
                    'description': 'Search news',
                    'documentation': "News search. Use for 'latest news on...'",
                    'category': 'research',
                    'trigger': {'type': 'on_demand'},
                }
            }
        }

        with patch('services.tool_registry_service.ToolRegistryService', return_value=mock_registry):
            result = svc.get_triage_summaries()

        # Skills filtered out -> by_domain empty -> manifest fallback fires
        assert 'news_tool' in result

    def test_manifest_fallback_includes_triggers(self, db):
        """Manifest fallback should include trigger phrases in summaries."""
        svc = ToolProfileService(get_shared_db_service())

        mock_registry = MagicMock()
        mock_registry.get_on_demand_tools.return_value = ['news_tool']
        mock_registry.tools = {
            'news_tool': {
                'manifest': {
                    'name': 'news_tool',
                    'description': 'Search news',
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

    def test_extracts_triggers_from_documentation_quotes(self, db):
        svc = ToolProfileService(get_shared_db_service())
        manifest = {
            'name': 'news_tool',
            'documentation': "Use for 'latest news on...', 'any updates about...', 'catch me up on...'",
            'category': 'research',
        }
        profile = svc._fallback_profile('news_tool', manifest)
        triggers = profile['triage_triggers']
        assert len(triggers) == 3
        assert 'latest news on' in triggers
        assert 'any updates about' in triggers
        assert 'catch me up on' in triggers

    def test_sets_domain_from_category(self, db):
        svc = ToolProfileService(get_shared_db_service())
        manifest = {
            'name': 'news_tool',
            'documentation': 'Search news articles',
            'category': 'research',
        }
        profile = svc._fallback_profile('news_tool', manifest)
        assert profile['domain'] == 'Research'

    def test_domain_normalizes_underscores(self, db):
        svc = ToolProfileService(get_shared_db_service())
        manifest = {
            'name': 'weather',
            'documentation': 'Search the web',
            'category': 'information_retrieval',
        }
        profile = svc._fallback_profile('weather', manifest)
        assert profile['domain'] == 'Information Retrieval'

    def test_no_documentation_yields_empty_triggers(self, db):
        svc = ToolProfileService(get_shared_db_service())
        manifest = {
            'name': 'simple_tool',
            'description': 'A simple tool with no quotes',
        }
        profile = svc._fallback_profile('simple_tool', manifest)
        assert profile['triage_triggers'] == []
        assert profile['domain'] == 'Other'

    def test_short_quoted_phrases_ignored(self, db):
        """Phrases shorter than 4 chars should not be extracted as triggers."""
        svc = ToolProfileService(get_shared_db_service())
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

    def test_scenarios_capped_at_50_in_build_profile(self, db):
        svc = ToolProfileService(get_shared_db_service())

        # LLM returns 60 scenarios -- should be capped to 50
        with patch.object(svc, '_get_llm') as mock_get_llm, \
             patch.object(svc, '_get_embedding_service') as mock_get_emb, \
             patch.object(svc, '_get_related_episodes', return_value="No episodes"), \
             patch.object(svc, 'check_staleness', return_value=True), \
             patch.object(svc, '_invalidate_cache'):

            mock_llm = MagicMock()
            mock_llm.send_message.return_value = MagicMock(text=json.dumps({
                'short_summary': 'Test tool',
                'full_profile': 'A test tool profile',
                'usage_scenarios': [f"scenario {i}" for i in range(60)],
                'anti_scenarios': [],
                'complementary_skills': [],
            }))
            mock_get_llm.return_value = mock_llm

            mock_emb = MagicMock()
            mock_emb.generate_embedding.return_value = [0.1] * 256
            mock_get_emb.return_value = mock_emb

            svc.build_profile("test_tool", _make_manifest())

            # Verify the stored scenarios are capped
            row = db.execute(
                "SELECT usage_scenarios FROM tool_capability_profiles WHERE tool_name = ?",
                ("test_tool",)
            ).fetchone()
            assert row is not None
            scenarios = json.loads(row['usage_scenarios'])
            assert len(scenarios) <= MAX_SCENARIOS
