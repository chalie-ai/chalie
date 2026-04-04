"""Unit tests for ToolProfileService."""
import json
import pytest
from unittest.mock import MagicMock, patch

from services.tool_profile_service import (
    ToolProfileService,
    _compute_manifest_hash,
    MAX_SCENARIOS,
    TRIAGE_SUMMARIES_CACHE_KEY,
)
from services.database_service import get_shared_db_service
from services.memory_store import MemoryStore
from services.tool_library_service import BUILTIN_TOOL_PROFILES, TOOL_METADATA

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


# -- Helpers for seed tests --------------------------------------------------

def _make_embedding(dims=256):
    return [0.1] * dims


def _patch_embedding(mock_get_emb, dims=256):
    mock_emb = MagicMock()
    mock_emb.generate_embedding.return_value = _make_embedding(dims)
    mock_get_emb.return_value = mock_emb
    return mock_emb


# ---------------------------------------------------------------------------


class TestSeedBuiltinProfiles:
    """Tests for ToolProfileService.seed_builtin_profiles()."""

    def test_seeds_all_builtin_tools(self, db):
        """Every tool in BUILTIN_TOOL_PROFILES gets a row in tool_capability_profiles."""
        svc = ToolProfileService(get_shared_db_service())

        with patch.object(svc, '_get_embedding_service') as mock_get_emb, \
             patch.object(svc, '_invalidate_cache'):
            _patch_embedding(mock_get_emb)
            svc.seed_builtin_profiles()

        for tool_name, profile in BUILTIN_TOOL_PROFILES.items():
            row = db.execute(
                "SELECT short_summary, full_profile FROM tool_capability_profiles WHERE tool_name = ?",
                (tool_name,)
            ).fetchone()
            assert row is not None, f"{tool_name} was not seeded"
            assert row['short_summary'] == profile['short_summary'][:200]
            assert row['full_profile'] == profile['full_profile']

    def test_seed_uses_manifest_hash(self, db):
        """The stored manifest_hash matches _compute_manifest_hash(TOOL_METADATA[name])."""
        svc = ToolProfileService(get_shared_db_service())

        with patch.object(svc, '_get_embedding_service') as mock_get_emb, \
             patch.object(svc, '_invalidate_cache'):
            _patch_embedding(mock_get_emb)
            svc.seed_builtin_profiles()

        for tool_name in BUILTIN_TOOL_PROFILES:
            manifest = TOOL_METADATA.get(tool_name)
            if manifest is None:
                continue
            expected_hash = _compute_manifest_hash(manifest)
            row = db.execute(
                "SELECT manifest_hash FROM tool_capability_profiles WHERE tool_name = ?",
                (tool_name,)
            ).fetchone()
            assert row is not None
            assert row['manifest_hash'] == expected_hash, (
                f"{tool_name}: stored hash {row['manifest_hash']!r} != expected {expected_hash!r}"
            )

    def test_seed_skips_when_hash_matches(self, db):
        """Calling seed_builtin_profiles() twice does not re-embed on the second call."""
        svc = ToolProfileService(get_shared_db_service())

        with patch.object(svc, '_get_embedding_service') as mock_get_emb, \
             patch.object(svc, '_invalidate_cache'):
            mock_emb = _patch_embedding(mock_get_emb)
            svc.seed_builtin_profiles()
            first_call_count = mock_emb.generate_embedding.call_count

        # Second seed call — all hashes now match, so no embeddings should be generated
        with patch.object(svc, '_get_embedding_service') as mock_get_emb2, \
             patch.object(svc, '_invalidate_cache'):
            mock_emb2 = _patch_embedding(mock_get_emb2)
            svc.seed_builtin_profiles()
            second_call_count = mock_emb2.generate_embedding.call_count

        assert first_call_count > 0, "Expected embeddings to be generated on first seed"
        assert second_call_count == 0, "No embeddings should be generated when hash matches"

    def test_seed_creates_vec_embedding(self, db):
        """tool_capability_profiles_vec has an entry for each seeded tool."""
        svc = ToolProfileService(get_shared_db_service())

        with patch.object(svc, '_get_embedding_service') as mock_get_emb, \
             patch.object(svc, '_invalidate_cache'):
            _patch_embedding(mock_get_emb)
            svc.seed_builtin_profiles()

        for tool_name in BUILTIN_TOOL_PROFILES:
            row = db.execute(
                """
                SELECT v.rowid FROM tool_capability_profiles_vec v
                JOIN tool_capability_profiles tcp ON tcp.rowid = v.rowid
                WHERE tcp.tool_name = ?
                """,
                (tool_name,)
            ).fetchone()
            assert row is not None, f"{tool_name} has no vec entry"

    def test_seed_populates_rebuild_guard_fields(self, db):
        """Seeded rows pass _profile_needs_rebuild — usage_scenarios, triage_triggers, descriptor non-empty."""
        svc = ToolProfileService(get_shared_db_service())

        with patch.object(svc, '_get_embedding_service') as mock_get_emb, \
             patch.object(svc, '_invalidate_cache'):
            _patch_embedding(mock_get_emb)
            svc.seed_builtin_profiles()

        for tool_name in BUILTIN_TOOL_PROFILES:
            profile = svc.get_full_profile(tool_name)
            assert profile is not None
            # _profile_needs_rebuild checks these four fields
            assert profile.get('usage_scenarios'), f"{tool_name}: usage_scenarios empty"
            assert profile.get('triage_triggers'), f"{tool_name}: triage_triggers empty"
            assert profile.get('descriptor'), f"{tool_name}: descriptor empty"
            assert not svc._profile_needs_rebuild(profile), (
                f"{tool_name}: _profile_needs_rebuild returned True after seeding"
            )

    def test_seed_handles_embedding_failure(self, db):
        """If embedding generation raises, the profile row is still upserted (no vec entry)."""
        svc = ToolProfileService(get_shared_db_service())

        with patch.object(svc, '_get_embedding_service') as mock_get_emb, \
             patch.object(svc, '_invalidate_cache'):
            mock_emb = MagicMock()
            mock_emb.generate_embedding.side_effect = RuntimeError("model not loaded")
            mock_get_emb.return_value = mock_emb
            svc.seed_builtin_profiles()

        # Profile rows must exist despite embedding failure
        for tool_name in BUILTIN_TOOL_PROFILES:
            row = db.execute(
                "SELECT tool_name FROM tool_capability_profiles WHERE tool_name = ?",
                (tool_name,)
            ).fetchone()
            assert row is not None, f"{tool_name}: profile row missing after embedding failure"

        # No vec entries should exist when embedding failed
        for tool_name in BUILTIN_TOOL_PROFILES:
            vec_row = db.execute(
                """
                SELECT v.rowid FROM tool_capability_profiles_vec v
                JOIN tool_capability_profiles tcp ON tcp.rowid = v.rowid
                WHERE tcp.tool_name = ?
                """,
                (tool_name,)
            ).fetchone()
            assert vec_row is None, f"{tool_name}: unexpected vec entry when embedding failed"

    def test_bootstrap_skips_seeded_tools(self, db):
        """After seed_builtin_profiles(), bootstrap_all() does not call build_profile for seeded tools."""
        svc = ToolProfileService(get_shared_db_service())

        # First seed so hashes are current
        with patch.object(svc, '_get_embedding_service') as mock_get_emb, \
             patch.object(svc, '_invalidate_cache'):
            _patch_embedding(mock_get_emb)
            svc.seed_builtin_profiles()

        # Now run bootstrap_all; build_profile should NOT be called for any builtin tool
        with patch.object(svc, 'build_profile') as mock_build, \
             patch.object(svc, '_get_embedding_service') as mock_get_emb2, \
             patch.object(svc, '_invalidate_cache'), \
             patch('services.tool_registry_service.ToolRegistryService') as mock_reg_cls:
            _patch_embedding(mock_get_emb2)
            # Registry has no external tools — only builtins were seeded
            mock_registry = MagicMock()
            mock_registry.tools = {}
            mock_reg_cls.return_value = mock_registry

            svc.bootstrap_all()

        called_tools = [call.args[0] for call in mock_build.call_args_list]
        for builtin_name in BUILTIN_TOOL_PROFILES:
            assert builtin_name not in called_tools, (
                f"bootstrap_all() called build_profile for already-seeded tool: {builtin_name}"
            )


class TestTruncateKeywords:
    """Tests for _truncate_keywords() helper."""

    def test_within_limit_returns_unchanged(self):
        """Keywords shorter than the limit pass through unmodified."""
        from services.tool_profile_service import _truncate_keywords
        kw = "search,web,lookup,find"
        assert _truncate_keywords(kw, max_len=256) == kw

    def test_at_exactly_limit_returns_unchanged(self):
        """Keywords exactly at the limit should pass through unmodified."""
        from services.tool_profile_service import _truncate_keywords
        kw = "a" * 256
        assert _truncate_keywords(kw, max_len=256) == kw

    def test_over_limit_truncates_at_last_complete_keyword(self):
        """Over-limit string should be cut at the last comma before max_len."""
        from services.tool_profile_service import _truncate_keywords
        # 'weather,forecast' is 16 chars; 'temperature' would push it over 20
        kw = "weather,forecast,temperature"
        result = _truncate_keywords(kw, max_len=20)
        # 'weather,forecast' = 16 chars — fits; adding ',temperature' would be 28 -> truncated
        assert result == "weather,forecast"
        assert len(result) <= 20
        assert ',' not in result or not result.endswith(',')

    def test_empty_string_returns_empty(self):
        """Empty keywords string returns empty string."""
        from services.tool_profile_service import _truncate_keywords
        assert _truncate_keywords("", max_len=256) == ""

    def test_single_keyword_over_limit_returns_empty(self):
        """A single keyword longer than max_len cannot be truncated cleanly, returns empty."""
        from services.tool_profile_service import _truncate_keywords
        long_kw = "a" * 300
        result = _truncate_keywords(long_kw, max_len=256)
        # Can't fit even the first keyword — result must be empty
        assert result == ""

    def test_no_partial_keywords_in_result(self):
        """Result must never end mid-keyword — only complete comma-delimited terms."""
        from services.tool_profile_service import _truncate_keywords
        kw = "alpha,bravo,charlie,delta,echo"
        result = _truncate_keywords(kw, max_len=15)
        # Each term in result must be a full keyword from the original list
        if result:
            for term in result.split(','):
                assert term in kw.split(','), f"Partial keyword found: {term!r}"

    def test_multiple_keywords_all_fit(self):
        """When all keywords fit, the full string is returned unchanged."""
        from services.tool_profile_service import _truncate_keywords
        kw = "a,b,c,d,e"
        assert _truncate_keywords(kw, max_len=50) == kw

    def test_default_max_len_is_256(self):
        """Default max_len must be 256 (matches DB column constraint)."""
        from services.tool_profile_service import _truncate_keywords
        # Just over 256 chars worth of keywords
        parts = ["keyword" + str(i) for i in range(40)]  # ~320 chars when joined
        kw = ",".join(parts)
        result = _truncate_keywords(kw)
        assert len(result) <= 256


class TestProfileNeedsRebuild:
    """Tests for ToolProfileService._profile_needs_rebuild()."""

    def _full_profile(self, **overrides):
        """Return a minimal valid profile dict that passes all rebuild checks."""
        base = {
            'tool_type': 'tool',
            'domain': 'Research',
            'triage_triggers': ['some trigger'],
            'usage_scenarios': ['some scenario'],
            'descriptor': 'my_tool',
            'keywords': 'search,web',
        }
        base.update(overrides)
        return base

    def test_returns_false_when_all_fields_present(self, db):
        svc = ToolProfileService(get_shared_db_service())
        profile = self._full_profile()
        assert svc._profile_needs_rebuild(profile) is False

    def test_returns_true_when_keywords_empty_string(self, db):
        """Empty keywords string triggers rebuild."""
        svc = ToolProfileService(get_shared_db_service())
        profile = self._full_profile(keywords='')
        assert svc._profile_needs_rebuild(profile) is True

    def test_returns_true_when_keywords_none(self, db):
        """NULL keywords (missing key) triggers rebuild."""
        svc = ToolProfileService(get_shared_db_service())
        profile = self._full_profile(keywords=None)
        assert svc._profile_needs_rebuild(profile) is True

    def test_returns_true_when_keywords_key_absent(self, db):
        """Profile dict with no 'keywords' key at all triggers rebuild."""
        svc = ToolProfileService(get_shared_db_service())
        profile = self._full_profile()
        del profile['keywords']
        assert svc._profile_needs_rebuild(profile) is True

    def test_returns_true_when_descriptor_missing(self, db):
        svc = ToolProfileService(get_shared_db_service())
        profile = self._full_profile(descriptor='')
        assert svc._profile_needs_rebuild(profile) is True

    def test_returns_true_when_usage_scenarios_empty(self, db):
        svc = ToolProfileService(get_shared_db_service())
        profile = self._full_profile(usage_scenarios=[])
        assert svc._profile_needs_rebuild(profile) is True

    def test_returns_true_when_triage_triggers_empty(self, db):
        svc = ToolProfileService(get_shared_db_service())
        profile = self._full_profile(triage_triggers=[])
        assert svc._profile_needs_rebuild(profile) is True


class TestSeedBuiltinProfilesKeywords:
    """Tests for keywords handling in seed_builtin_profiles()."""

    def test_seed_writes_keywords_to_db(self, db):
        """seed_builtin_profiles() must write keywords column for each seeded tool."""
        svc = ToolProfileService(get_shared_db_service())

        with patch.object(svc, '_get_embedding_service') as mock_get_emb, \
             patch.object(svc, '_invalidate_cache'):
            _patch_embedding(mock_get_emb)
            svc.seed_builtin_profiles()

        for tool_name, profile in BUILTIN_TOOL_PROFILES.items():
            if not profile.get('keywords'):
                continue
            row = db.execute(
                "SELECT keywords FROM tool_capability_profiles WHERE tool_name = ?",
                (tool_name,)
            ).fetchone()
            assert row is not None, f"{tool_name} not seeded"
            assert row['keywords'], f"{tool_name}: keywords column is empty after seeding"

    def test_seed_reseeds_when_existing_profile_has_empty_keywords(self, db):
        """If a seeded tool row has empty keywords, bootstrap_all triggers a rebuild."""
        svc = ToolProfileService(get_shared_db_service())

        # Pick a tool known to have keywords in BUILTIN_TOOL_PROFILES
        tool_name = 'weather'
        assert BUILTIN_TOOL_PROFILES[tool_name].get('keywords'), \
            "Test assumption: 'weather' must have keywords defined"

        # Seed normally first
        with patch.object(svc, '_get_embedding_service') as mock_get_emb, \
             patch.object(svc, '_invalidate_cache'):
            _patch_embedding(mock_get_emb)
            svc.seed_builtin_profiles()

        # Wipe the keywords column to simulate a pre-migration row
        db.execute(
            "UPDATE tool_capability_profiles SET keywords = '' WHERE tool_name = ?",
            (tool_name,)
        )
        db.commit()

        # _profile_needs_rebuild should now return True for this tool
        profile = svc.get_full_profile(tool_name)
        assert svc._profile_needs_rebuild(profile) is True, \
            "Profile with empty keywords should need rebuild"

    def test_seed_keywords_are_truncated_to_256(self, db):
        """Keywords written to DB must not exceed 256 characters."""
        svc = ToolProfileService(get_shared_db_service())

        with patch.object(svc, '_get_embedding_service') as mock_get_emb, \
             patch.object(svc, '_invalidate_cache'):
            _patch_embedding(mock_get_emb)
            svc.seed_builtin_profiles()

        for tool_name in BUILTIN_TOOL_PROFILES:
            row = db.execute(
                "SELECT keywords FROM tool_capability_profiles WHERE tool_name = ?",
                (tool_name,)
            ).fetchone()
            if row and row['keywords']:
                assert len(row['keywords']) <= 256, \
                    f"{tool_name}: keywords exceeds 256 chars ({len(row['keywords'])})"


class TestEmbeddingTextFormula:
    """Verify embedding text uses short_summary + keywords, not full_profile."""

    def test_build_profile_embeds_summary_plus_keywords(self, db):
        """build_profile() must embed 'short_summary keywords', not full_profile."""
        svc = ToolProfileService(get_shared_db_service())

        profile_data = {
            'short_summary': 'Search the web',
            'full_profile': 'This is a very long full profile that should NOT be embedded.',
            'usage_scenarios': ['web search'],
            'anti_scenarios': [],
            'complementary_skills': [],
            'triage_triggers': ['search for'],
            'domain': 'Research',
            'effort_tier': 'light',
            'descriptor': 'search',
            'keywords': 'search,web,lookup',
        }

        with patch.object(svc, '_get_llm') as mock_get_llm, \
             patch.object(svc, '_get_embedding_service') as mock_get_emb, \
             patch.object(svc, '_get_related_episodes', return_value=""), \
             patch.object(svc, 'check_staleness', return_value=True), \
             patch.object(svc, '_invalidate_cache'), \
             patch.object(svc, '_load_prompt', return_value="{{manifest}}{{episodes}}{{source_code}}"):

            import json
            from services.llm_service import LLMResponse
            mock_llm = MagicMock()
            mock_llm.send_message.return_value = LLMResponse(
                text=json.dumps(profile_data),
                model='test',
                provider='mock',
            )
            mock_get_llm.return_value = mock_llm

            mock_emb = MagicMock()
            mock_emb.generate_embedding.return_value = [0.1] * 256
            mock_get_emb.return_value = mock_emb

            svc.build_profile("search", {"name": "search", "description": "Web search tool"})

        # Verify the embedding was called with "short_summary keywords" not full_profile
        assert mock_emb.generate_embedding.called
        call_arg = mock_emb.generate_embedding.call_args[0][0]
        assert 'Search the web' in call_arg
        assert 'search,web,lookup' in call_arg or 'search' in call_arg
        assert 'very long full profile' not in call_arg

    def test_seed_builtin_profiles_embeds_summary_plus_keywords(self, db):
        """seed_builtin_profiles() must embed 'short_summary keywords' not full_profile."""
        svc = ToolProfileService(get_shared_db_service())
        embedded_texts = []

        def capture_embed(text):
            embedded_texts.append(text)
            return [0.1] * 256

        with patch.object(svc, '_get_embedding_service') as mock_get_emb, \
             patch.object(svc, '_invalidate_cache'):
            mock_emb = MagicMock()
            mock_emb.generate_embedding.side_effect = capture_embed
            mock_get_emb.return_value = mock_emb
            svc.seed_builtin_profiles()

        assert embedded_texts, "No embeddings were generated"

        # For each generated embedding, it should be short_summary (+ keywords),
        # never a long multi-line full_profile string
        for text in embedded_texts:
            # full_profile values are very long (hundreds of chars with newlines)
            # embedding text must be compact
            assert '\n' not in text, f"Embedding text contains newline (looks like full_profile): {text[:100]!r}"
            # Must not contain the literal full_profile marker used in builtin profiles
            assert 'Wikipedia:' not in text, \
                "Embedding text contains full_profile content ('Wikipedia:' marker)"


class TestBootstrapTOCInjection:
    """Tests for dynamic TOC injection into find_tools TOOL_SCHEMA."""

    def test_bootstrap_injects_toc_when_tools_have_keywords(self, db):
        """After bootstrap_all(), TOOL_SCHEMA query description includes keywords as TOC."""
        import services.innate_skills.find_tools_skill as fts_mod

        svc = ToolProfileService(get_shared_db_service())

        # Save the original description to restore after test
        original_description = fts_mod.TOOL_SCHEMA["input_schema"]["properties"]["query"]["description"]

        # Include 'toc_tool' in registry.tools so the purge logic does not delete its profile.
        # The profile is seeded via seed_builtin_profiles mock (it will be inserted after).
        with patch.object(svc, 'seed_builtin_profiles'), \
             patch('services.tool_registry_service.ToolRegistryService') as mock_reg_cls, \
             patch.object(svc, '_invalidate_cache'):
            mock_registry = MagicMock()
            mock_registry.tools = {
                'toc_tool': {
                    'manifest': {
                        'name': 'toc_tool',
                        'description': 'A tool for TOC injection testing',
                    }
                }
            }
            mock_registry.get_on_demand_tools.return_value = []

            # Intercept check_staleness to prevent build_profile for toc_tool
            original_check = svc.check_staleness
            def _fake_staleness(name, h=None):
                if name == 'toc_tool':
                    return False  # Already profiled
                return original_check(name, h)

            with patch.object(svc, 'check_staleness', side_effect=_fake_staleness), \
                 patch.object(svc, 'get_full_profile', side_effect=lambda n: {'keywords': 'alpha,beta'} if n == 'toc_tool' else None), \
                 patch.object(svc, '_profile_needs_rebuild', return_value=False):
                mock_reg_cls.return_value = mock_registry

                # Insert the profile row with keywords before bootstrap reads it
                db.execute(
                    """INSERT INTO tool_capability_profiles
                       (tool_name, tool_type, short_summary, full_profile, usage_scenarios,
                        anti_scenarios, complementary_skills, manifest_hash, domain,
                        triage_triggers, effort, descriptor, keywords, updated_at)
                       VALUES ('toc_tool', 'tool', 'Test', 'Full', '[]', '[]', '[]',
                               'hash1', 'Research', '[]', 'light', 'toc_tool',
                               'alpha,beta', datetime('now'))"""
                )
                db.commit()

                svc.bootstrap_all()

        try:
            query_description = fts_mod.TOOL_SCHEMA["input_schema"]["properties"]["query"]["description"]
            # TOC should include at least one keyword from our seeded tool
            assert 'alpha' in query_description or 'beta' in query_description, \
                "TOC not injected into find_tools TOOL_SCHEMA after bootstrap"
        finally:
            # Restore original description so other tests are not affected
            fts_mod.TOOL_SCHEMA["input_schema"]["properties"]["query"]["description"] = original_description

    def test_bootstrap_toc_not_injected_when_no_tools_with_keywords(self, db):
        """When no tools have keywords, the original query description is preserved."""
        import services.innate_skills.find_tools_skill as fts_mod

        svc = ToolProfileService(get_shared_db_service())
        original_description = fts_mod.TOOL_SCHEMA["input_schema"]["properties"]["query"]["description"]

        with patch.object(svc, 'seed_builtin_profiles'), \
             patch('services.tool_registry_service.ToolRegistryService') as mock_reg_cls, \
             patch.object(svc, '_invalidate_cache'):
            mock_registry = MagicMock()
            mock_registry.tools = {}
            mock_registry.get_on_demand_tools.return_value = []
            mock_reg_cls.return_value = mock_registry
            svc.bootstrap_all()

        # Description should not have been mutated when there are no tools with keywords
        current_description = fts_mod.TOOL_SCHEMA["input_schema"]["properties"]["query"]["description"]
        assert current_description == original_description
