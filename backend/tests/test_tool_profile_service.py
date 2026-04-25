"""Unit tests for ToolProfileService."""
import pytest
from unittest.mock import MagicMock, patch

from services.tool_profile_service import (
    ToolProfileService,
    _compute_manifest_hash,
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
        'anti_scenarios': '[]',
        'complementary_skills': '["recall"]',
        'manifest_hash': 'test_hash',
        'domain': 'Other',
        'effort': 'moderate',
        'descriptor': tool_name,
        'keywords': 'test,keywords',
    }
    defaults.update(overrides)
    db.execute("""
        INSERT OR REPLACE INTO tool_capability_profiles
            (tool_name, tool_type, short_summary, full_profile,
             anti_scenarios, complementary_skills, manifest_hash, domain,
             effort, descriptor, keywords, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
    """, (
        tool_name,
        defaults['tool_type'],
        defaults['short_summary'],
        defaults['full_profile'],
        defaults['anti_scenarios'],
        defaults['complementary_skills'],
        defaults['manifest_hash'],
        defaults['domain'],
        defaults['effort'],
        defaults['descriptor'],
        defaults['keywords'],
    ))
    db.commit()


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
                      anti_scenarios='[]',
                      complementary_skills='["recall"]')

        svc = ToolProfileService(get_shared_db_service())
        profile = svc.get_full_profile("test_tool")
        assert profile is not None
        assert profile['tool_name'] == 'test_tool'
        assert isinstance(profile['anti_scenarios'], list)


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
                      effort='moderate')
        _seed_profile(db, 'weather',
                      tool_type='tool',
                      short_summary='Check weather',
                      domain='Environment',
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
        """Seeded rows pass _profile_needs_rebuild — domain, descriptor, keywords non-empty."""
        svc = ToolProfileService(get_shared_db_service())

        with patch.object(svc, '_get_embedding_service') as mock_get_emb, \
             patch.object(svc, '_invalidate_cache'):
            _patch_embedding(mock_get_emb)
            svc.seed_builtin_profiles()

        for tool_name in BUILTIN_TOOL_PROFILES:
            profile = svc.get_full_profile(tool_name)
            assert profile is not None
            # _profile_needs_rebuild checks domain, descriptor, keywords
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


class TestProfileNeedsRebuild:
    """Tests for ToolProfileService._profile_needs_rebuild()."""

    def _full_profile(self, **overrides):
        """Return a minimal valid profile dict that passes all rebuild checks."""
        base = {
            'tool_type': 'tool',
            'domain': 'Research',
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


