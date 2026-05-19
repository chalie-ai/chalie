"""
Tests for SelfModelService — foundational interoception.

Uses real MemoryStore (via mock_store fixture) and real DB (via db fixture).
SelfModelService aggregates signals from many sources — we test the aggregation
logic, noteworthy detection, caching/TTL, memory richness, and prompt formatting.
"""

import json
from unittest.mock import patch, MagicMock

import pytest

# Patch get_shared_db_service for every test in this module so that any service
# instantiated inside SelfModelService (e.g. ToolRegistryService, ToolConfigService)
# uses the in-memory test DB rather than the real chalie.db on the SMB mount.
pytestmark = pytest.mark.usefixtures('db')


def _make_service(db=None):
    """Create SelfModelService with optional DB override.

    With ``pytestmark`` applying the ``db`` fixture to every test, passing
    ``db=None`` is safe: ``_get_db()`` will fall through to the patched
    ``get_shared_db_service()`` singleton rather than opening chalie.db.
    Pass ``db`` explicitly only when you need to verify DB-backed behaviour.
    """
    from services.self_model_service import SelfModelService
    return SelfModelService(db_service=db)


def _populate_healthy_state(store):
    """Set up MemoryStore to reflect a healthy system.

    Seeds working-memory and FOK signals under the ``"general"`` topic key,
    which is the hardcoded default used by ``SelfModelService._gather_epistemic``
    now that the dynamic ``recent_topic`` look-up has been removed.
    """
    store.setex("self_model:thread_health", 60, json.dumps({
        "alive": ["rest-api-worker", "decay-engine", "episodic-memory"],
        "dead": [],
        "total": 3,
    }))
    for i in range(3):
        store.lpush("working_memory:general", json.dumps({"role": "user", "text": f"msg {i}"}))
    store.set("fok:general", "3")


def _mock_providers_assigned():
    """Return a provider mock where a provider is selected and active."""
    mock_provider = MagicMock()
    mock_provider.get_all_providers.return_value = [
        {"is_active": True, "platform": "anthropic"},
    ]
    mock_provider.get_selected_provider.return_value = {
        "id": 1, "is_active": True, "platform": "anthropic", "model": "claude-sonnet-4-6",
    }
    return mock_provider


@pytest.mark.unit
class TestGetSnapshot:
    """Snapshot structure, caching, and TTL behaviour."""

    def test_snapshot_contains_all_top_level_sections(self, mock_store):
        svc = _make_service()
        snapshot = svc.get_snapshot()

        assert set(snapshot.keys()) >= {
            "epistemic", "operational", "capability", "noteworthy", "refreshed_at",
        }

    def test_snapshot_populates_epistemic_from_store(self, mock_store):
        """Working memory entries and FOK in MemoryStore drive epistemic signals.

        Data is seeded under the ``"general"`` topic key because
        ``_gather_epistemic`` no longer reads ``recent_topic`` from the store —
        it uses ``"general"`` as a hardcoded default.
        """
        for i in range(4):
            mock_store.lpush("working_memory:general", f"turn-{i}")
        mock_store.set("fok:general", "5")

        svc = _make_service()
        ep = svc.get_snapshot()["epistemic"]

        assert ep["working_memory_depth"] == 4
        assert ep["partial_match_signal"] == 5
        # warmth formula: 0.6 * clamped(working_memory/max) + 0.4 * clamped(fok/max), yielding 1.0 here
        assert ep["context_warmth"] == pytest.approx(1.0, abs=1e-9)

    def test_snapshot_caches_in_store(self, mock_store):
        """Second call within TTL returns identical snapshot without re-computing."""
        svc = _make_service()
        snap1 = svc.get_snapshot()
        snap2 = svc.get_snapshot()

        assert snap1["refreshed_at"] == snap2["refreshed_at"]

    def test_memory_pressure_from_db(self, mock_store, db):
        """_get_memory_pressure reads episode/concept/trait counts from DB."""

        # Seed episodes (42 total)
        for i in range(42):
            db.execute(
                "INSERT INTO episodes (id, gist, salience, channel) VALUES (?, ?, ?, 'test')",
                (f"ep-{i}", f"gist {i}", 5),
            )

        # Seed 15 user_specific rows in data_graph.
        # _get_memory_pressure counts kind='user_specific' for both concept_count
        # and trait_count (identical queries), so both will equal 15.
        for i in range(15):
            db.execute(
                "INSERT INTO data_graph (kind, key, value, source)"
                " VALUES ('user_specific', ?, ?, 'test')",
                (f"concept_{i}", f"value_{i}"),
            )

        db.commit()

        # db fixture yields raw sqlite3.Connection for seeding; get_shared_db_service()
        # returns the DatabaseService singleton patched by the fixture.
        from services.database_service import get_shared_db_service
        svc = _make_service(get_shared_db_service())
        pressure = svc._get_memory_pressure()

        assert pressure["episode_count"] == 42
        assert pressure["concept_count"] == 15
        assert pressure["trait_count"] == 15
        assert "avg_activation" not in pressure

@pytest.mark.unit
class TestNoteworthy:
    """Noteworthy detection fires on real degraded state, stays empty when healthy."""

    def test_healthy_system_no_noteworthy(self, mock_store):
        """When threads are alive and providers assigned, noteworthy is empty."""
        _populate_healthy_state(mock_store)

        mock_provider = _mock_providers_assigned()
        with patch('services.provider_db_service.ProviderDbService', return_value=mock_provider):
            svc = _make_service()
            snapshot = svc._refresh()

        assert snapshot["noteworthy"] == []

    def test_dead_threads_trigger_noteworthy(self, mock_store):
        """Dead worker threads produce a signal with severity 0.6."""
        mock_store.setex("self_model:thread_health", 60, json.dumps({
            "alive": ["rest-api"],
            "dead": ["dmn-service", "scheduler-service"],
            "total": 3,
        }))

        svc = _make_service()
        noteworthy = svc._refresh()["noteworthy"]
        dead_signals = [n for n in noteworthy if "thread" in n["signal"].lower()]

        assert len(dead_signals) == 1
        assert dead_signals[0]["severity"] == pytest.approx(0.6, abs=1e-9)
        assert "dmn-service" in dead_signals[0]["signal"]


@pytest.mark.unit
class TestFormatForPrompt:
    """Prompt formatting: empty when healthy, structured when degraded."""

    def test_includes_header_and_directives_when_degraded(self, mock_store):
        """Dead threads produce a prompt with Self-Awareness header and directives."""
        mock_store.setex("self_model:thread_health", 60, json.dumps({
            "alive": ["w1"], "dead": ["drift-engine"], "total": 2,
        }))

        svc = _make_service()
        svc._refresh()
        result = svc.format_for_prompt()

        assert "## Self-Awareness" in result
        assert "thread" in result.lower()
        assert "Adapt your behavior" in result

    def test_memory_directive_on_recall_failure(self, mock_store):
        """High recall failure triggers memory uncertainty directive."""
        from services.self_model_service import CACHE_KEY

        svc = _make_service()
        snapshot = svc._refresh()
        snapshot["noteworthy"] = [{
            "signal": "Memory recall unreliable for current topic (failure rate: 55%)",
            "severity": 0.3,
        }]
        mock_store.setex(CACHE_KEY, 45, json.dumps(snapshot))

        result = svc.format_for_prompt()
        assert "recall" in result.lower() or "uncertainty" in result.lower()
        assert "## Self-Awareness" in result


@pytest.mark.unit
class TestMemoryRichness:
    """Memory richness score: logarithmic composite from snapshot data."""

    def test_richness_increases_with_context_warmth(self, mock_store):
        """Working memory depth drives context_warmth which contributes to richness."""
        from services.self_model_service import CACHE_KEY

        svc = _make_service()
        # Cold snapshot
        svc._refresh()
        richness_cold = svc.get_memory_richness()

        # Add working memory to raise context_warmth — seed under "general"
        # because _gather_epistemic uses "general" as its hardcoded topic key.
        for i in range(4):
            mock_store.lpush("working_memory:general", f"turn-{i}")
        mock_store.set("fok:general", "5")
        mock_store.delete(CACHE_KEY)

        richness_warm = svc.get_memory_richness()
        assert richness_warm > richness_cold


@pytest.mark.unit
class TestCapabilityCategories:
    """Tool categorization from manifest keywords."""

    def test_tools_categorized_by_keywords(self, mock_store):
        """Capabilities are bucketed into the correct CATEGORY_KEYWORDS categories.

        Uses the real AbilityRegistry — no mocks.  The assertions are grounded in
        actual SUMMARY strings on live Ability subclasses:

        - 'search' ability SUMMARY contains "search" → lands in 'search' category
        - 'news' ability SUMMARY contains "news" → lands in 'news' category
        - 'schedule' ability SUMMARY contains "schedule" → lands in 'productivity'
        - 'code_eval' ability SUMMARY contains "code" → lands in 'development'
        """
        svc = _make_service()
        cats = svc._refresh()["capability"]["capability_categories"]

        assert "search" in cats, "Expected 'search' category from abilities with search/find/query keywords"
        assert "search" in cats["search"], "'search' ability must appear in search category"

        assert "news" in cats, "Expected 'news' category from 'news' ability"
        assert "news" in cats["news"], "'news' ability must appear in news category"

        assert "productivity" in cats, "Expected 'productivity' category (schedule ability has 'schedule' keyword)"
        assert "schedule" in cats["productivity"], "'schedule' ability must appear in productivity category"

        assert "development" in cats, "Expected 'development' category (code_eval ability has 'code' keyword)"
        assert "code_eval" in cats["development"], "'code_eval' ability must appear in development category"


