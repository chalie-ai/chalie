"""
Tests for SelfModelService — foundational interoception.

Uses real MemoryStore (via mock_store fixture) and real DB (via db fixture).
SelfModelService aggregates signals from many sources — we test the aggregation
logic, noteworthy detection, caching/TTL, memory richness, and prompt formatting.
"""

import json
import time
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
    """Return a provider mock where all critical jobs are assigned."""
    mock_provider = MagicMock()
    mock_provider.get_all_providers.return_value = [
        {"is_active": True, "platform": "anthropic"},
    ]
    mock_provider.get_all_job_assignments.return_value = [
        {"job_name": "frontal-cortex-unified", "provider_id": 1},
        {"job_name": "cognitive-triage", "provider_id": 1},
    ]
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
        # warmth = 0.6 * min(1, 4/4) + 0.4 * min(1, 5/5) = 1.0
        assert ep["context_warmth"] == 1.0

    def test_snapshot_epistemic_cold_when_empty(self, mock_store):
        """No working memory and no FOK yields zero warmth."""
        svc = _make_service()
        ep = svc.get_snapshot()["epistemic"]

        assert ep["working_memory_depth"] == 0
        assert ep["partial_match_signal"] == 0
        assert ep["context_warmth"] == 0.0

    def test_snapshot_caches_in_store(self, mock_store):
        """Second call within TTL returns identical snapshot without re-computing."""
        svc = _make_service()
        snap1 = svc.get_snapshot()
        snap2 = svc.get_snapshot()

        assert snap1["refreshed_at"] == snap2["refreshed_at"]

    def test_snapshot_refreshes_after_cache_eviction(self, mock_store):
        """After manually evicting cache, get_snapshot re-computes."""
        from services.self_model_service import CACHE_KEY

        svc = _make_service()
        snap1 = svc.get_snapshot()
        ts1 = snap1["refreshed_at"]

        mock_store.delete(CACHE_KEY)
        time.sleep(0.01)

        snap2 = svc.get_snapshot()
        assert snap2["refreshed_at"] != ts1

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

    def test_queue_depth_from_store(self, mock_store):
        """Queue depths reflect actual list lengths in MemoryStore."""
        from services.background_llm_queue import QUEUE_KEY

        for i in range(7):
            mock_store.lpush(QUEUE_KEY, f"job-{i}")

        svc = _make_service()
        qd = svc.get_snapshot()["operational"]["queue_depth"]

        assert qd["bg_llm"] == 7


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
        assert dead_signals[0]["severity"] == 0.6
        assert "dmn-service" in dead_signals[0]["signal"]

    def test_queue_congestion_triggers_noteworthy(self, mock_store):
        """Queue depth >15 fires a congestion signal with severity 0.4."""
        from services.background_llm_queue import QUEUE_KEY

        for i in range(20):
            mock_store.lpush(QUEUE_KEY, f"job-{i}")

        svc = _make_service()
        noteworthy = svc._refresh()["noteworthy"]
        queue_signals = [n for n in noteworthy if "queue" in n["signal"].lower()]

        assert len(queue_signals) == 1
        assert queue_signals[0]["severity"] == 0.4
        assert "20" in queue_signals[0]["signal"]

    def test_stale_heartbeat_triggers_noteworthy(self, mock_store):
        """Background LLM heartbeat older than threshold fires signal."""
        from services.background_llm_queue import HEARTBEAT_KEY

        mock_store.set(HEARTBEAT_KEY, str(time.time() - 60))

        svc = _make_service()
        noteworthy = svc._refresh()["noteworthy"]
        hb_signals = [n for n in noteworthy if "heartbeat" in n["signal"].lower()]

        assert len(hb_signals) == 1
        assert hb_signals[0]["severity"] == 0.5

    def test_no_congestion_below_threshold(self, mock_store):
        """Queue depth <=15 does NOT produce congestion signal."""
        from services.background_llm_queue import QUEUE_KEY

        for i in range(10):
            mock_store.lpush(QUEUE_KEY, f"job-{i}")

        svc = _make_service()
        noteworthy = svc._refresh()["noteworthy"]
        queue_signals = [n for n in noteworthy if "queue" in n["signal"].lower()]

        assert len(queue_signals) == 0

    def test_noteworthy_item_structure(self, mock_store):
        """Every noteworthy item has signal (str) and severity (float 0-1)."""
        mock_store.setex("self_model:thread_health", 60, json.dumps({
            "alive": [], "dead": ["worker-1", "worker-2"], "total": 2,
        }))

        svc = _make_service()
        noteworthy = svc._refresh()["noteworthy"]

        assert len(noteworthy) > 0
        for item in noteworthy:
            assert isinstance(item["signal"], str)
            assert isinstance(item["severity"], float)
            assert 0.0 <= item["severity"] <= 1.0


@pytest.mark.unit
class TestFormatForPrompt:
    """Prompt formatting: empty when healthy, structured when degraded."""

    def test_empty_string_when_healthy(self, mock_store):
        """No degradation => empty string => zero token cost."""
        _populate_healthy_state(mock_store)

        mock_provider = _mock_providers_assigned()
        with patch('services.provider_db_service.ProviderDbService', return_value=mock_provider):
            svc = _make_service()
            svc._refresh()

        assert svc.format_for_prompt() == ""

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

    def test_zero_when_no_data(self, mock_store):
        """Empty system yields zero richness."""
        svc = _make_service()
        assert svc.get_memory_richness() == 0.0

    def test_nonzero_with_episodes(self, mock_store):
        """Even a few episodes push richness above zero due to log curve."""
        from services.self_model_service import CACHE_KEY

        # Seed the cache with a snapshot that has episode/concept/trait counts
        snapshot = {
            "epistemic": {"context_warmth": 0.3},
            "operational": {"memory_pressure": {
                "episode_count": 5, "concept_count": 3,
                "trait_count": 2,
            }},
            "capability": {},
            "noteworthy": [],
            "refreshed_at": "2026-01-01T00:00:00+00:00",
        }
        mock_store.setex(CACHE_KEY, 45, json.dumps(snapshot))

        svc = _make_service()
        richness = svc.get_memory_richness()

        assert richness > 0.0
        assert richness <= 1.0

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

    def test_innate_skills_from_registry(self, mock_store):
        """Capability section includes innate skills from authoritative registry."""
        svc = _make_service()
        cap = svc.get_snapshot()["capability"]

        assert len(cap["innate_skills"]) > 0
        assert "memory" in cap["innate_skills"]
        assert cap["innate_skills"] == sorted(cap["innate_skills"])
