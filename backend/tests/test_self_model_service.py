"""
Tests for SelfModelService — foundational interoception.

Uses real MemoryStore (via mock_store fixture) and mock DB (via mock_db_rows).
SelfModelService aggregates signals from many sources — we test the aggregation
logic, noteworthy detection, caching/TTL, memory richness, and prompt formatting.
"""

import json
import time
from unittest.mock import patch, MagicMock

import pytest


def _make_service(db=None):
    """Create SelfModelService with optional DB override."""
    from services.self_model_service import SelfModelService
    return SelfModelService(db_service=db)


def _populate_healthy_state(store):
    """Set up MemoryStore to reflect a healthy system."""
    store.setex("self_model:thread_health", 60, json.dumps({
        "alive": ["rest-api-worker", "decay-engine", "episodic-memory"],
        "dead": [],
        "total": 3,
    }))
    store.set("recent_topic", "cooking")
    store.expire("recent_topic", 1800)
    for i in range(3):
        store.lpush("working_memory:cooking", json.dumps({"role": "user", "text": f"msg {i}"}))
    store.set("fok:cooking", "3")


def _mock_providers_assigned():
    """Return a provider mock where all critical jobs are assigned."""
    mock_provider = MagicMock()
    mock_provider.get_all_providers.return_value = [
        {"is_active": True, "platform": "anthropic"},
    ]
    mock_provider.get_all_job_assignments.return_value = [
        {"job_name": "frontal-cortex", "provider_id": 1},
        {"job_name": "cognitive-triage", "provider_id": 1},
        {"job_name": "cognitive-drift", "provider_id": 1},
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
        """Working memory entries and FOK in MemoryStore drive epistemic signals."""
        mock_store.set("recent_topic", "python")
        mock_store.expire("recent_topic", 1800)
        for i in range(4):
            mock_store.lpush("working_memory:python", f"turn-{i}")
        mock_store.set("fok:python", "5")

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

    def test_memory_pressure_from_db(self, mock_store, mock_db_rows):
        """_get_memory_pressure reads episode/concept/trait counts from DB."""
        db, cursor = mock_db_rows
        cursor.fetchone.side_effect = [
            (42,),   # episodes count
            (15,),   # concepts count
            (8,),    # traits count
            (0.65,), # avg activation
        ]

        svc = _make_service(db=db)
        pressure = svc._get_memory_pressure()

        assert pressure["episode_count"] == 42
        assert pressure["concept_count"] == 15
        assert pressure["trait_count"] == 8
        assert pressure["avg_activation"] == 0.65

    def test_queue_depth_from_store(self, mock_store):
        """Queue depths reflect actual list lengths in MemoryStore."""
        from services.background_llm_queue import QUEUE_KEY

        for i in range(7):
            mock_store.lpush(QUEUE_KEY, f"job-{i}")
        for i in range(3):
            mock_store.lpush("prompt-queue", f"task-{i}")

        svc = _make_service()
        qd = svc.get_snapshot()["operational"]["queue_depth"]

        assert qd["bg_llm"] == 7
        assert qd["prompt_queue"] == 3


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
            "dead": ["cognitive-drift-engine", "scheduler-service"],
            "total": 3,
        }))

        svc = _make_service()
        noteworthy = svc._refresh()["noteworthy"]
        dead_signals = [n for n in noteworthy if "thread" in n["signal"].lower()]

        assert len(dead_signals) == 1
        assert dead_signals[0]["severity"] == 0.6
        assert "cognitive-drift-engine" in dead_signals[0]["signal"]

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
                "trait_count": 2, "avg_activation": 0.7,
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
        snap_cold = svc._refresh()
        richness_cold = svc.get_memory_richness()

        # Add working memory to raise context_warmth
        mock_store.set("recent_topic", "music")
        mock_store.expire("recent_topic", 1800)
        for i in range(4):
            mock_store.lpush("working_memory:music", f"turn-{i}")
        mock_store.set("fok:music", "5")
        mock_store.delete(CACHE_KEY)

        richness_warm = svc.get_memory_richness()
        assert richness_warm > richness_cold


@pytest.mark.unit
class TestCapabilityCategories:
    """Tool categorization from manifest keywords."""

    def test_tools_categorized_by_keywords(self, mock_store):
        mock_registry = MagicMock()
        mock_registry.get_tool_names.return_value = ["web_search", "news_reader"]
        mock_registry.get_tool_full_description.side_effect = lambda name: {
            "web_search": {
                "documentation": "Search the web for information and find answers",
                "description": "Web search tool",
            },
            "news_reader": {
                "documentation": "Read news articles and headlines from feeds",
                "description": "News aggregator",
            },
        }.get(name)

        with patch('services.tool_registry_service.ToolRegistryService', return_value=mock_registry):
            svc = _make_service()
            cats = svc._refresh()["capability"]["capability_categories"]

        assert "search" in cats
        assert "web_search" in cats["search"]
        assert "news" in cats
        assert "news_reader" in cats["news"]

    def test_innate_skills_from_registry(self, mock_store):
        """Capability section includes innate skills from authoritative registry."""
        svc = _make_service()
        cap = svc.get_snapshot()["capability"]

        assert len(cap["innate_skills"]) > 0
        assert "recall" in cap["innate_skills"]
        assert "memorize" in cap["innate_skills"]
        assert cap["innate_skills"] == sorted(cap["innate_skills"])
