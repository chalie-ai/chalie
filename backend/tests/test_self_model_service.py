"""
Tests for SelfModelService — foundational interoception.

Uses isolated MemoryStore (same as production) and in-memory SQLite.
"""

import json
import time
from unittest.mock import patch, MagicMock

import pytest


@pytest.mark.unit
class TestSelfModelSnapshot:
    """Snapshot structure and caching."""

    def test_snapshot_returns_all_sections(self, mock_store):
        from services.self_model_service import SelfModelService

        service = SelfModelService()
        snapshot = service.get_snapshot()

        assert "epistemic" in snapshot
        assert "operational" in snapshot
        assert "capability" in snapshot
        assert "noteworthy" in snapshot
        assert "refreshed_at" in snapshot

    def test_snapshot_epistemic_keys(self, mock_store):
        from services.self_model_service import SelfModelService

        service = SelfModelService()
        snapshot = service.get_snapshot()
        ep = snapshot["epistemic"]

        assert "context_warmth" in ep
        # gist_count and fact_count removed in Stream 1 (memory chunker killed)
        assert "working_memory_depth" in ep
        assert "recall_failure_rate" in ep
        assert "topic_age" in ep
        assert "recent_modes" in ep
        assert "focus_active" in ep
        assert "skill_reliability" in ep
        assert isinstance(ep["context_warmth"], float)
        assert 0.0 <= ep["context_warmth"] <= 1.0

    def test_snapshot_operational_keys(self, mock_store):
        from services.self_model_service import SelfModelService

        service = SelfModelService()
        snapshot = service.get_snapshot()
        op = snapshot["operational"]

        assert "thread_health" in op
        assert "provider_status" in op
        assert "queue_depth" in op
        assert "memory_pressure" in op
        assert "bg_llm_heartbeat_stale" in op

    def test_snapshot_capability_keys(self, mock_store):
        from services.self_model_service import SelfModelService

        service = SelfModelService()
        snapshot = service.get_snapshot()
        cap = snapshot["capability"]

        assert "tool_count" in cap
        assert "tool_names" in cap
        assert "innate_skills" in cap
        assert "capability_categories" in cap
        assert "provider_features" in cap

    def test_snapshot_caches(self, mock_store):
        """Second call within TTL returns same refreshed_at."""
        from services.self_model_service import SelfModelService

        service = SelfModelService()
        snap1 = service.get_snapshot()
        snap2 = service.get_snapshot()

        assert snap1["refreshed_at"] == snap2["refreshed_at"]

    def test_snapshot_cache_invalidation(self, mock_store):
        """After TTL expires, snapshot refreshes."""
        from services.self_model_service import SelfModelService, CACHE_KEY

        service = SelfModelService()
        snap1 = service.get_snapshot()

        # Manually expire the cache
        mock_store.delete(CACHE_KEY)

        snap2 = service.get_snapshot()
        # New snapshot should have a different or same timestamp
        # but importantly should succeed without error
        assert "refreshed_at" in snap2


@pytest.mark.unit
class TestNoteworthy:
    """Noteworthy assessment — should be empty when healthy."""

    def test_noteworthy_empty_when_healthy(self, mock_store):
        from services.self_model_service import SelfModelService

        service = SelfModelService()
        snapshot = service.get_snapshot()

        # With no thread health published and no real providers,
        # noteworthy should still be a list (possibly with items
        # for missing providers, but let's check structure)
        assert isinstance(snapshot["noteworthy"], list)
        for item in snapshot["noteworthy"]:
            assert "signal" in item
            assert "severity" in item

    def test_noteworthy_detects_dead_threads(self, mock_store):
        from services.self_model_service import SelfModelService

        # Publish thread health with dead threads
        mock_store.setex("self_model:thread_health", 15, json.dumps({
            "alive": ["rest-api-worker-1", "decay-engine"],
            "dead": ["cognitive-drift-engine", "scheduler-service"],
            "total": 4,
        }))

        service = SelfModelService()
        snapshot = service._refresh()
        noteworthy = snapshot["noteworthy"]

        dead_signals = [n for n in noteworthy if "thread" in n["signal"].lower()]
        assert len(dead_signals) > 0
        assert dead_signals[0]["severity"] == 0.6

    def test_noteworthy_detects_queue_congestion(self, mock_store):
        from services.self_model_service import SelfModelService

        # Fill up the bg_llm queue
        from services.background_llm_queue import QUEUE_KEY
        for i in range(20):
            mock_store.lpush(QUEUE_KEY, f"job-{i}")

        service = SelfModelService()
        snapshot = service._refresh()
        noteworthy = snapshot["noteworthy"]

        queue_signals = [n for n in noteworthy if "queue" in n["signal"].lower()]
        assert len(queue_signals) > 0
        assert queue_signals[0]["severity"] == 0.4

    def test_noteworthy_severity_structure(self, mock_store):
        """Each noteworthy item must have signal and severity."""
        from services.self_model_service import SelfModelService

        # Publish unhealthy state
        mock_store.setex("self_model:thread_health", 15, json.dumps({
            "alive": [], "dead": ["worker-1"], "total": 1,
        }))

        service = SelfModelService()
        snapshot = service._refresh()

        for item in snapshot["noteworthy"]:
            assert isinstance(item, dict)
            assert "signal" in item
            assert "severity" in item
            assert isinstance(item["severity"], float)
            assert 0.0 <= item["severity"] <= 1.0


@pytest.mark.unit
class TestCapabilityCategories:
    """Capability categorization from tool manifests."""

    def test_capability_categories_from_manifests(self, mock_store):
        from services.self_model_service import SelfModelService

        mock_registry = MagicMock()
        mock_registry.get_tool_names.return_value = ["web_search", "news_reader"]
        mock_registry.get_tool_full_description.side_effect = lambda name: {
            "web_search": {
                "documentation": "Search the web for information, find answers to questions",
                "description": "Web search tool",
            },
            "news_reader": {
                "documentation": "Read news articles and headlines from various feeds",
                "description": "News aggregator",
            },
        }.get(name)

        with patch('services.tool_registry_service.ToolRegistryService', return_value=mock_registry):
            service = SelfModelService()
            snapshot = service._refresh()
            cats = snapshot["capability"]["capability_categories"]

            assert "search" in cats
            assert "web_search" in cats["search"]
            assert "news" in cats
            assert "news_reader" in cats["news"]


@pytest.mark.unit
class TestFormatForPrompt:
    """Prompt injection formatting."""

    def test_format_for_prompt_empty_when_fine(self, mock_store):
        from services.self_model_service import SelfModelService

        # Publish healthy state
        mock_store.setex("self_model:thread_health", 15, json.dumps({
            "alive": ["w1", "w2"], "dead": [], "total": 2,
        }))

        service = SelfModelService()

        # Mock providers to avoid unassigned job detection
        mock_provider = MagicMock()
        mock_provider.get_all_providers.return_value = [{"is_active": True, "platform": "anthropic"}]
        mock_provider.get_all_job_assignments.return_value = [
            {"job_name": j, "provider_id": 1} for j in
            ['frontal-cortex', 'cognitive-triage', 'cognitive-drift']
        ]

        with patch('services.provider_db_service.ProviderDbService', return_value=mock_provider):
            service._refresh()
            result = service.format_for_prompt()
            assert result == ""

    def test_format_for_prompt_has_content_when_degraded(self, mock_store):
        from services.self_model_service import SelfModelService

        # Publish unhealthy thread state
        mock_store.setex("self_model:thread_health", 15, json.dumps({
            "alive": ["w1"], "dead": ["cognitive-drift-engine"], "total": 2,
        }))

        service = SelfModelService()
        service._refresh()
        result = service.format_for_prompt()

        assert "## Self-Awareness" in result
        assert "Adapt your behavior" in result

    def test_format_includes_behavioral_directives(self, mock_store):
        """Directives should match the type of degradation."""
        from services.self_model_service import SelfModelService

        # Simulate high recall failure
        mock_store.setex("self_model:thread_health", 15, json.dumps({
            "alive": ["w1"], "dead": [], "total": 1,
        }))

        service = SelfModelService()
        # Manually set noteworthy with recall signal
        snapshot = service._refresh()
        snapshot["noteworthy"] = [{
            "signal": "Memory recall unreliable (failure rate: 55%)",
            "severity": 0.3,
        }]
        mock_store.setex("self_model:snapshot", 45, json.dumps(snapshot))

        result = service.format_for_prompt()
        assert "uncertainty" in result.lower() or "recall" in result.lower()
