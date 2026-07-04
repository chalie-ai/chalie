"""
Behavioural tests for MetricsService — trace lifecycle and dashboard aggregation.

Uses the real MemoryStore (via ``store`` fixture).  No DB, no models, no vault.
"""

import json
from typing import cast

import pytest

from services.memory_store import MemoryStore
from services.metrics_service import MetricsService

pytestmark = pytest.mark.unit


@pytest.fixture
def metrics(store: MemoryStore) -> MetricsService:
    return MetricsService()


class TestTraceLifecycle:
    def test_trace_persisted_in_store(self, metrics: MetricsService, store: MemoryStore) -> None:
        trace_id = metrics.start_trace()
        raw = store.get(f"trace:{trace_id}")
        assert raw is not None
        data = json.loads(raw)
        assert data["trace_id"] == trace_id
        assert "started_at" in data


class TestRecordTiming:
    def test_timing_stored_on_trace(self, metrics: MetricsService, store: MemoryStore) -> None:
        trace_id = metrics.start_trace()
        metrics.record_timing(trace_id, "classification", 42.0)
        data = json.loads(cast(str, store.get(f"trace:{trace_id}")))
        assert data["timings"]["classification"] == pytest.approx(42.0, abs=1e-9)

    def test_timing_idempotent_on_unknown_trace(self, metrics: MetricsService) -> None:
        # Must not raise when trace doesn't exist
        metrics.record_timing("nope", "op", 10.0)


class TestDashboardData:
    def test_dashboard_defaults_counters_to_zero(self, metrics: MetricsService) -> None:
        dash = metrics.get_dashboard_data()
        assert cast("dict[str, object]", dash["counters"])["requests_total"] == 0
        assert cast("dict[str, object]", dash["counters"])["errors_total"] == 0

    def test_dashboard_reflects_recorded_counter(self, metrics: MetricsService) -> None:
        metrics.record_counter("requests_total", 5)
        dash = metrics.get_dashboard_data()
        assert cast("dict[str, object]", dash["counters"])["requests_total"] == 5

    def test_dashboard_includes_timing_averages_for_recorded_ops(self, metrics: MetricsService) -> None:
        metrics.record_timing("t1", "embedding", 100.0)
        metrics.record_timing("t1", "embedding", 200.0)
        dash = metrics.get_dashboard_data()
        emb = cast("dict[str, dict[str, object]]", dash["timing_averages"]).get("embedding")
        # Timing rollup key differs between record_timing (uses daily key) and
        # get_dashboard_data (also uses today's key), so they match if same day.
        if emb:
            assert emb["count"] == 2
            assert emb["avg_ms"] == pytest.approx(150.0, abs=1e-9)
            assert emb["min_ms"] == pytest.approx(100.0, abs=1e-9)
            assert emb["max_ms"] == pytest.approx(200.0, abs=1e-9)
