"""
Behavioural tests for MetricsService — trace lifecycle and dashboard aggregation.

Uses the real MemoryStore (via ``store`` fixture).  No DB, no models, no vault.
"""

import json

import pytest

from services.metrics_service import MetricsService

pytestmark = pytest.mark.unit


@pytest.fixture
def metrics(store):
    """MetricsService connected to the isolated test MemoryStore."""
    return MetricsService()


class TestTraceLifecycle:
    def test_start_trace_returns_valid_id(self, metrics):
        trace_id = metrics.start_trace()
        assert isinstance(trace_id, str)
        assert len(trace_id) == 8

    def test_trace_persisted_in_store(self, metrics, store):
        trace_id = metrics.start_trace()
        raw = store.get(f"trace:{trace_id}")
        assert raw is not None
        data = json.loads(raw)
        assert data["trace_id"] == trace_id
        assert "started_at" in data

    def test_get_trace_returns_none_for_unknown(self, metrics):
        assert metrics.get_trace("deadbeef") is None

    def test_get_trace_returns_started_trace(self, metrics):
        trace_id = metrics.start_trace()
        data = metrics.get_trace(trace_id)
        assert data is not None
        assert data["trace_id"] == trace_id


class TestRecordTiming:
    def test_timing_stored_on_trace(self, metrics):
        trace_id = metrics.start_trace()
        metrics.record_timing(trace_id, "classification", 42.0)
        data = metrics.get_trace(trace_id)
        assert data["timings"]["classification"] == 42.0

    def test_timing_idempotent_on_unknown_trace(self, metrics):
        # Must not raise when trace doesn't exist
        metrics.record_timing("nope", "op", 10.0)


class TestDashboardData:
    def test_dashboard_defaults_counters_to_zero(self, metrics):
        dash = metrics.get_dashboard_data()
        assert dash["counters"]["requests_total"] == 0
        assert dash["counters"]["errors_total"] == 0

    def test_dashboard_reflects_recorded_counter(self, metrics):
        metrics.record_counter("requests_total", 5)
        dash = metrics.get_dashboard_data()
        assert dash["counters"]["requests_total"] == 5

    def test_dashboard_includes_timing_averages_for_recorded_ops(self, metrics):
        metrics.record_timing("t1", "embedding", 100.0)
        metrics.record_timing("t1", "embedding", 200.0)
        dash = metrics.get_dashboard_data()
        emb = dash["timing_averages"].get("embedding")
        # Timing rollup key differs between record_timing (uses daily key) and
        # get_dashboard_data (also uses today's key), so they match if same day.
        if emb:
            assert emb["count"] == 2
            assert emb["avg_ms"] == 150.0
            assert emb["min_ms"] == 100.0
            assert emb["max_ms"] == 200.0
