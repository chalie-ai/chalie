"""Unit tests for ToolPerformanceService."""
import pytest

from services.tool_performance_service import ToolPerformanceService
from services.database_service import get_shared_db_service

pytestmark = pytest.mark.unit


class TestRecordInvocation:
    def test_records_successful_invocation(self, db):
        svc = ToolPerformanceService(get_shared_db_service())

        svc.record_invocation("duckduckgo_search", "exch_123", True, 250.0)

        row = db.execute(
            "SELECT * FROM tool_performance_metrics WHERE tool_name = ?",
            ("duckduckgo_search",)
        ).fetchone()
        assert row is not None
        assert row['invocation_success'] == 1
        assert row['latency_ms'] == pytest.approx(250.0)

    def test_records_failed_invocation(self, db):
        svc = ToolPerformanceService(get_shared_db_service())

        svc.record_invocation("weather", "exch_456", False, 5100.0)

        row = db.execute(
            "SELECT * FROM tool_performance_metrics WHERE tool_name = ?",
            ("weather",)
        ).fetchone()
        assert row is not None
        assert row['invocation_success'] == 0
        assert row['latency_ms'] == pytest.approx(5100.0)


class TestGetToolStats:
    def test_returns_defaults_for_no_data(self, db):
        svc = ToolPerformanceService(get_shared_db_service())

        stats = svc.get_tool_stats("unknown_tool")
        assert 'success_rate' in stats
        assert stats['success_rate'] == 0.5  # default

    def test_computes_correct_success_rate(self, db):
        svc = ToolPerformanceService(get_shared_db_service())

        # Seed 10 invocations: 8 success, 2 failure
        for i in range(8):
            svc.record_invocation("good_tool", f"exch_{i}", True, 300.0)
        for i in range(8, 10):
            svc.record_invocation("good_tool", f"exch_{i}", False, 300.0)

        stats = svc.get_tool_stats("good_tool")
        assert stats['success_rate'] == pytest.approx(0.8)
        assert stats['avg_latency'] == pytest.approx(300.0)


class TestRankCandidates:
    def test_returns_sorted_by_score(self, db):
        svc = ToolPerformanceService(get_shared_db_service())

        # Mock get_tool_stats, _get_user_preference, _get_reliability
        def mock_stats(tool_name, days=30):
            return {
                'duckduckgo_search': {'success_rate': 0.9, 'avg_latency': 200, 'avg_cost': 0},
                'weather': {'success_rate': 0.5, 'avg_latency': 800, 'avg_cost': 0},
            }.get(tool_name, {'success_rate': 0.5, 'avg_latency': 0, 'avg_cost': 0})

        svc.get_tool_stats = mock_stats
        svc._get_user_preference = lambda tool: 0.0
        svc._get_reliability = lambda tool: 1.0

        result = svc.rank_candidates(['weather', 'duckduckgo_search'])
        assert len(result) == 2
        # duckduckgo_search should rank higher (better success rate, lower latency)
        assert result[0]['name'] == 'duckduckgo_search'

    def test_empty_candidates_returns_empty(self, db):
        svc = ToolPerformanceService(get_shared_db_service())
        assert svc.rank_candidates([]) == []

    def test_single_candidate_returned(self, db):
        svc = ToolPerformanceService(get_shared_db_service())

        svc.get_tool_stats = lambda tool, days=30: {'success_rate': 0.7, 'avg_latency': 400, 'avg_cost': 0}
        svc._get_user_preference = lambda tool: 0.0
        svc._get_reliability = lambda tool: 0.8

        result = svc.rank_candidates(['weather'])
        assert len(result) == 1
        assert result[0]['name'] == 'weather'
        assert 0 <= result[0]['score'] <= 1

    def test_score_is_between_0_and_1(self, db):
        svc = ToolPerformanceService(get_shared_db_service())

        svc.get_tool_stats = lambda tool, days=30: {'success_rate': 1.0, 'avg_latency': 0, 'avg_cost': 0}
        svc._get_user_preference = lambda tool: 1.0
        svc._get_reliability = lambda tool: 1.0

        result = svc.rank_candidates(['perfect_tool'])
        assert result[0]['score'] <= 1.0
        assert result[0]['score'] >= 0.0


class TestNormalization:
    def test_normalize_latency_fast(self, db):
        svc = ToolPerformanceService(get_shared_db_service())
        assert svc._normalize_latency(0) == 0.0

    def test_normalize_latency_slow(self, db):
        svc = ToolPerformanceService(get_shared_db_service())
        assert svc._normalize_latency(10000) == 1.0  # Capped at 1

    def test_normalize_preference_neutral(self, db):
        svc = ToolPerformanceService(get_shared_db_service())
        assert svc._normalize_preference(0.0) == pytest.approx(0.5)  # 0 -> middle

    def test_normalize_preference_positive(self, db):
        svc = ToolPerformanceService(get_shared_db_service())
        assert svc._normalize_preference(2.0) == pytest.approx(1.0)

    def test_normalize_preference_negative(self, db):
        svc = ToolPerformanceService(get_shared_db_service())
        assert svc._normalize_preference(-2.0) == pytest.approx(0.0)


class TestPreferenceDecay:
    def test_decay_updates_old_preferences(self, db):
        svc = ToolPerformanceService(get_shared_db_service())

        # Seed a preference row with old last_used_at
        db.execute("""
            INSERT INTO user_tool_preferences
                (tool_name, implicit_preference, last_used_at, updated_at)
            VALUES (?, 0.5, datetime('now', '-31 days'), datetime('now'))
        """, ("old_tool",))
        db.commit()

        svc.apply_preference_decay()

        row = db.execute(
            "SELECT implicit_preference FROM user_tool_preferences WHERE tool_name = ?",
            ("old_tool",)
        ).fetchone()
        assert row is not None
        # 0.5 * 0.8 = 0.4
        assert row['implicit_preference'] == pytest.approx(0.4, abs=0.01)
