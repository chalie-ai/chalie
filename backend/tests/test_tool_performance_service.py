"""Unit tests for ToolPerformanceService."""
import pytest
from unittest.mock import MagicMock, patch

pytestmark = pytest.mark.unit


class TestRecordInvocation:
    def test_records_successful_invocation(self):
        from services.tool_performance_service import ToolPerformanceService
        svc = ToolPerformanceService()
        mock_db = MagicMock()
        svc._db = mock_db

        svc.record_invocation("duckduckgo_search", "exch_123", True, 250.0)

        assert mock_db.execute.call_count >= 1
        # Check the INSERT call happened
        first_call = mock_db.execute.call_args_list[0]
        assert 'tool_performance_metrics' in first_call[0][0]

    def test_records_failed_invocation(self):
        from services.tool_performance_service import ToolPerformanceService
        svc = ToolPerformanceService()
        mock_db = MagicMock()
        svc._db = mock_db

        svc.record_invocation("weather", "exch_456", False, 5100.0)
        assert mock_db.execute.call_count >= 1


class TestGetToolStats:
    def test_returns_defaults_for_no_data(self):
        from services.tool_performance_service import ToolPerformanceService
        svc = ToolPerformanceService()
        mock_db = MagicMock()
        mock_db.fetch_all.return_value = [{'total': 0, 'successes': 0, 'avg_latency': None, 'avg_cost': None}]
        svc._db = mock_db

        stats = svc.get_tool_stats("unknown_tool")
        assert 'success_rate' in stats
        assert stats['success_rate'] == 0.5  # default

    def test_computes_correct_success_rate(self):
        from services.tool_performance_service import ToolPerformanceService
        svc = ToolPerformanceService()
        mock_db = MagicMock()
        mock_db.fetch_all.return_value = [{
            'total': 10,
            'successes': 8,
            'avg_latency': 300.0,
            'avg_cost': 0.0,
            'total_cost': 0.0,
        }]
        svc._db = mock_db

        stats = svc.get_tool_stats("good_tool")
        assert stats['success_rate'] == pytest.approx(0.8)
        assert stats['avg_latency'] == pytest.approx(300.0)


class TestRankCandidates:
    def test_returns_sorted_by_score(self):
        from services.tool_performance_service import ToolPerformanceService
        svc = ToolPerformanceService()

        # Mock get_tool_stats, _get_user_preference, _get_reliability
        def mock_stats(tool_name, days=30):
            return {
                'duckduckgo_search': {'success_rate': 0.9, 'avg_latency': 200, 'avg_cost': 0},
                'weather': {'success_rate': 0.5, 'avg_latency': 800, 'avg_cost': 0},
            }.get(tool_name, {'success_rate': 0.5, 'avg_latency': 0, 'avg_cost': 0})

        svc.get_tool_stats = mock_stats
        svc._get_user_preference = lambda tool, user: 0.0
        svc._get_reliability = lambda tool: 1.0

        result = svc.rank_candidates(['weather', 'duckduckgo_search'])
        assert len(result) == 2
        # duckduckgo_search should rank higher (better success rate, lower latency)
        assert result[0]['name'] == 'duckduckgo_search'

    def test_empty_candidates_returns_empty(self):
        from services.tool_performance_service import ToolPerformanceService
        svc = ToolPerformanceService()
        assert svc.rank_candidates([]) == []

    def test_single_candidate_returned(self):
        from services.tool_performance_service import ToolPerformanceService
        svc = ToolPerformanceService()

        svc.get_tool_stats = lambda tool, days=30: {'success_rate': 0.7, 'avg_latency': 400, 'avg_cost': 0}
        svc._get_user_preference = lambda tool, user: 0.0
        svc._get_reliability = lambda tool: 0.8

        result = svc.rank_candidates(['weather'])
        assert len(result) == 1
        assert result[0]['name'] == 'weather'
        assert 0 <= result[0]['score'] <= 1

    def test_score_is_between_0_and_1(self):
        from services.tool_performance_service import ToolPerformanceService
        svc = ToolPerformanceService()

        svc.get_tool_stats = lambda tool, days=30: {'success_rate': 1.0, 'avg_latency': 0, 'avg_cost': 0}
        svc._get_user_preference = lambda tool, user: 1.0
        svc._get_reliability = lambda tool: 1.0

        result = svc.rank_candidates(['perfect_tool'])
        assert result[0]['score'] <= 1.0
        assert result[0]['score'] >= 0.0


class TestNormalization:
    def test_normalize_latency_fast(self):
        from services.tool_performance_service import ToolPerformanceService
        svc = ToolPerformanceService()
        assert svc._normalize_latency(0) == 0.0

    def test_normalize_latency_slow(self):
        from services.tool_performance_service import ToolPerformanceService
        svc = ToolPerformanceService()
        assert svc._normalize_latency(10000) == 1.0  # Capped at 1

    def test_normalize_preference_neutral(self):
        from services.tool_performance_service import ToolPerformanceService
        svc = ToolPerformanceService()
        assert svc._normalize_preference(0.0) == pytest.approx(0.5)  # 0 → middle

    def test_normalize_preference_positive(self):
        from services.tool_performance_service import ToolPerformanceService
        svc = ToolPerformanceService()
        assert svc._normalize_preference(2.0) == pytest.approx(1.0)

    def test_normalize_preference_negative(self):
        from services.tool_performance_service import ToolPerformanceService
        svc = ToolPerformanceService()
        assert svc._normalize_preference(-2.0) == pytest.approx(0.0)


class TestPreferenceDecay:
    def test_decay_updates_old_preferences(self):
        from services.tool_performance_service import ToolPerformanceService
        svc = ToolPerformanceService()
        mock_db = MagicMock()
        svc._db = mock_db

        svc.apply_preference_decay('default')
        mock_db.execute.assert_called_once()
        call_sql = mock_db.execute.call_args[0][0]
        assert 'implicit_preference' in call_sql
        assert '30 days' in call_sql
