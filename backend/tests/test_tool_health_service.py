"""Unit tests for ToolHealthService — ephemeral tool effectiveness tracking."""

import pytest
import json
from unittest.mock import patch, MagicMock


@pytest.fixture
def mock_store():
    """In-memory dict simulating MemoryStore with TTL support."""
    store_data = {}

    class FakeStore:
        def get(self, key):
            return store_data.get(key)

        def setex(self, key, ttl, value):
            store_data[key] = value

        def keys(self, pattern):
            import fnmatch
            return [k for k in store_data if fnmatch.fnmatch(k, pattern)]

    return FakeStore(), store_data


@pytest.fixture(autouse=True)
def patch_memory_store(mock_store):
    store, _ = mock_store
    with patch(
        'services.tool_health_service._get_store',
        return_value=store,
    ):
        yield


@pytest.mark.unit
class TestGetPotential:
    def test_default_potential_is_1(self):
        from services.tool_health_service import get_potential
        assert get_potential('mealdb') == 1.0

    def test_reads_stored_potential(self, mock_store):
        store, data = mock_store
        data['tool_health:mealdb'] = json.dumps({'potential': 0.5})
        from services.tool_health_service import get_potential
        assert get_potential('mealdb') == 0.5


@pytest.mark.unit
class TestRecordOutcome:
    def test_success_boosts_potential(self, mock_store):
        store, data = mock_store
        data['tool_health:web_search'] = json.dumps({
            'potential': 0.5, 'failures': 2, 'successes': 0,
        })
        from services.tool_health_service import record_outcome
        new = record_outcome('web_search', 'success')
        assert new == 0.65  # 0.5 + 0.15

    def test_success_capped_at_1(self):
        from services.tool_health_service import record_outcome
        new = record_outcome('fresh_tool', 'success')
        assert new == 1.0

    def test_empty_decays_by_half(self):
        from services.tool_health_service import record_outcome
        new = record_outcome('mealdb', 'empty')
        assert new == 0.5  # 1.0 * 0.5

    def test_critic_correction_harsh_decay(self, mock_store):
        store, data = mock_store
        data['tool_health:mealdb'] = json.dumps({
            'potential': 1.0, 'failures': 0, 'successes': 0,
        })
        from services.tool_health_service import record_outcome
        new = record_outcome('mealdb', 'critic_correction')
        assert new == 0.4  # 1.0 * 0.4

    def test_error_severe_decay(self):
        from services.tool_health_service import record_outcome
        new = record_outcome('ddg', 'error')
        assert new == 0.3  # 1.0 * 0.3

    def test_floor_prevents_zero(self, mock_store):
        store, data = mock_store
        data['tool_health:broken'] = json.dumps({
            'potential': 0.06, 'failures': 5, 'successes': 0,
        })
        from services.tool_health_service import record_outcome
        new = record_outcome('broken', 'error')
        assert new == 0.05  # floor

    def test_repeated_failures_compound(self):
        from services.tool_health_service import record_outcome
        p1 = record_outcome('ddg', 'empty')
        assert p1 == 0.5
        p2 = record_outcome('ddg', 'empty')
        assert p2 == 0.25
        p3 = record_outcome('ddg', 'empty')
        assert p3 == 0.125

    def test_recovery_after_failures(self):
        from services.tool_health_service import record_outcome
        record_outcome('ddg', 'empty')    # 0.5
        record_outcome('ddg', 'empty')    # 0.25
        p = record_outcome('ddg', 'success')  # 0.25 + 0.15 = 0.40
        assert abs(p - 0.40) < 0.01


@pytest.mark.unit
class TestClassifyResult:
    def test_error_status(self):
        from services.tool_health_service import classify_result
        assert classify_result({'status': 'error'}) == 'error'

    def test_timeout_status(self):
        from services.tool_health_service import classify_result
        assert classify_result({'status': 'timeout'}) == 'timeout'

    def test_critic_correction(self):
        from services.tool_health_service import classify_result
        assert classify_result({'status': 'critic_correction'}) == 'critic_correction'

    def test_success_with_results(self):
        from services.tool_health_service import classify_result
        r = classify_result({
            'status': 'success',
            'result': {'count': 5, 'results': [1, 2, 3]},
        })
        assert r == 'success'

    def test_success_with_empty_count(self):
        from services.tool_health_service import classify_result
        r = classify_result({
            'status': 'success',
            'result': {'count': 0, 'results': []},
        })
        assert r == 'empty'

    def test_success_with_no_results_text(self):
        from services.tool_health_service import classify_result
        r = classify_result({
            'status': 'success',
            'result': 'No results found for "French toast"',
        })
        assert r == 'empty'

    def test_success_with_content(self):
        from services.tool_health_service import classify_result
        r = classify_result({
            'status': 'success',
            'result': 'Here are 3 recipes for chicken curry...',
        })
        assert r == 'success'


@pytest.mark.unit
class TestFormatHealthHint:
    def test_no_hint_when_healthy(self):
        from services.tool_health_service import format_health_hint
        assert format_health_hint({'mealdb': 1.0, 'ddg': 0.9}) == ''

    def test_hint_for_degraded_tool(self):
        from services.tool_health_service import format_health_hint
        hint = format_health_hint({'mealdb': 0.6})
        assert 'mealdb' in hint
        assert 'slightly degraded' in hint

    def test_hint_for_very_low_tool(self):
        from services.tool_health_service import format_health_hint
        hint = format_health_hint({'ddg': 0.1})
        assert 'very low effectiveness' in hint
        assert 'own knowledge' in hint

    def test_hint_for_medium_degraded(self):
        from services.tool_health_service import format_health_hint
        hint = format_health_hint({'mealdb': 0.35})
        assert 'reduced effectiveness' in hint


@pytest.mark.unit
class TestGetAllHealth:
    def test_returns_all_tracked_tools(self, mock_store):
        store, data = mock_store
        data['tool_health:mealdb'] = json.dumps({'potential': 0.5, 'failures': 2})
        data['tool_health:ddg'] = json.dumps({'potential': 0.3, 'failures': 3})
        from services.tool_health_service import get_all_health
        health = get_all_health()
        assert 'mealdb' in health
        assert 'ddg' in health
        assert health['mealdb']['potential'] == 0.5

    def test_empty_when_no_data(self):
        from services.tool_health_service import get_all_health
        assert get_all_health() == {}
