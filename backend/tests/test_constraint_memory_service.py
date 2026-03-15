"""Tests for ConstraintMemoryService — constraint/gate rejection query layer."""

import json
import pytest
from unittest.mock import patch, MagicMock

from services.constraint_memory_service import ConstraintMemoryService, ALL_REJECTION_TYPES


def _make_db_mock():
    """Create a mock DatabaseService with context-managed connection."""
    db = MagicMock()
    conn = MagicMock()
    db.connection.return_value.__enter__ = MagicMock(return_value=conn)
    db.connection.return_value.__exit__ = MagicMock(return_value=False)
    return db, conn


def _make_store_mock():
    """MemoryStore mock that starts empty (no cache)."""
    store = MagicMock()
    store.get.return_value = None
    return store


def _make_rejection_event(event_type, payload, topic=None):
    """Helper to build a rejection event dict."""
    return {
        'id': 'test-id',
        'event_type': event_type,
        'topic': topic,
        'payload': payload,
        'source': 'test',
        'created_at': '2026-03-10T12:00:00+00:00',
    }


# ── Summary computation ──────────────────────────────────────────────


@pytest.mark.unit
class TestConstraintSummary:

    def test_empty_when_no_events(self):
        db, _ = _make_db_mock()
        store = _make_store_mock()

        with patch('services.memory_client.MemoryClientService.create_connection', return_value=store), \
             patch('services.interaction_log_service.InteractionLogService.get_events_by_types', return_value=[]):
            service = ConstraintMemoryService(db)
            summary = service.get_constraint_summary()

        assert summary['total_rejections'] == 0
        assert summary['rejection_counts'] == {}
        assert summary['top_reasons'] == []
        assert summary['blocked_actions'] == []

    def test_counts_by_event_type(self):
        db, _ = _make_db_mock()
        store = _make_store_mock()

        events = [
            _make_rejection_event('action_gate_rejected', {
                'rejections': [{'action': 'COMMUNICATE', 'reason': 'quiet_hours'}],
            }),
            _make_rejection_event('action_gate_rejected', {
                'rejections': [{'action': 'COMMUNICATE', 'reason': 'quiet_hours'}],
            }),
            _make_rejection_event('plan_rejected', {
                'rejection_type': 'dag_invalid',
            }),
        ]

        with patch('services.memory_client.MemoryClientService.create_connection', return_value=store), \
             patch('services.interaction_log_service.InteractionLogService.get_events_by_types', return_value=events):
            service = ConstraintMemoryService(db)
            summary = service.get_constraint_summary()

        assert summary['total_rejections'] == 3
        assert summary['rejection_counts']['action_gate_rejected'] == 2
        assert summary['rejection_counts']['plan_rejected'] == 1
        assert 'COMMUNICATE' in summary['blocked_actions']
        assert 'quiet_hours' in summary['top_reasons']

    def test_summary_cached_in_memorystore(self):
        db, _ = _make_db_mock()
        store = _make_store_mock()

        events = [
            _make_rejection_event('triage_override', {'rule': 'act_no_tools_available'}),
        ]

        with patch('services.memory_client.MemoryClientService.create_connection', return_value=store), \
             patch('services.interaction_log_service.InteractionLogService.get_events_by_types', return_value=events):
            service = ConstraintMemoryService(db)
            service.get_constraint_summary()

        # Verify cache write
        store.setex.assert_called_once()
        cache_key = store.setex.call_args[0][0]
        assert cache_key == 'constraint_memory:summary'

    def test_summary_returns_cached_value(self):
        db, _ = _make_db_mock()
        store = _make_store_mock()

        cached_summary = json.dumps({
            'rejection_counts': {'plan_rejected': 5},
            'top_reasons': ['dag_invalid'],
            'blocked_actions': [],
            'total_rejections': 5,
        })
        store.get.return_value = cached_summary

        with patch('services.memory_client.MemoryClientService.create_connection', return_value=store):
            service = ConstraintMemoryService(db)
            summary = service.get_constraint_summary()

        assert summary['total_rejections'] == 5
        assert summary['rejection_counts']['plan_rejected'] == 5


# ── Prompt formatting ────────────────────────────────────────────────


@pytest.mark.unit
class TestFormatForPrompt:

    def test_empty_when_no_rejections(self):
        db, _ = _make_db_mock()
        store = _make_store_mock()

        with patch('services.memory_client.MemoryClientService.create_connection', return_value=store), \
             patch('services.interaction_log_service.InteractionLogService.get_events_by_types', return_value=[]):
            service = ConstraintMemoryService(db)
            result = service.format_for_prompt(mode='act')

        assert result == ''

    def test_act_mode_includes_blocked_actions(self):
        db, _ = _make_db_mock()
        store = _make_store_mock()

        events = [
            _make_rejection_event('action_gate_rejected', {
                'rejections': [
                    {'action': 'PLAN', 'reason': 'cooldown'},
                    {'action': 'COMMUNICATE', 'reason': 'quiet_hours'},
                ],
            }),
        ]

        with patch('services.memory_client.MemoryClientService.create_connection', return_value=store), \
             patch('services.interaction_log_service.InteractionLogService.get_events_by_types', return_value=events):
            service = ConstraintMemoryService(db)
            result = service.format_for_prompt(mode='act')

        assert 'PLAN' in result or 'COMMUNICATE' in result

    def test_plan_mode_surfaces_plan_rejections(self):
        db, _ = _make_db_mock()
        store = _make_store_mock()

        events = [
            _make_rejection_event('plan_rejected', {'rejection_type': 'dag_invalid'}),
            _make_rejection_event('plan_rejected', {'rejection_type': 'step_quality'}),
        ]

        with patch('services.memory_client.MemoryClientService.create_connection', return_value=store), \
             patch('services.interaction_log_service.InteractionLogService.get_events_by_types', return_value=events):
            service = ConstraintMemoryService(db)
            result = service.format_for_prompt(mode='plan')

        assert 'plan rejection' in result.lower() or '2x' in result

    def test_respond_mode_very_light(self):
        db, _ = _make_db_mock()
        store = _make_store_mock()

        # Only action_gate_rejected — RESPOND mode should produce nothing
        events = [
            _make_rejection_event('action_gate_rejected', {
                'rejections': [{'action': 'PLAN', 'reason': 'cooldown'}],
            }),
        ]

        with patch('services.memory_client.MemoryClientService.create_connection', return_value=store), \
             patch('services.interaction_log_service.InteractionLogService.get_events_by_types', return_value=events):
            service = ConstraintMemoryService(db)
            result = service.format_for_prompt(mode='respond')

        # RESPOND only surfaces triage overrides
        assert result == ''

    def test_token_budget_truncation(self):
        db, _ = _make_db_mock()
        store = _make_store_mock()

        # Generate lots of events to exceed budget
        events = [
            _make_rejection_event('action_gate_rejected', {
                'rejections': [{'action': f'ACTION_{i}', 'reason': f'reason_{i}'}],
            })
            for i in range(100)
        ]

        with patch('services.memory_client.MemoryClientService.create_connection', return_value=store), \
             patch('services.interaction_log_service.InteractionLogService.get_events_by_types', return_value=events):
            service = ConstraintMemoryService(db)
            result = service.format_for_prompt(mode='act', max_tokens=50)

        # Should be truncated
        assert len(result) <= 50 * 4 + 20  # budget + truncation marker


# ── Blocked action patterns ──────────────────────────────────────────


@pytest.mark.unit
class TestBlockedActionPatterns:

    def test_filters_below_threshold(self):
        db, _ = _make_db_mock()
        store = _make_store_mock()

        # Only 2 rejections for PLAN — below threshold of 3
        events = [
            _make_rejection_event('action_gate_rejected', {
                'rejections': [{'action': 'PLAN', 'reason': 'cooldown'}],
            }),
            _make_rejection_event('action_gate_rejected', {
                'rejections': [{'action': 'PLAN', 'reason': 'cooldown'}],
            }),
        ]

        with patch('services.memory_client.MemoryClientService.create_connection', return_value=store), \
             patch('services.interaction_log_service.InteractionLogService.get_events_by_types', return_value=events):
            service = ConstraintMemoryService(db)
            patterns = service.get_blocked_action_patterns()

        assert len(patterns) == 0

    def test_returns_patterns_above_threshold(self):
        db, _ = _make_db_mock()
        store = _make_store_mock()

        events = [
            _make_rejection_event('action_gate_rejected', {
                'rejections': [{'action': 'COMMUNICATE', 'reason': 'quiet_hours'}],
            })
            for _ in range(5)
        ]

        with patch('services.memory_client.MemoryClientService.create_connection', return_value=store), \
             patch('services.interaction_log_service.InteractionLogService.get_events_by_types', return_value=events):
            service = ConstraintMemoryService(db)
            patterns = service.get_blocked_action_patterns()

        assert len(patterns) == 1
        assert patterns[0]['action'] == 'COMMUNICATE'
        assert patterns[0]['total_rejections'] == 5
        assert patterns[0]['top_reason'] == 'quiet_hours'

    def test_sorted_by_frequency_descending(self):
        db, _ = _make_db_mock()
        store = _make_store_mock()

        events = (
            [_make_rejection_event('action_gate_rejected', {
                'rejections': [{'action': 'PLAN', 'reason': 'cooldown'}],
            }) for _ in range(3)]
            +
            [_make_rejection_event('action_gate_rejected', {
                'rejections': [{'action': 'COMMUNICATE', 'reason': 'quiet_hours'}],
            }) for _ in range(7)]
        )

        with patch('services.memory_client.MemoryClientService.create_connection', return_value=store), \
             patch('services.interaction_log_service.InteractionLogService.get_events_by_types', return_value=events):
            service = ConstraintMemoryService(db)
            patterns = service.get_blocked_action_patterns()

        assert len(patterns) == 2
        assert patterns[0]['action'] == 'COMMUNICATE'
        assert patterns[1]['action'] == 'PLAN'


# ── ALL_REJECTION_TYPES constant ─────────────────────────────────────


@pytest.mark.unit
class TestRejectionTypes:

    def test_all_six_types_present(self):
        assert len(ALL_REJECTION_TYPES) == 6
        assert 'action_gate_rejected' in ALL_REJECTION_TYPES
        assert 'plan_rejected' in ALL_REJECTION_TYPES
        assert 'assimilation_rejected' in ALL_REJECTION_TYPES
        assert 'triage_override' in ALL_REJECTION_TYPES
        assert 'routing_anti_oscillation' in ALL_REJECTION_TYPES
        assert 'reliability_warning' in ALL_REJECTION_TYPES
