"""
Tests for introspect_skill — covering the Plan 07 additions:
- _get_recent_decision_explanations(): pulls routing decision data with scores/signals
- _get_recent_autonomous_actions(): filters interaction_log to user-relevant event types

Both functions import their services inside the function body (lazy imports), so
patches target the source modules rather than the using module.
"""

import pytest
from unittest.mock import MagicMock, patch, call
from contextlib import contextmanager


def _make_decision(**overrides):
    """Build a mock routing decision dict with sensible defaults."""
    base = {
        'selected_mode': 'RESPOND',
        'router_confidence': 0.82,
        'scores': {'RESPOND': 0.82, 'ACT': 0.54, 'CLARIFY': 0.30, 'ACKNOWLEDGE': 0.15},
        'tiebreaker_used': False,
        'tiebreaker_candidates': None,
        'margin': 0.28,
        'signal_snapshot': {
            'context_warmth': 0.65,
            'has_question_mark': False,
            'greeting_pattern': False,
            'explicit_feedback': None,
            'memory_confidence': 0.70,
            'is_new_topic': False,
        },
    }
    base.update(overrides)
    return base


@pytest.mark.unit
class TestGetRecentDecisionExplanations:

    def _mock_service(self, decisions):
        mock_svc = MagicMock()
        mock_svc.get_recent_decisions.return_value = decisions
        return mock_svc

    def test_returns_list_of_explanations(self):
        from services.innate_skills.introspect_skill import _get_recent_decision_explanations
        mock_svc = self._mock_service([_make_decision()])
        mock_db = MagicMock()

        with patch('services.routing_decision_service.RoutingDecisionService', return_value=mock_svc), \
             patch('services.database_service.get_shared_db_service', return_value=mock_db):
            result = _get_recent_decision_explanations(limit=3)

        assert isinstance(result, list)
        assert len(result) == 1

    def test_explanation_contains_required_fields(self):
        from services.innate_skills.introspect_skill import _get_recent_decision_explanations
        mock_svc = self._mock_service([_make_decision()])
        mock_db = MagicMock()

        with patch('services.routing_decision_service.RoutingDecisionService', return_value=mock_svc), \
             patch('services.database_service.get_shared_db_service', return_value=mock_db):
            result = _get_recent_decision_explanations()

        exp = result[0]
        assert 'mode' in exp
        assert 'confidence' in exp
        assert 'scores' in exp
        assert 'tiebreaker_used' in exp
        assert 'key_signals' in exp
        assert 'margin' in exp

    def test_mode_mapped_correctly(self):
        from services.innate_skills.introspect_skill import _get_recent_decision_explanations
        mock_svc = self._mock_service([_make_decision(selected_mode='ACT')])
        mock_db = MagicMock()

        with patch('services.routing_decision_service.RoutingDecisionService', return_value=mock_svc), \
             patch('services.database_service.get_shared_db_service', return_value=mock_db):
            result = _get_recent_decision_explanations()

        assert result[0]['mode'] == 'ACT'

    def test_confidence_rounded_to_3_places(self):
        from services.innate_skills.introspect_skill import _get_recent_decision_explanations
        mock_svc = self._mock_service([_make_decision(router_confidence=0.8215678)])
        mock_db = MagicMock()

        with patch('services.routing_decision_service.RoutingDecisionService', return_value=mock_svc), \
             patch('services.database_service.get_shared_db_service', return_value=mock_db):
            result = _get_recent_decision_explanations()

        # Should be rounded to 3 decimal places
        assert result[0]['confidence'] == round(0.8215678, 3)

    def test_key_signals_context_warmth_extracted(self):
        from services.innate_skills.introspect_skill import _get_recent_decision_explanations
        decision = _make_decision(signal_snapshot={'context_warmth': 0.753})
        mock_svc = self._mock_service([decision])
        mock_db = MagicMock()

        with patch('services.routing_decision_service.RoutingDecisionService', return_value=mock_svc), \
             patch('services.database_service.get_shared_db_service', return_value=mock_db):
            result = _get_recent_decision_explanations()

        assert result[0]['key_signals']['context_warmth'] == round(0.753, 2)

    def test_key_signals_question_detected(self):
        from services.innate_skills.introspect_skill import _get_recent_decision_explanations
        decision = _make_decision(signal_snapshot={'has_question_mark': True, 'context_warmth': 0.5})
        mock_svc = self._mock_service([decision])
        mock_db = MagicMock()

        with patch('services.routing_decision_service.RoutingDecisionService', return_value=mock_svc), \
             patch('services.database_service.get_shared_db_service', return_value=mock_db):
            result = _get_recent_decision_explanations()

        assert result[0]['key_signals'].get('question_detected') is True

    def test_key_signals_false_question_not_included(self):
        """When has_question_mark is False, question_detected should be absent (not False)."""
        from services.innate_skills.introspect_skill import _get_recent_decision_explanations
        decision = _make_decision(signal_snapshot={'has_question_mark': False})
        mock_svc = self._mock_service([decision])
        mock_db = MagicMock()

        with patch('services.routing_decision_service.RoutingDecisionService', return_value=mock_svc), \
             patch('services.database_service.get_shared_db_service', return_value=mock_db):
            result = _get_recent_decision_explanations()

        assert 'question_detected' not in result[0]['key_signals']

    def test_tiebreaker_info_included_when_used(self):
        from services.innate_skills.introspect_skill import _get_recent_decision_explanations
        decision = _make_decision(
            tiebreaker_used=True,
            tiebreaker_candidates=['RESPOND', 'ACT'],
            signal_snapshot={},
        )
        mock_svc = self._mock_service([decision])
        mock_db = MagicMock()

        with patch('services.routing_decision_service.RoutingDecisionService', return_value=mock_svc), \
             patch('services.database_service.get_shared_db_service', return_value=mock_db):
            result = _get_recent_decision_explanations()

        assert result[0]['tiebreaker_used'] is True
        assert result[0]['tiebreaker_candidates'] == ['RESPOND', 'ACT']

    def test_empty_decisions_returns_empty_list(self):
        from services.innate_skills.introspect_skill import _get_recent_decision_explanations
        mock_svc = self._mock_service([])
        mock_db = MagicMock()

        with patch('services.routing_decision_service.RoutingDecisionService', return_value=mock_svc), \
             patch('services.database_service.get_shared_db_service', return_value=mock_db):
            result = _get_recent_decision_explanations()

        assert result == []

    def test_error_returns_empty_list(self):
        from services.innate_skills.introspect_skill import _get_recent_decision_explanations
        with patch('services.database_service.get_shared_db_service', side_effect=Exception("db down")):
            result = _get_recent_decision_explanations()

        assert result == []

    def test_scores_rounded(self):
        from services.innate_skills.introspect_skill import _get_recent_decision_explanations
        decision = _make_decision(scores={'RESPOND': 0.8212345, 'ACT': 0.3456789})
        mock_svc = self._mock_service([decision])
        mock_db = MagicMock()

        with patch('services.routing_decision_service.RoutingDecisionService', return_value=mock_svc), \
             patch('services.database_service.get_shared_db_service', return_value=mock_db):
            result = _get_recent_decision_explanations()

        scores = result[0]['scores']
        assert scores['RESPOND'] == round(0.8212345, 3)
        assert scores['ACT'] == round(0.3456789, 3)

    def test_none_scores_becomes_empty_dict(self):
        from services.innate_skills.introspect_skill import _get_recent_decision_explanations
        decision = _make_decision(scores=None, signal_snapshot={})
        mock_svc = self._mock_service([decision])
        mock_db = MagicMock()

        with patch('services.routing_decision_service.RoutingDecisionService', return_value=mock_svc), \
             patch('services.database_service.get_shared_db_service', return_value=mock_db):
            result = _get_recent_decision_explanations()

        assert result[0]['scores'] == {}

    def test_multiple_decisions_all_returned(self):
        from services.innate_skills.introspect_skill import _get_recent_decision_explanations
        decisions = [
            _make_decision(selected_mode='RESPOND'),
            _make_decision(selected_mode='ACT'),
            _make_decision(selected_mode='CLARIFY'),
        ]
        mock_svc = self._mock_service(decisions)
        mock_db = MagicMock()

        with patch('services.routing_decision_service.RoutingDecisionService', return_value=mock_svc), \
             patch('services.database_service.get_shared_db_service', return_value=mock_db):
            result = _get_recent_decision_explanations(limit=3)

        assert len(result) == 3
        assert [r['mode'] for r in result] == ['RESPOND', 'ACT', 'CLARIFY']


@pytest.mark.unit
class TestGetRecentAutonomousActions:

    def _make_cursor(self, rows):
        """Build a mock cursor that returns given rows."""
        cursor = MagicMock()
        cursor.fetchall.return_value = rows
        return cursor

    def _make_db_service(self, rows):
        """Build a mock db_service whose connection() yields a mock conn."""
        cursor = self._make_cursor(rows)
        conn = MagicMock()
        conn.cursor.return_value = cursor

        db_service = MagicMock()
        db_service.connection.return_value.__enter__ = MagicMock(return_value=conn)
        db_service.connection.return_value.__exit__ = MagicMock(return_value=False)
        return db_service, conn, cursor

    def test_returns_list(self):
        from services.innate_skills.introspect_skill import _get_recent_autonomous_actions
        db_service, _, _ = self._make_db_service([])

        with patch('services.database_service.get_shared_db_service', return_value=db_service):
            result = _get_recent_autonomous_actions()

        assert isinstance(result, list)

    def test_formats_rows_correctly(self):
        from services.innate_skills.introspect_skill import _get_recent_autonomous_actions
        rows = [
            ('proactive_sent', {'message': 'hello'}, '2024-01-01 10:00:00'),
        ]
        db_service, _, _ = self._make_db_service(rows)

        with patch('services.database_service.get_shared_db_service', return_value=db_service):
            result = _get_recent_autonomous_actions()

        assert len(result) == 1
        assert result[0]['event_type'] == 'proactive_sent'
        assert 'payload_summary' in result[0]
        assert 'created_at' in result[0]

    def test_payload_summary_truncated_to_200(self):
        from services.innate_skills.introspect_skill import _get_recent_autonomous_actions
        long_payload = {'data': 'x' * 500}
        rows = [('cron_tool_executed', long_payload, '2024-01-01 10:00:00')]
        db_service, _, _ = self._make_db_service(rows)

        with patch('services.database_service.get_shared_db_service', return_value=db_service):
            result = _get_recent_autonomous_actions()

        assert len(result[0]['payload_summary']) <= 200

    def test_non_dict_payload_handled(self):
        """Non-dict payload (e.g. string or None from db) should not crash."""
        from services.innate_skills.introspect_skill import _get_recent_autonomous_actions
        rows = [('plan_proposed', None, '2024-01-01 10:00:00')]
        db_service, _, _ = self._make_db_service(rows)

        with patch('services.database_service.get_shared_db_service', return_value=db_service):
            result = _get_recent_autonomous_actions()

        # Should not raise, and payload_summary should be the str({}) representation
        assert result[0]['payload_summary'] == '{}'

    def test_empty_rows_returns_empty_list(self):
        from services.innate_skills.introspect_skill import _get_recent_autonomous_actions
        db_service, _, _ = self._make_db_service([])

        with patch('services.database_service.get_shared_db_service', return_value=db_service):
            result = _get_recent_autonomous_actions()

        assert result == []

    def test_error_returns_empty_list(self):
        from services.innate_skills.introspect_skill import _get_recent_autonomous_actions
        with patch('services.database_service.get_shared_db_service', side_effect=Exception("db down")):
            result = _get_recent_autonomous_actions()

        assert result == []

    def test_query_filters_only_relevant_types(self):
        """The SQL query should only request proactive_sent, cron_tool_executed, plan_proposed."""
        from services.innate_skills.introspect_skill import _get_recent_autonomous_actions
        db_service, conn, cursor = self._make_db_service([])

        with patch('services.database_service.get_shared_db_service', return_value=db_service):
            _get_recent_autonomous_actions()

        # Verify the execute call was made and contained the right event types
        assert cursor.execute.called
        sql_call = cursor.execute.call_args
        sql, params = sql_call[0]

        assert 'proactive_sent' in params
        assert 'cron_tool_executed' in params
        assert 'plan_proposed' in params

    def test_multiple_rows_returned(self):
        from services.innate_skills.introspect_skill import _get_recent_autonomous_actions
        rows = [
            ('proactive_sent', {'msg': 'a'}, '2024-01-02 09:00:00'),
            ('cron_tool_executed', {'tool': 'search'}, '2024-01-01 20:00:00'),
            ('plan_proposed', {'goal': 'organise'}, '2024-01-01 15:00:00'),
        ]
        db_service, _, _ = self._make_db_service(rows)

        with patch('services.database_service.get_shared_db_service', return_value=db_service):
            result = _get_recent_autonomous_actions(limit=5)

        assert len(result) == 3
        assert result[0]['event_type'] == 'proactive_sent'
        assert result[1]['event_type'] == 'cron_tool_executed'


@pytest.mark.unit
class TestFormatStateWithNewFields:
    """Verify _format_state() renders decision_explanations and recent_autonomous_actions."""

    def _build_state(self):
        """Minimal state dict with all required keys."""
        return {
            'context_warmth': 0.5,
            'gist_count': 2,
            'fact_count': 3,
            'working_memory_depth': 4,
            'topic_age': '5min',
            'partial_match_signal': 1,
            'recall_failure_rate': 0.1,
            'focus_active': False,
            'communication_style': {},
            'recent_modes': ['RESPOND'],
            'skill_stats': {},
            'world_state': '',
            'decision_explanations': [],
            'recent_autonomous_actions': [],
        }

    def test_empty_decision_explanations_renders_none_message(self):
        from services.innate_skills.introspect_skill import _format_state
        state = self._build_state()
        output = _format_state(state, 'test_topic')
        assert 'none in last hour' in output

    def test_decision_explanation_renders_mode(self):
        from services.innate_skills.introspect_skill import _format_state
        state = self._build_state()
        state['decision_explanations'] = [{
            'mode': 'RESPOND',
            'confidence': 0.82,
            'margin': 0.28,
            'scores': {'RESPOND': 0.82},
            'key_signals': {'context_warmth': 0.65},
            'tiebreaker_used': False,
            'tiebreaker_candidates': None,
        }]
        output = _format_state(state, 'test_topic')
        assert 'mode=RESPOND' in output
        assert 'confidence=0.82' in output

    def test_tiebreaker_info_rendered_when_used(self):
        from services.innate_skills.introspect_skill import _format_state
        state = self._build_state()
        state['decision_explanations'] = [{
            'mode': 'ACT',
            'confidence': 0.45,
            'margin': 0.02,
            'scores': {},
            'key_signals': {},
            'tiebreaker_used': True,
            'tiebreaker_candidates': ['ACT', 'RESPOND'],
        }]
        output = _format_state(state, 'test_topic')
        assert 'tiebreaker between' in output

    def test_empty_autonomous_actions_renders_none_message(self):
        from services.innate_skills.introspect_skill import _format_state
        state = self._build_state()
        output = _format_state(state, 'test_topic')
        assert 'none recently' in output

    def test_autonomous_action_rendered(self):
        from services.innate_skills.introspect_skill import _format_state
        state = self._build_state()
        state['recent_autonomous_actions'] = [{
            'event_type': 'proactive_sent',
            'payload_summary': '{"message": "hello"}',
            'created_at': '2024-01-01 10:00:00',
        }]
        output = _format_state(state, 'test_topic')
        assert 'proactive_sent' in output
        assert '2024-01-01 10:00:00' in output
