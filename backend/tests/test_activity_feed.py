"""Tests for unified activity feed — InteractionLogService.get_activity_feed() and _summarize_event()."""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

pytestmark = pytest.mark.unit


# ── _summarize_event ──────────────────────────────────────────────────────────

class TestSummarizeEvent:
    def test_proactive_sent_includes_response_preview(self):
        from services.interaction_log_service import _summarize_event
        result = _summarize_event('proactive_sent', {'response': 'Hello world'})
        assert 'Hello world' in result

    def test_act_loop_telemetry_includes_action_count(self):
        from services.interaction_log_service import _summarize_event
        result = _summarize_event('act_loop_telemetry', {'actions_total': 5, 'termination_reason': 'budget_exhausted'})
        assert '5' in result
        assert 'budget_exhausted' in result

    def test_cron_tool_executed_includes_tool_name(self):
        from services.interaction_log_service import _summarize_event
        result = _summarize_event('cron_tool_executed', {'tool_name': 'web_search'})
        assert 'web_search' in result

    def test_unknown_event_type_returns_humanized_string(self):
        from services.interaction_log_service import _summarize_event
        result = _summarize_event('some_unknown_event', {})
        assert result == 'Some Unknown Event'

    def test_summarize_handles_none_payload(self):
        from services.interaction_log_service import _summarize_event
        result = _summarize_event('spark_nurture_sent', None)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_summarize_handles_empty_payload(self):
        from services.interaction_log_service import _summarize_event
        result = _summarize_event('plan_proposed', {})
        assert isinstance(result, str)

    def test_proactive_sent_truncates_long_response(self):
        from services.interaction_log_service import _summarize_event
        long_response = 'A' * 200
        result = _summarize_event('proactive_sent', {'response': long_response})
        # Preview capped at 80 chars
        assert len(result) < 200


# ── _ACTIVITY_EVENT_TYPES ────────────────────────────────────────────────────

class TestActivityEventTypes:
    def test_contains_expected_types(self):
        from services.interaction_log_service import _ACTIVITY_EVENT_TYPES
        expected = {
            'proactive_sent', 'act_loop_telemetry', 'cron_tool_executed',
            'plan_proposed', 'curiosity_thread_seeded', 'spark_nurture_sent',
        }
        for t in expected:
            assert t in _ACTIVITY_EVENT_TYPES

    def test_excludes_conversation_types(self):
        from services.interaction_log_service import _ACTIVITY_EVENT_TYPES
        for excluded in ('user_input', 'system_response', 'classification'):
            assert excluded not in _ACTIVITY_EVENT_TYPES


# ── InteractionLogService constructor ────────────────────────────────────────

class TestInteractionLogServiceConstructor:
    def test_accepts_database_service_arg(self):
        from services.interaction_log_service import InteractionLogService
        db = MagicMock()
        svc = InteractionLogService(db)
        assert svc.db_service is db

    def test_uses_shared_db_when_no_arg(self):
        from services.interaction_log_service import InteractionLogService
        mock_db = MagicMock()
        with patch('services.database_service.get_shared_db_service', return_value=mock_db):
            svc = InteractionLogService()
        assert svc.db_service is mock_db


# ── get_activity_feed ─────────────────────────────────────────────────────────

class TestGetActivityFeed:
    def _make_service(self, interaction_rows=None, task_rows=None, scheduled_rows=None):
        """Build an InteractionLogService with a mocked DB returning given rows."""
        from services.interaction_log_service import InteractionLogService
        db = MagicMock()
        conn = MagicMock()
        cursor = MagicMock()
        db.connection.return_value.__enter__ = lambda *a: conn
        db.connection.return_value.__exit__ = MagicMock(return_value=False)
        conn.cursor.return_value = cursor

        # fetchall returns different rows for each successive call
        cursor.fetchall.side_effect = [
            interaction_rows or [],
            task_rows or [],
            scheduled_rows or [],
        ]
        return InteractionLogService(db)

    def test_returns_dict_with_items_total_since_hours(self):
        svc = self._make_service()
        result = svc.get_activity_feed(since_hours=24)
        assert 'items' in result
        assert 'total' in result
        assert result['since_hours'] == 24

    def test_empty_sources_returns_empty_items(self):
        svc = self._make_service()
        result = svc.get_activity_feed()
        assert result['items'] == []
        assert result['total'] == 0

    def test_interaction_log_events_included(self):
        now = datetime.now(timezone.utc)
        interaction_rows = [
            ('abc', 'proactive_sent', 'AI', {'response': 'Hey!'}, 'drift', now),
        ]
        svc = self._make_service(interaction_rows=interaction_rows)
        result = svc.get_activity_feed()
        assert len(result['items']) == 1
        assert result['items'][0]['source'] == 'autonomous'
        assert result['items'][0]['type'] == 'proactive_sent'

    def test_persistent_task_events_included(self):
        now = datetime.now(timezone.utc)
        task_rows = [
            (1, 'Research topic', 'in_progress', {'last_summary': 'Found 3 sources', 'coverage_estimate': 0.4, 'cycles_completed': 2}, now),
        ]
        svc = self._make_service(task_rows=task_rows)
        result = svc.get_activity_feed()
        assert len(result['items']) == 1
        assert result['items'][0]['source'] == 'background_task'
        assert result['items'][0]['summary'] == 'Found 3 sources'

    def test_scheduler_events_included(self):
        now = datetime.now(timezone.utc)
        scheduled_rows = [
            ('s1', 'Team standup', 'notification', None, now),
        ]
        svc = self._make_service(scheduled_rows=scheduled_rows)
        result = svc.get_activity_feed()
        assert len(result['items']) == 1
        assert result['items'][0]['source'] == 'scheduler'
        assert result['items'][0]['type'] == 'reminder_fired'

    def test_items_sorted_newest_first(self):
        now = datetime.now(timezone.utc)
        older = now - timedelta(hours=2)
        interaction_rows = [
            ('a', 'proactive_sent', None, {}, 'drift', older),
            ('b', 'plan_proposed', None, {}, 'drift', now),
        ]
        svc = self._make_service(interaction_rows=interaction_rows)
        result = svc.get_activity_feed()
        occurred = [item['occurred_at'] for item in result['items']]
        assert occurred == sorted(occurred, reverse=True)

    def test_pagination_applied(self):
        now = datetime.now(timezone.utc)
        # 5 interaction rows
        rows = [('id', 'proactive_sent', None, {}, 'drift', now - timedelta(minutes=i)) for i in range(5)]
        svc = self._make_service(interaction_rows=rows)
        result = svc.get_activity_feed(limit=3, offset=1)
        assert len(result['items']) == 3
        assert result['total'] == 5

    def test_db_error_returns_empty_feed(self):
        from services.interaction_log_service import InteractionLogService
        db = MagicMock()
        db.connection.side_effect = Exception('DB down')
        svc = InteractionLogService(db)
        result = svc.get_activity_feed()
        assert result['items'] == []
        assert result['total'] == 0


# ── /system/activity endpoint (structural check) ─────────────────────────────

class TestActivityEndpoint:
    def test_endpoint_registered_in_system_blueprint(self):
        """Verify the /system/activity route is registered on system_bp."""
        from api.system import system_bp
        from flask import Flask
        app = Flask(__name__)
        app.register_blueprint(system_bp)
        rules = [str(rule) for rule in app.url_map.iter_rules()]
        assert any('activity' in r for r in rules), (
            "Expected /system/activity route in system blueprint"
        )

    def test_since_hours_clamped_to_valid_range(self):
        """Verify clamping logic in the endpoint source."""
        import inspect
        from api.system import activity_feed
        src = inspect.getsource(activity_feed)
        assert 'max(1, min(since_hours, 168))' in src

    def test_limit_capped_at_200(self):
        """Verify the limit cap in the endpoint source."""
        import inspect
        from api.system import activity_feed
        src = inspect.getsource(activity_feed)
        assert '200' in src
