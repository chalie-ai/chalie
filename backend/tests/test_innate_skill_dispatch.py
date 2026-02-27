"""Tests for innate skill direct dispatch — _is_innate_skill_only, _handle_innate_skill_dispatch, scheduler dedup."""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from workers.digest_worker import _is_innate_skill_only, _CONTEXTUAL_SKILLS, _PRIMITIVES


pytestmark = pytest.mark.unit


# ── Fake TriageResult for testing ────────────────────────────────────

@dataclass
class FakeTriageResult:
    branch: str = 'act'
    mode: str = 'ACT'
    tools: list = None
    skills: list = None
    confidence_internal: float = 0.3
    confidence_tool_need: float = 0.8
    freshness_risk: float = 0.2
    decision_entropy: float = 0.0
    reasoning: str = 'test'
    triage_time_ms: float = 1.0
    fast_filtered: bool = False
    self_eval_override: bool = False
    self_eval_reason: str = ''

    def __post_init__(self):
        if self.tools is None:
            self.tools = []
        if self.skills is None:
            self.skills = []


# ── _is_innate_skill_only ────────────────────────────────────────────

class TestIsInnateSkillOnly:

    def test_true_with_schedule(self):
        """skills=['recall','schedule'], tools=[] → True"""
        tr = FakeTriageResult(tools=[], skills=['recall', 'schedule'])
        assert _is_innate_skill_only(tr) is True

    def test_true_with_list(self):
        """skills=['memorize','list'], tools=[] → True"""
        tr = FakeTriageResult(tools=[], skills=['memorize', 'list'])
        assert _is_innate_skill_only(tr) is True

    def test_true_with_focus(self):
        tr = FakeTriageResult(tools=[], skills=['recall', 'focus'])
        assert _is_innate_skill_only(tr) is True

    def test_true_with_persistent_task(self):
        tr = FakeTriageResult(tools=[], skills=['recall', 'persistent_task'])
        assert _is_innate_skill_only(tr) is True

    def test_false_with_tools(self):
        """When tools are present, should return False regardless of skills."""
        tr = FakeTriageResult(tools=['searxng'], skills=['recall', 'schedule'])
        assert _is_innate_skill_only(tr) is False

    def test_false_with_only_primitives(self):
        """Primitives only (no contextual skill) → False."""
        tr = FakeTriageResult(tools=[], skills=['recall', 'memorize', 'introspect'])
        assert _is_innate_skill_only(tr) is False

    def test_false_with_empty_skills(self):
        tr = FakeTriageResult(tools=[], skills=[])
        assert _is_innate_skill_only(tr) is False


# ── Constants ────────────────────────────────────────────────────────

class TestConstants:

    def test_contextual_skills_set(self):
        assert _CONTEXTUAL_SKILLS == {'schedule', 'list', 'focus', 'persistent_task'}

    def test_primitives_set(self):
        assert _PRIMITIVES == {'recall', 'memorize', 'introspect', 'associate'}


# ── _handle_innate_skill_dispatch ────────────────────────────────────

class TestHandleInnateSkillDispatch:

    def test_schedule_dispatched_once(self):
        """Dispatcher.dispatch_action should be called exactly once for a schedule skill."""
        from workers.digest_worker import _handle_innate_skill_dispatch

        mock_config = MagicMock()
        mock_config.resolve_agent_config.return_value = {'model': 'test'}
        mock_config.get_agent_prompt.return_value = 'test prompt'

        mock_cortex = MagicMock()
        mock_cortex.generate_response.return_value = {
            'actions': [{'action_type': 'schedule', 'message': 'drink water', 'due_at': '2026-03-01T09:00:00Z'}],
            'generation_time': 0.5,
        }

        mock_dispatcher = MagicMock()
        mock_dispatcher.dispatch_action.return_value = {
            'action_type': 'schedule',
            'status': 'success',
            'execution_time': 0.1,
        }

        triage = FakeTriageResult(tools=[], skills=['recall', 'schedule'])
        thread_conv = MagicMock()
        thread_conv.get_conversation_history.return_value = []
        wm = MagicMock()

        # Patch at the package level where the function does its lazy imports
        with patch('services.ConfigService', mock_config), \
             patch('services.FrontalCortexService', return_value=mock_cortex), \
             patch('services.act_dispatcher_service.ActDispatcherService', return_value=mock_dispatcher):
            result = _handle_innate_skill_dispatch(
                triage, "remind me to drink water", "health", "thread-1", {},
                {'type': 'request'}, wm, thread_conv, 0.5, "ex-1",
            )

        assert result['mode'] == 'ACT'
        assert result['response'] == ''
        mock_dispatcher.dispatch_action.assert_called_once()

    def test_no_actions_returns_response_data(self):
        """When LLM returns no actions, should return response_data directly."""
        from workers.digest_worker import _handle_innate_skill_dispatch

        mock_config = MagicMock()
        mock_config.resolve_agent_config.return_value = {'model': 'test'}
        mock_config.get_agent_prompt.return_value = 'test prompt'

        mock_cortex = MagicMock()
        response_data = {
            'actions': [],
            'response': 'I understand you want a reminder.',
            'generation_time': 0.3,
        }
        mock_cortex.generate_response.return_value = response_data

        triage = FakeTriageResult(tools=[], skills=['recall', 'schedule'])
        thread_conv = MagicMock()
        thread_conv.get_conversation_history.return_value = []
        wm = MagicMock()

        with patch('services.ConfigService', mock_config), \
             patch('services.FrontalCortexService', return_value=mock_cortex):
            result = _handle_innate_skill_dispatch(
                triage, "maybe set a reminder", "general", "thread-1", {},
                {'type': 'request'}, wm, thread_conv, 0.5, "ex-1",
            )

        assert result == response_data

    def test_no_contextual_skills_returns_none(self):
        """If triage has no contextual skills, should return None (fallback signal)."""
        from workers.digest_worker import _handle_innate_skill_dispatch

        triage = FakeTriageResult(tools=[], skills=['recall', 'memorize'])
        thread_conv = MagicMock()
        wm = MagicMock()

        result = _handle_innate_skill_dispatch(
            triage, "test", "general", "thread-1", {},
            {}, wm, thread_conv, 0.5, "ex-1",
        )

        assert result is None


# ── Scheduler dedup guard ────────────────────────────────────────────

class TestSchedulerDedup:

    def test_dedup_rejects_within_60s(self, mock_db_rows):
        """If same message exists within 60s, should return dedup message."""
        from services.innate_skills.scheduler_skill import _create

        db, cursor = mock_db_rows

        # First fetchone returns existing row (dedup hit)
        cursor.fetchone.return_value = ('existing-id',)

        with patch('services.database_service.get_shared_db_service', return_value=db):
            result = _create("test-topic", {
                "message": "drink water",
                "due_at": "2099-01-01T12:00:00Z",
                "item_type": "notification",
            })

        assert "already exists" in result
        assert "existing-id" in result

    def test_dedup_allows_new_message(self, mock_db_rows):
        """If no duplicate found, should proceed with insert."""
        from services.innate_skills.scheduler_skill import _create

        db, cursor = mock_db_rows

        # fetchone returns None (no duplicate)
        cursor.fetchone.return_value = None

        with patch('services.database_service.get_shared_db_service', return_value=db), \
             patch('services.scheduler_card_service.SchedulerCardService'):
            result = _create("test-topic", {
                "message": "drink water",
                "due_at": "2099-01-01T12:00:00Z",
                "item_type": "notification",
            })

        assert result == "__CARD_ONLY__"
        # Should have executed 2 queries: SELECT (dedup) + INSERT
        assert cursor.execute.call_count == 2
