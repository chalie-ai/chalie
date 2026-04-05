"""Tests for sub_agent_skill — spawn a focused sub-agent for a specific task."""

import pytest
from unittest.mock import patch, MagicMock

from services.innate_skills.sub_agent_skill import (
    handle_sub_agent,
    TOOL_SCHEMA,
    _SUB_AGENT_SKILLS,
)


pytestmark = pytest.mark.unit


# ── Schema / constants ────────────────────────────────────────────────

class TestSubAgentSkillConstants:

    def test_tool_name(self):
        assert TOOL_SCHEMA['name'] == 'sub_agent'

    def test_goal_is_required(self):
        assert 'goal' in TOOL_SCHEMA['input_schema']['required']

    def test_sub_agent_skills_excludes_sub_agent(self):
        """Prevent recursive sub_agent spawning."""
        assert 'sub_agent' not in _SUB_AGENT_SKILLS

    def test_sub_agent_skills_excludes_persistent_task(self):
        """Prevent sub-agents from creating background tasks."""
        assert 'persistent_task' not in _SUB_AGENT_SKILLS

    def test_sub_agent_skills_includes_core_tools(self):
        """Sub-agents must have access to memory, find_tools, document, read."""
        for skill in ('memory', 'find_tools', 'document', 'read'):
            assert skill in _SUB_AGENT_SKILLS


# ── handle_sub_agent — happy path ─────────────────────────────────────

class TestHandleSubAgentHappyPath:

    @patch('services.act_orchestrator_service.ACTOrchestrator')
    @patch('services.FrontalCortexService')
    @patch('services.config_service.ConfigService.resolve_agent_config', return_value={})
    def test_returns_final_response(self, _mock_config, mock_cortex_cls, mock_orch_cls):
        """Happy path — goal in, final_response returned."""
        from services.act_orchestrator_service import ACTResult

        mock_cortex_cls.return_value = MagicMock()

        act_result = ACTResult()
        act_result.final_response = "I found 3 relevant papers."
        act_result.iterations_used = 4
        act_result.termination_reason = 'max_response'

        mock_orchestrator = MagicMock()
        mock_orchestrator.run.return_value = act_result
        mock_orch_cls.return_value = mock_orchestrator

        result = handle_sub_agent('parent-topic', {'goal': 'Find papers on LLMs'})

        assert result == "I found 3 relevant papers."

    @patch('services.act_orchestrator_service.ACTOrchestrator')
    @patch('services.FrontalCortexService')
    @patch('services.config_service.ConfigService.resolve_agent_config', return_value={})
    def test_no_final_response_returns_summary(self, _mock_config, mock_cortex_cls, mock_orch_cls):
        """When ACT result has no final_response, a summary string with termination_reason is returned."""
        from services.act_orchestrator_service import ACTResult

        mock_cortex_cls.return_value = MagicMock()

        act_result = ACTResult()
        act_result.final_response = ''
        act_result.iterations_used = 50
        act_result.termination_reason = 'max_iterations'

        mock_orchestrator = MagicMock()
        mock_orchestrator.run.return_value = act_result
        mock_orch_cls.return_value = mock_orchestrator

        result = handle_sub_agent('topic', {'goal': 'Do something'})

        assert 'max_iterations' in result
        assert '50' in result


# ── handle_sub_agent — parameter validation ───────────────────────────

class TestHandleSubAgentValidation:

    def test_missing_goal_returns_error(self):
        """Missing goal parameter returns an error string immediately."""
        result = handle_sub_agent('topic', {})

        assert 'error' in result.lower()
        assert 'goal' in result.lower()

    def test_empty_goal_returns_error(self):
        """Empty string goal returns an error string."""
        result = handle_sub_agent('topic', {'goal': '   '})

        assert 'error' in result.lower()
        assert 'goal' in result.lower()

    def test_none_goal_returns_error(self):
        """None goal returns an error string."""
        result = handle_sub_agent('topic', {'goal': None})

        assert 'error' in result.lower()


# ── handle_sub_agent — ACT orchestrator configuration ────────────────

class TestHandleSubAgentOrchestration:

    def _make_act_result(self, final_response='done'):
        from services.act_orchestrator_service import ACTResult
        r = ACTResult()
        r.final_response = final_response
        r.iterations_used = 1
        r.termination_reason = 'max_response'
        return r

    @patch('services.act_orchestrator_service.ACTOrchestrator')
    @patch('services.FrontalCortexService')
    @patch('services.config_service.ConfigService.resolve_agent_config', return_value={})
    def test_orchestrator_created_with_50_iterations_no_timeout(
        self, _mock_config, mock_cortex_cls, mock_orch_cls
    ):
        """Sub-agent ACTOrchestrator uses max_iterations=50 and both timeouts=None."""
        mock_cortex_cls.return_value = MagicMock()
        mock_orchestrator = MagicMock()
        mock_orchestrator.run.return_value = self._make_act_result()
        mock_orch_cls.return_value = mock_orchestrator

        handle_sub_agent('topic', {'goal': 'Search for X'})

        mock_orch_cls.assert_called_once()
        ctor_kwargs = mock_orch_cls.call_args.kwargs
        assert ctor_kwargs['max_iterations'] == 50
        assert ctor_kwargs['cumulative_timeout'] is None
        assert ctor_kwargs['per_action_timeout'] is None

    @patch('services.act_orchestrator_service.ACTOrchestrator')
    @patch('services.FrontalCortexService')
    @patch('services.config_service.ConfigService.resolve_agent_config', return_value={})
    def test_run_called_with_sub_agent_skills(
        self, _mock_config, mock_cortex_cls, mock_orch_cls
    ):
        """ACTOrchestrator.run() receives _SUB_AGENT_SKILLS as selected_skills."""
        mock_cortex_cls.return_value = MagicMock()
        mock_orchestrator = MagicMock()
        mock_orchestrator.run.return_value = self._make_act_result()
        mock_orch_cls.return_value = mock_orchestrator

        handle_sub_agent('topic', {'goal': 'Find data'})

        run_kwargs = mock_orchestrator.run.call_args.kwargs
        assert run_kwargs['selected_skills'] == _SUB_AGENT_SKILLS

    @patch('services.act_orchestrator_service.ACTOrchestrator')
    @patch('services.FrontalCortexService')
    @patch('services.config_service.ConfigService.resolve_agent_config', return_value={})
    def test_run_called_with_clean_chat_history(
        self, _mock_config, mock_cortex_cls, mock_orch_cls
    ):
        """Sub-agent runs with clean empty chat_history — no parent context leaked."""
        mock_cortex_cls.return_value = MagicMock()
        mock_orchestrator = MagicMock()
        mock_orchestrator.run.return_value = self._make_act_result()
        mock_orch_cls.return_value = mock_orchestrator

        handle_sub_agent('topic', {'goal': 'Summarise docs'})

        run_kwargs = mock_orchestrator.run.call_args.kwargs
        assert run_kwargs['chat_history'] == []

    @patch('services.act_orchestrator_service.ACTOrchestrator')
    @patch('services.FrontalCortexService')
    @patch('services.config_service.ConfigService.resolve_agent_config', return_value={})
    def test_goal_included_in_act_prompt(
        self, _mock_config, mock_cortex_cls, mock_orch_cls
    ):
        """The goal text is injected into the act_prompt before running."""
        mock_cortex_cls.return_value = MagicMock()
        mock_orchestrator = MagicMock()
        mock_orchestrator.run.return_value = self._make_act_result()
        mock_orch_cls.return_value = mock_orchestrator

        handle_sub_agent('topic', {'goal': 'Audit the security logs for January'})

        run_kwargs = mock_orchestrator.run.call_args.kwargs
        assert 'Audit the security logs for January' in run_kwargs['act_prompt']

    @patch('services.act_orchestrator_service.ACTOrchestrator')
    @patch('services.FrontalCortexService')
    @patch('services.config_service.ConfigService.resolve_agent_config', return_value={})
    def test_topic_namespaced_with_sub_agent_prefix(
        self, _mock_config, mock_cortex_cls, mock_orch_cls
    ):
        """Sub-agent topic is namespaced with 'sub_agent_' prefix to avoid collision."""
        mock_cortex_cls.return_value = MagicMock()
        mock_orchestrator = MagicMock()
        mock_orchestrator.run.return_value = self._make_act_result()
        mock_orch_cls.return_value = mock_orchestrator

        handle_sub_agent('parent-topic-xyz', {'goal': 'Do work'})

        run_kwargs = mock_orchestrator.run.call_args.kwargs
        assert 'parent-topic-xyz' in run_kwargs['topic']
        assert run_kwargs['topic'].startswith('sub_agent_')


# ── handle_sub_agent — error handling ────────────────────────────────

class TestHandleSubAgentErrors:

    @patch('services.act_orchestrator_service.ACTOrchestrator')
    @patch('services.FrontalCortexService')
    @patch('services.config_service.ConfigService.resolve_agent_config', return_value={})
    def test_act_loop_exception_returns_error_string(
        self, _mock_config, mock_cortex_cls, mock_orch_cls
    ):
        """ACT loop failure returns an error string — does not raise."""
        mock_cortex_cls.return_value = MagicMock()

        mock_orchestrator = MagicMock()
        mock_orchestrator.run.side_effect = RuntimeError("LLM timeout")
        mock_orch_cls.return_value = mock_orchestrator

        result = handle_sub_agent('topic', {'goal': 'Do something'})

        assert 'error' in result.lower()
        assert 'LLM timeout' in result

    @patch('services.config_service.ConfigService.resolve_agent_config',
           side_effect=Exception("No config found"))
    def test_config_exception_returns_error_string(self, _mock_config):
        """Config resolution failure returns an error string — does not raise."""
        result = handle_sub_agent('topic', {'goal': 'Do something'})

        assert 'error' in result.lower()

    @patch('services.act_orchestrator_service.ACTOrchestrator')
    @patch('services.FrontalCortexService')
    @patch('services.config_service.ConfigService.resolve_agent_config', return_value={})
    def test_error_message_truncated_to_200_chars(
        self, _mock_config, mock_cortex_cls, mock_orch_cls
    ):
        """Error message in returned string is capped at 200 characters."""
        mock_cortex_cls.return_value = MagicMock()

        long_error = "X" * 500
        mock_orchestrator = MagicMock()
        mock_orchestrator.run.side_effect = RuntimeError(long_error)
        mock_orch_cls.return_value = mock_orchestrator

        result = handle_sub_agent('topic', {'goal': 'Do something'})

        # The error portion (after the prefix) should not exceed 200 chars
        assert len(result) < 300  # prefix + 200 chars max
