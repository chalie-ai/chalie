"""Tests for persistent_task_worker and persistent_task_skill — event-driven constants and skill actions."""

import json
import pytest
from unittest.mock import patch, MagicMock, call

from workers.persistent_task_worker import (
    EXECUTE_QUEUE_KEY,
    BLPOP_TIMEOUT,
    OUTER_SKILLS,
)


pytestmark = pytest.mark.unit


# ── Constants ────────────────────────────────────────────────────────

class TestConstants:

    def test_execute_queue_key(self):
        assert EXECUTE_QUEUE_KEY == "persistent_task:execute"

    def test_blpop_timeout(self):
        """Heartbeat fallback fires every 5 minutes."""
        assert BLPOP_TIMEOUT == 300

    def test_outer_skills_includes_sub_agent(self):
        """Outer execution loop must include sub_agent for parallel step execution."""
        assert 'sub_agent' in OUTER_SKILLS

    def test_outer_skills_excludes_persistent_task(self):
        """Prevent recursive persistent task creation from the worker."""
        assert 'persistent_task' not in OUTER_SKILLS

    def test_outer_skills_includes_required_tools(self):
        """Outer loop must have memory, find_tools, document, read alongside sub_agent."""
        for skill in ('memory', 'find_tools', 'document', 'read'):
            assert skill in OUTER_SKILLS


# ── Skill: complete action ──────────────────────────────────────────

class TestSkillCompleteAction:

    @patch('services.innate_skills.persistent_task_skill._get_service')
    @patch('services.innate_skills.persistent_task_skill._get_account_id', return_value=1)
    def test_complete_happy_path(self, _mock_account, mock_get_svc):
        from services.innate_skills.persistent_task_skill import _complete

        mock_service = MagicMock()
        mock_service.get_task.return_value = {
            'id': 42, 'goal': 'Research quantum computing', 'thread_id': 't1',
            'progress': {},
        }
        mock_service.get_active_tasks.return_value = []
        mock_get_svc.return_value = mock_service

        result = _complete('topic', {'task_id': '42', 'result': 'Found 5 papers.'})

        mock_service.complete_task.assert_called_once_with(42, 'Found 5 papers.', None)
        assert '42' in result
        assert 'completed' in result.lower()

    @patch('services.innate_skills.persistent_task_skill._get_service')
    @patch('services.innate_skills.persistent_task_skill._get_account_id', return_value=1)
    def test_complete_task_not_found(self, _mock_account, mock_get_svc):
        from services.innate_skills.persistent_task_skill import _complete

        mock_service = MagicMock()
        mock_service.get_task.return_value = None
        mock_service.get_active_tasks.return_value = []
        mock_get_svc.return_value = mock_service

        result = _complete('topic', {'task_id': '99'})
        assert 'not found' in result.lower()

    @patch('services.innate_skills.persistent_task_skill._get_service')
    @patch('services.innate_skills.persistent_task_skill._get_account_id', return_value=1)
    def test_complete_uses_context_extras_fallback(self, _mock_account, mock_get_svc):
        """When no task_id in params, falls back to persistent_task_id from context_extras."""
        from services.innate_skills.persistent_task_skill import _complete

        mock_service = MagicMock()
        mock_service.get_task.return_value = {
            'id': 7, 'goal': 'Analyze logs', 'thread_id': 't2',
            'progress': {},
        }
        mock_service.get_active_tasks.return_value = []
        mock_get_svc.return_value = mock_service

        result = _complete('topic', {'persistent_task_id': 7, 'result': 'Done.'})

        mock_service.complete_task.assert_called_once_with(7, 'Done.', None)
        assert 'completed' in result.lower()

    @patch('services.innate_skills.persistent_task_skill._get_service')
    @patch('services.innate_skills.persistent_task_skill._get_account_id', return_value=1)
    def test_complete_no_task_id_at_all(self, _mock_account, mock_get_svc):
        from services.innate_skills.persistent_task_skill import _complete

        mock_service = MagicMock()
        mock_service.get_active_tasks.return_value = []
        mock_get_svc.return_value = mock_service

        result = _complete('topic', {})
        assert 'could not identify' in result.lower()


# ── _generate_plan ────────────────────────────────────────────────────

class TestGeneratePlan:

    def _make_cortex_mock(self, response_text):
        mock_cortex = MagicMock()
        mock_cortex.generate_response.return_value = {'response': response_text}
        return mock_cortex

    @patch('services.FrontalCortexService')
    @patch('services.config_service.ConfigService.get_agent_prompt', return_value="Plan this: {{goal}}")
    @patch('services.config_service.ConfigService.resolve_agent_config', return_value={})
    def test_valid_json_plan_parsed(self, _mock_resolve, _mock_prompt, mock_cortex_cls):
        """Valid JSON plan returned by LLM is parsed and returned as-is."""
        from workers.persistent_task_worker import _generate_plan

        plan_data = {
            "steps": [
                {"id": "s1", "action": "Search databases", "depends_on": [], "parallel": False},
                {"id": "s2", "action": "Summarise findings", "depends_on": ["s1"], "parallel": False},
            ]
        }
        mock_cortex_cls.return_value = self._make_cortex_mock(json.dumps(plan_data))

        result = _generate_plan("Research quantum computing")

        assert result['steps'] == plan_data['steps']
        assert len(result['steps']) == 2

    @patch('services.FrontalCortexService')
    @patch('services.config_service.ConfigService.get_agent_prompt', return_value="Plan: {{goal}}")
    @patch('services.config_service.ConfigService.resolve_agent_config', return_value={})
    def test_json_embedded_in_prose_parsed(self, _mock_resolve, _mock_prompt, mock_cortex_cls):
        """JSON embedded in surrounding prose text is extracted correctly."""
        from workers.persistent_task_worker import _generate_plan

        plan_data = {"steps": [{"id": "s1", "action": "Do the thing", "depends_on": [], "parallel": False}]}
        prose_response = f"Here is my plan:\n\n{json.dumps(plan_data)}\n\nI hope that helps!"
        mock_cortex_cls.return_value = self._make_cortex_mock(prose_response)

        result = _generate_plan("Do the thing")

        assert 'steps' in result
        assert result['steps'][0]['id'] == 's1'

    @patch('services.FrontalCortexService')
    @patch('services.config_service.ConfigService.get_agent_prompt', return_value="Plan: {{goal}}")
    @patch('services.config_service.ConfigService.resolve_agent_config', return_value={})
    def test_malformed_json_falls_back_to_single_step(self, _mock_resolve, _mock_prompt, mock_cortex_cls):
        """When LLM returns non-JSON, a single-step fallback plan is produced."""
        from workers.persistent_task_worker import _generate_plan

        mock_cortex_cls.return_value = self._make_cortex_mock("Sure, I can help with that!")

        result = _generate_plan("Research quantum computing")

        assert 'steps' in result
        assert len(result['steps']) == 1
        assert result['steps'][0]['id'] == 's1'
        assert result['steps'][0]['action'] == 'Research quantum computing'

    @patch('services.FrontalCortexService')
    @patch('services.config_service.ConfigService.get_agent_prompt', return_value="Plan: {{goal}}")
    @patch('services.config_service.ConfigService.resolve_agent_config', return_value={})
    def test_json_without_steps_key_falls_back(self, _mock_resolve, _mock_prompt, mock_cortex_cls):
        """JSON response that lacks 'steps' key triggers single-step fallback."""
        from workers.persistent_task_worker import _generate_plan

        # Valid JSON but missing 'steps'
        mock_cortex_cls.return_value = self._make_cortex_mock('{"plan": "do stuff"}')

        result = _generate_plan("My goal")

        assert len(result['steps']) == 1
        assert result['steps'][0]['action'] == 'My goal'

    @patch('services.FrontalCortexService')
    @patch('services.config_service.ConfigService.get_agent_prompt', return_value="Plan this goal: {{goal}}")
    @patch('services.config_service.ConfigService.resolve_agent_config', return_value={})
    def test_goal_inserted_into_prompt(self, _mock_resolve, _mock_prompt, mock_cortex_cls):
        """The goal is substituted into the prompt template before the LLM call."""
        from workers.persistent_task_worker import _generate_plan

        captured_calls = []

        mock_cortex = MagicMock()
        mock_cortex.generate_response.side_effect = lambda **kwargs: (
            captured_calls.append(kwargs) or
            {'response': '{"steps": [{"id": "s1", "action": "x", "depends_on": [], "parallel": false}]}'}
        )
        mock_cortex_cls.return_value = mock_cortex

        _generate_plan("Audit security logs")

        assert captured_calls
        assert "Audit security logs" in captured_calls[0]['system_prompt_template']
        assert "{{goal}}" not in captured_calls[0]['system_prompt_template']

    @patch('services.FrontalCortexService')
    @patch('services.config_service.ConfigService.get_agent_prompt', return_value="Plan: {{goal}}")
    @patch('services.config_service.ConfigService.resolve_agent_config', return_value={})
    def test_cortex_exception_propagates_to_caller(self, _mock_resolve, _mock_prompt, mock_cortex_cls):
        """A cortex LLM exception propagates from _generate_plan — caught by _process_task (crash-safe)."""
        from workers.persistent_task_worker import _generate_plan

        mock_cortex = MagicMock()
        mock_cortex.generate_response.side_effect = RuntimeError("LLM unavailable")
        mock_cortex_cls.return_value = mock_cortex

        # _generate_plan does NOT catch LLM errors — it propagates to _process_task
        with pytest.raises(RuntimeError, match="LLM unavailable"):
            _generate_plan("My goal")


# ── _execute_with_plan ────────────────────────────────────────────────

class TestExecuteWithPlan:

    def _make_act_result(self, final_response='Task done.', iterations=5, reason='max_response'):
        from services.act_orchestrator_service import ACTResult
        r = ACTResult()
        r.final_response = final_response
        r.iterations_used = iterations
        r.termination_reason = reason
        return r

    def _setup_execute_mocks(
        self,
        mock_config,
        mock_cortex_cls,
        mock_get_db,
        mock_task_svc_cls,
        task_status='in_progress',
        task_id=10,
    ):
        mock_config.resolve_agent_config.return_value = {}
        mock_config.get_agent_prompt.return_value = "Execute: {{goal}} {{plan_json}} {{completed_steps}}"
        mock_cortex_cls.return_value = MagicMock()

        mock_task_svc = MagicMock()
        mock_task_svc.get_task.return_value = {'id': task_id, 'status': task_status}
        mock_task_svc_cls.return_value = mock_task_svc
        mock_get_db.return_value = MagicMock()
        return mock_task_svc

    @patch('services.act_orchestrator_service.ACTOrchestrator')
    @patch('services.persistent_task_service.PersistentTaskService')
    @patch('services.database_service.get_shared_db_service')
    @patch('services.FrontalCortexService')
    @patch('services.config_service.ConfigService.get_agent_prompt', return_value="Execute: {{goal}} {{plan_json}} {{completed_steps}}")
    @patch('services.config_service.ConfigService.resolve_agent_config', return_value={})
    def test_orchestrator_created_with_200_iterations_no_timeout(
        self, _mock_resolve, _mock_prompt, mock_cortex_cls, mock_get_db, mock_task_svc_cls, mock_orch_cls
    ):
        """ACTOrchestrator must be created with max_iterations=200 and both timeouts=None."""
        from workers.persistent_task_worker import _execute_with_plan

        mock_cortex_cls.return_value = MagicMock()
        mock_task_svc = MagicMock()
        mock_task_svc.get_task.return_value = {'id': 10, 'status': 'in_progress'}
        mock_task_svc_cls.return_value = mock_task_svc
        mock_get_db.return_value = MagicMock()

        mock_orchestrator = MagicMock()
        mock_orchestrator.run.return_value = self._make_act_result()
        mock_orch_cls.return_value = mock_orchestrator

        task = {'id': 10, 'goal': 'Research topic', 'thread_id': 't1'}
        plan = {"steps": [{"id": "s1", "action": "Search", "depends_on": [], "parallel": False}]}

        _execute_with_plan(task, plan)

        mock_orch_cls.assert_called_once()
        ctor_kwargs = mock_orch_cls.call_args.kwargs
        assert ctor_kwargs['max_iterations'] == 200
        assert ctor_kwargs['cumulative_timeout'] is None
        assert ctor_kwargs['per_action_timeout'] is None

    @patch('services.act_orchestrator_service.ACTOrchestrator')
    @patch('services.persistent_task_service.PersistentTaskService')
    @patch('services.database_service.get_shared_db_service')
    @patch('services.FrontalCortexService')
    @patch('services.config_service.ConfigService.get_agent_prompt', return_value="Execute: {{goal}} {{plan_json}} {{completed_steps}}")
    @patch('services.config_service.ConfigService.resolve_agent_config', return_value={})
    def test_run_called_with_outer_skills(
        self, _mock_resolve, _mock_prompt, mock_cortex_cls, mock_get_db, mock_task_svc_cls, mock_orch_cls
    ):
        """ACTOrchestrator.run() must receive OUTER_SKILLS as selected_skills."""
        from workers.persistent_task_worker import _execute_with_plan, OUTER_SKILLS

        mock_cortex_cls.return_value = MagicMock()
        mock_task_svc = MagicMock()
        mock_task_svc.get_task.return_value = {'id': 10, 'status': 'in_progress'}
        mock_task_svc_cls.return_value = mock_task_svc
        mock_get_db.return_value = MagicMock()

        mock_orchestrator = MagicMock()
        mock_orchestrator.run.return_value = self._make_act_result()
        mock_orch_cls.return_value = mock_orchestrator

        task = {'id': 10, 'goal': 'Do work', 'thread_id': 't1'}
        plan = {"steps": []}

        _execute_with_plan(task, plan)

        run_kwargs = mock_orchestrator.run.call_args.kwargs
        assert run_kwargs['selected_skills'] == OUTER_SKILLS

    @patch('services.act_orchestrator_service.ACTOrchestrator')
    @patch('services.persistent_task_service.PersistentTaskService')
    @patch('services.database_service.get_shared_db_service')
    @patch('services.FrontalCortexService')
    @patch('services.config_service.ConfigService.get_agent_prompt', return_value="Execute: {{goal}} {{plan_json}} {{completed_steps}}")
    @patch('services.config_service.ConfigService.resolve_agent_config', return_value={})
    def test_run_called_with_empty_chat_history(
        self, _mock_resolve, _mock_prompt, mock_cortex_cls, mock_get_db, mock_task_svc_cls, mock_orch_cls
    ):
        """ACT loop runs with clean empty chat_history (no parent context)."""
        from workers.persistent_task_worker import _execute_with_plan

        mock_cortex_cls.return_value = MagicMock()
        mock_task_svc = MagicMock()
        mock_task_svc.get_task.return_value = {'id': 5, 'status': 'in_progress'}
        mock_task_svc_cls.return_value = mock_task_svc
        mock_get_db.return_value = MagicMock()

        mock_orchestrator = MagicMock()
        mock_orchestrator.run.return_value = self._make_act_result()
        mock_orch_cls.return_value = mock_orchestrator

        task = {'id': 5, 'goal': 'Do something', 'thread_id': 't1'}
        plan = {"steps": []}

        _execute_with_plan(task, plan)

        run_kwargs = mock_orchestrator.run.call_args.kwargs
        assert run_kwargs['chat_history'] == []

    def _run_execute_with_plan_capture_callback(self, task_status, task_id=3):
        """Helper: run _execute_with_plan and capture the on_iteration_complete callback."""
        from workers.persistent_task_worker import _execute_with_plan
        from services.act_orchestrator_service import ACTOrchestrator as RealOrchestrator

        captured_callback = []

        mock_task_svc = MagicMock()
        mock_task_svc.get_task.return_value = (
            {'id': task_id, 'status': task_status} if task_status else None
        )

        mock_orchestrator = MagicMock()

        def capture_run(**kwargs):
            captured_callback.append(kwargs.get('on_iteration_complete'))
            return self._make_act_result()

        mock_orchestrator.run.side_effect = capture_run

        with patch('services.config_service.ConfigService.resolve_agent_config', return_value={}), \
             patch('services.config_service.ConfigService.get_agent_prompt',
                   return_value="Execute: {{goal}} {{plan_json}} {{completed_steps}}"), \
             patch('services.FrontalCortexService', return_value=MagicMock()), \
             patch('services.database_service.get_shared_db_service', return_value=MagicMock()), \
             patch('services.persistent_task_service.PersistentTaskService', return_value=mock_task_svc), \
             patch('services.act_orchestrator_service.ACTOrchestrator', return_value=mock_orchestrator):
            task = {'id': task_id, 'goal': 'Some goal', 'thread_id': None}
            plan = {"steps": []}
            _execute_with_plan(task, plan)

        return captured_callback, mock_task_svc

    def test_on_iteration_complete_returns_cancelled_when_task_paused(self):
        """on_iteration_complete callback returns 'cancelled' when task status is 'paused'."""
        captured_callback, _ = self._run_execute_with_plan_capture_callback('paused', task_id=3)
        assert captured_callback[0] is not None
        result = captured_callback[0](MagicMock(), MagicMock(), [], None)
        assert result == 'cancelled'

    def test_on_iteration_complete_returns_cancelled_when_task_cancelled(self):
        """on_iteration_complete callback returns 'cancelled' when task status is 'cancelled'."""
        captured_callback, _ = self._run_execute_with_plan_capture_callback('cancelled', task_id=4)
        result = captured_callback[0](MagicMock(), MagicMock(), [], None)
        assert result == 'cancelled'

    def test_on_iteration_complete_returns_none_when_task_in_progress(self):
        """on_iteration_complete returns None (continue) when task is still in_progress."""
        captured_callback, _ = self._run_execute_with_plan_capture_callback('in_progress', task_id=5)
        result = captured_callback[0](MagicMock(), MagicMock(), [], None)
        assert result is None

    def test_on_iteration_complete_returns_cancelled_when_task_missing(self):
        """on_iteration_complete returns 'cancelled' when task row no longer exists."""
        captured_callback, _ = self._run_execute_with_plan_capture_callback(None, task_id=6)
        result = captured_callback[0](MagicMock(), MagicMock(), [], None)
        assert result == 'cancelled'


# ── _surface_via_digest ───────────────────────────────────────────────

class TestSurfaceViaDigest:

    @patch('workers.digest_worker.digest_worker')
    def test_digest_worker_called_with_correct_metadata(self, mock_digest):
        """Phase 3 must call digest_worker with task metadata and source='persistent_task_result'."""
        from workers.persistent_task_worker import _surface_via_digest

        task = {'id': 55, 'goal': 'Compile report', 'thread_id': 'thread-abc'}
        _surface_via_digest(task, "Here are the results.")

        mock_digest.assert_called_once_with(
            text="Here are the results.",
            metadata={
                'source': 'persistent_task_result',
                'task_id': 55,
                'thread_id': 'thread-abc',
                'goal': 'Compile report',
            }
        )

    @patch('workers.digest_worker.digest_worker')
    def test_digest_worker_called_when_thread_id_missing(self, mock_digest):
        """thread_id=None is passed through — digest worker handles resolution."""
        from workers.persistent_task_worker import _surface_via_digest

        task = {'id': 7, 'goal': 'Run audit', 'thread_id': None}
        _surface_via_digest(task, "Audit done.")

        call_kwargs = mock_digest.call_args.kwargs
        assert call_kwargs['metadata']['thread_id'] is None
        assert call_kwargs['metadata']['source'] == 'persistent_task_result'

    @patch('workers.digest_worker.digest_worker', side_effect=RuntimeError("Digest exploded"))
    def test_digest_failure_does_not_crash(self, mock_digest):
        """A digest worker exception must be swallowed — does not propagate."""
        from workers.persistent_task_worker import _surface_via_digest

        task = {'id': 8, 'goal': 'Something', 'thread_id': None}
        # Must not raise
        _surface_via_digest(task, "result text")


# ── _process_task ─────────────────────────────────────────────────────

class TestProcessTask:

    def _make_task(self, status='accepted', task_id=1):
        return {
            'id': task_id,
            'goal': 'Research the topic',
            'thread_id': 'thread-1',
            'status': status,
            'progress': {},
        }

    def _make_act_result(self, final_response='Done.', reason='max_response'):
        from services.act_orchestrator_service import ACTResult
        r = ACTResult()
        r.final_response = final_response
        r.iterations_used = 10
        r.termination_reason = reason
        return r

    @patch('workers.persistent_task_worker._surface_via_digest')
    @patch('workers.persistent_task_worker._execute_with_plan')
    @patch('workers.persistent_task_worker._generate_plan')
    def test_happy_path_plan_execute_surface(
        self, mock_plan, mock_execute, mock_surface
    ):
        """Full three-phase execution: plan → execute → surface all complete successfully."""
        from workers.persistent_task_worker import _process_task

        mock_plan.return_value = {"steps": [{"id": "s1", "action": "x"}]}
        mock_execute.return_value = self._make_act_result('Final answer')

        task_service = MagicMock()
        task_service.transition.return_value = (True, "ok")
        # After planning, task is still in_progress
        task_service.get_task.return_value = self._make_task('in_progress')

        task = self._make_task('accepted')
        _process_task(task_service, task)

        mock_plan.assert_called_once_with('Research the topic')
        mock_execute.assert_called_once()
        task_service.complete_task.assert_called_once_with(1, 'Final answer')
        mock_surface.assert_called_once()

    @patch('workers.persistent_task_worker._surface_via_digest')
    @patch('workers.persistent_task_worker._execute_with_plan')
    @patch('workers.persistent_task_worker._generate_plan')
    def test_transitions_accepted_to_in_progress(
        self, mock_plan, mock_execute, mock_surface
    ):
        """Accepted task is transitioned to in_progress before execution begins."""
        from workers.persistent_task_worker import _process_task

        mock_plan.return_value = {"steps": []}
        mock_execute.return_value = self._make_act_result()

        task_service = MagicMock()
        task_service.transition.return_value = (True, "transitioned")
        task_service.get_task.return_value = self._make_task('in_progress')

        _process_task(task_service, self._make_task('accepted'))

        task_service.transition.assert_called_once_with(1, 'in_progress')

    @patch('workers.persistent_task_worker._generate_plan')
    def test_transition_failure_aborts_processing(self, mock_plan):
        """If transition to in_progress fails, processing is abandoned immediately."""
        from workers.persistent_task_worker import _process_task

        task_service = MagicMock()
        task_service.transition.return_value = (False, "already running")

        _process_task(task_service, self._make_task('accepted'))

        mock_plan.assert_not_called()

    @patch('workers.persistent_task_worker._surface_via_digest')
    @patch('workers.persistent_task_worker._execute_with_plan')
    @patch('workers.persistent_task_worker._generate_plan')
    def test_task_paused_between_plan_and_execute(
        self, mock_plan, mock_execute, mock_surface
    ):
        """If task is paused after planning, execution phase is skipped."""
        from workers.persistent_task_worker import _process_task

        mock_plan.return_value = {"steps": []}

        task_service = MagicMock()
        task_service.transition.return_value = (True, "ok")
        # Task is paused between phases
        task_service.get_task.return_value = self._make_task('paused')

        _process_task(task_service, self._make_task('accepted'))

        mock_execute.assert_not_called()
        mock_surface.assert_not_called()
        task_service.complete_task.assert_not_called()

    @patch('workers.persistent_task_worker._surface_via_digest')
    @patch('workers.persistent_task_worker._execute_with_plan')
    @patch('workers.persistent_task_worker._generate_plan')
    def test_task_cancelled_between_plan_and_execute(
        self, mock_plan, mock_execute, mock_surface
    ):
        """If task is cancelled after planning, execution phase is skipped."""
        from workers.persistent_task_worker import _process_task

        mock_plan.return_value = {"steps": []}

        task_service = MagicMock()
        task_service.transition.return_value = (True, "ok")
        task_service.get_task.return_value = self._make_task('cancelled')

        _process_task(task_service, self._make_task('accepted'))

        mock_execute.assert_not_called()
        mock_surface.assert_not_called()

    @patch('workers.persistent_task_worker._surface_via_digest')
    @patch('workers.persistent_task_worker._execute_with_plan')
    @patch('workers.persistent_task_worker._generate_plan', side_effect=RuntimeError("Plan boom"))
    def test_exception_during_planning_stalls_task(
        self, mock_plan, mock_execute, mock_surface
    ):
        """Exception in _generate_plan stalls the task and does not propagate."""
        from workers.persistent_task_worker import _process_task

        task_service = MagicMock()
        task_service.transition.return_value = (True, "ok")

        # Must not raise
        _process_task(task_service, self._make_task('accepted'))

        mock_execute.assert_not_called()
        mock_surface.assert_not_called()
        task_service.stall_task.assert_called_once_with(1, "Plan boom")

    @patch('workers.persistent_task_worker._surface_via_digest')
    @patch('workers.persistent_task_worker._execute_with_plan', side_effect=RuntimeError("Execute boom"))
    @patch('workers.persistent_task_worker._generate_plan')
    def test_exception_during_execution_stalls_task(
        self, mock_plan, mock_execute, mock_surface
    ):
        """Exception in _execute_with_plan stalls the task instead of leaving it stuck."""
        from workers.persistent_task_worker import _process_task

        mock_plan.return_value = {"steps": []}

        task_service = MagicMock()
        task_service.transition.return_value = (True, "ok")
        task_service.get_task.return_value = self._make_task('in_progress')

        # Must not raise
        _process_task(task_service, self._make_task('accepted'))

        mock_surface.assert_not_called()
        task_service.complete_task.assert_not_called()
        task_service.stall_task.assert_called_once_with(1, "Execute boom")

    @patch('workers.persistent_task_worker._surface_via_digest')
    @patch('workers.persistent_task_worker._execute_with_plan')
    @patch('workers.persistent_task_worker._generate_plan')
    def test_empty_final_response_falls_back_to_termination_reason(
        self, mock_plan, mock_execute, mock_surface
    ):
        """When ACT result has no final_response, a fallback message using termination_reason is used."""
        from workers.persistent_task_worker import _process_task
        from services.act_orchestrator_service import ACTResult

        mock_plan.return_value = {"steps": []}

        act_result = ACTResult()
        act_result.final_response = ''  # empty
        act_result.iterations_used = 50
        act_result.termination_reason = 'max_iterations'
        mock_execute.return_value = act_result

        task_service = MagicMock()
        task_service.transition.return_value = (True, "ok")
        task_service.get_task.return_value = self._make_task('in_progress')

        _process_task(task_service, self._make_task('accepted'))

        # complete_task should be called with a non-empty fallback string
        args = task_service.complete_task.call_args[0]
        assert 'max_iterations' in args[1]
        assert args[1]  # not empty


# ── _run_cycle ────────────────────────────────────────────────────────

class TestRunCycle:

    @patch('workers.persistent_task_worker._process_task')
    @patch('services.persistent_task_service.PersistentTaskService')
    @patch('services.database_service.get_shared_db_service')
    def test_run_cycle_picks_eligible_task(
        self, mock_get_db, mock_task_svc_cls, mock_process
    ):
        """_run_cycle calls _process_task when an eligible task exists."""
        from workers.persistent_task_worker import _run_cycle

        task = {'id': 10, 'goal': 'Do stuff', 'status': 'accepted', 'thread_id': None}
        mock_task_svc = MagicMock()
        mock_task_svc.get_eligible_task.return_value = task
        mock_task_svc.expire_stale_tasks.return_value = 0
        mock_task_svc_cls.return_value = mock_task_svc
        mock_get_db.return_value = MagicMock()

        _run_cycle()

        mock_process.assert_called_once_with(mock_task_svc, task)

    @patch('workers.persistent_task_worker._process_task')
    @patch('services.persistent_task_service.PersistentTaskService')
    @patch('services.database_service.get_shared_db_service')
    def test_run_cycle_no_eligible_tasks_returns_cleanly(
        self, mock_get_db, mock_task_svc_cls, mock_process
    ):
        """_run_cycle returns without error when no eligible tasks exist."""
        from workers.persistent_task_worker import _run_cycle

        mock_task_svc = MagicMock()
        mock_task_svc.get_eligible_task.return_value = None
        mock_task_svc.expire_stale_tasks.return_value = 0
        mock_task_svc_cls.return_value = mock_task_svc
        mock_get_db.return_value = MagicMock()

        _run_cycle()

        mock_process.assert_not_called()

    @patch('workers.persistent_task_worker._process_task')
    @patch('services.persistent_task_service.PersistentTaskService')
    @patch('services.database_service.get_shared_db_service')
    def test_run_cycle_expires_stale_tasks(
        self, mock_get_db, mock_task_svc_cls, mock_process
    ):
        """_run_cycle always calls expire_stale_tasks before checking for work."""
        from workers.persistent_task_worker import _run_cycle

        mock_task_svc = MagicMock()
        mock_task_svc.get_eligible_task.return_value = None
        mock_task_svc.expire_stale_tasks.return_value = 3
        mock_task_svc_cls.return_value = mock_task_svc
        mock_get_db.return_value = MagicMock()

        _run_cycle()

        mock_task_svc.expire_stale_tasks.assert_called_once()


# ── Skill: create action ──────────────────────────────────────────────

class TestSkillCreateAction:

    @patch('services.innate_skills.persistent_task_skill._get_service')
    @patch('services.innate_skills.persistent_task_skill._get_account_id', return_value=1)
    def test_create_returns_json_with_task_id(self, _mock_account, mock_get_svc):
        """create action returns JSON with success=True and a task_id."""
        from services.innate_skills.persistent_task_skill import _create

        mock_service = MagicMock()
        mock_service.find_duplicate.return_value = None
        mock_service.create_task.return_value = {'id': 99, 'goal': 'Do research', 'status': 'accepted'}
        mock_get_svc.return_value = mock_service

        result = _create('topic', {'goal': 'Do research'})
        data = json.loads(result)

        assert data['success'] is True
        assert data['task_id'] == 99

    @patch('services.innate_skills.persistent_task_skill._get_service')
    @patch('services.innate_skills.persistent_task_skill._get_account_id', return_value=1)
    def test_create_passes_max_iterations_200(self, _mock_account, mock_get_svc):
        """create action always passes max_iterations=200 to create_task."""
        from services.innate_skills.persistent_task_skill import _create

        mock_service = MagicMock()
        mock_service.find_duplicate.return_value = None
        mock_service.create_task.return_value = {'id': 1, 'goal': 'Do research', 'status': 'accepted'}
        mock_get_svc.return_value = mock_service

        _create('topic', {'goal': 'Do research'})

        call_kwargs = mock_service.create_task.call_args.kwargs
        assert call_kwargs.get('max_iterations') == 200

    @patch('services.innate_skills.persistent_task_skill._get_service')
    @patch('services.innate_skills.persistent_task_skill._get_account_id', return_value=1)
    def test_create_no_goal_returns_error_json(self, _mock_account, mock_get_svc):
        """create action with no goal returns JSON error — does not create a task."""
        from services.innate_skills.persistent_task_skill import _create

        mock_service = MagicMock()
        mock_get_svc.return_value = mock_service

        result = _create('topic', {})
        data = json.loads(result)

        assert data['success'] is False
        mock_service.create_task.assert_not_called()

    @patch('services.innate_skills.persistent_task_skill._get_service')
    @patch('services.innate_skills.persistent_task_skill._get_account_id', return_value=1)
    def test_create_blocks_duplicate_task(self, _mock_account, mock_get_svc):
        """create action returns error JSON when a similar task already exists."""
        from services.innate_skills.persistent_task_skill import _create

        mock_service = MagicMock()
        mock_service.find_duplicate.return_value = {
            'id': 5, 'goal': 'Do research on topic', 'status': 'in_progress'
        }
        mock_get_svc.return_value = mock_service

        result = _create('topic', {'goal': 'Do research'})
        data = json.loads(result)

        assert data['success'] is False
        assert 'similar' in data['error'].lower() or 'in progress' in data['error'].lower()
        mock_service.create_task.assert_not_called()


# ── Skill: status, list, pause, resume, cancel actions ───────────────

class TestSkillControlActions:

    @patch('services.innate_skills.persistent_task_skill._get_service')
    @patch('services.innate_skills.persistent_task_skill._get_account_id', return_value=1)
    def test_list_returns_active_tasks(self, _mock_account, mock_get_svc):
        """list action returns formatted list of active tasks."""
        from services.innate_skills.persistent_task_skill import _list_tasks

        mock_service = MagicMock()
        mock_service.get_active_tasks.return_value = [
            {'id': 1, 'status': 'in_progress', 'goal': 'Research quantum computing'},
            {'id': 2, 'status': 'accepted', 'goal': 'Compile audit report'},
        ]
        mock_get_svc.return_value = mock_service

        result = _list_tasks('topic', {})

        assert '#1' in result
        assert '#2' in result
        assert 'in_progress' in result

    @patch('services.innate_skills.persistent_task_skill._get_service')
    @patch('services.innate_skills.persistent_task_skill._get_account_id', return_value=1)
    def test_list_no_tasks_returns_empty_message(self, _mock_account, mock_get_svc):
        """list action returns friendly message when no tasks exist."""
        from services.innate_skills.persistent_task_skill import _list_tasks

        mock_service = MagicMock()
        mock_service.get_active_tasks.return_value = []
        mock_get_svc.return_value = mock_service

        result = _list_tasks('topic', {})

        assert 'no' in result.lower() or 'empty' in result.lower() or "don't have" in result.lower()

    @patch('services.innate_skills.persistent_task_skill._get_service')
    @patch('services.innate_skills.persistent_task_skill._get_account_id', return_value=1)
    def test_pause_calls_transition(self, _mock_account, mock_get_svc):
        """pause action calls service.transition(task_id, 'paused')."""
        from services.innate_skills.persistent_task_skill import _pause

        mock_service = MagicMock()
        mock_service.transition.return_value = (True, "Task paused.")
        mock_get_svc.return_value = mock_service

        result = _pause('topic', {'task_id': 3})

        mock_service.transition.assert_called_once_with(3, 'paused')

    @patch('services.innate_skills.persistent_task_skill._get_service')
    @patch('services.innate_skills.persistent_task_skill._get_account_id', return_value=1)
    def test_resume_calls_transition_in_progress(self, _mock_account, mock_get_svc):
        """resume action calls service.transition(task_id, 'in_progress')."""
        from services.innate_skills.persistent_task_skill import _resume

        mock_service = MagicMock()
        mock_service.transition.return_value = (True, "Task resumed.")
        mock_get_svc.return_value = mock_service

        result = _resume('topic', {'task_id': 3})

        mock_service.transition.assert_called_once_with(3, 'in_progress')

    @patch('services.innate_skills.persistent_task_skill._get_service')
    @patch('services.innate_skills.persistent_task_skill._get_account_id', return_value=1)
    def test_cancel_calls_transition_cancelled(self, _mock_account, mock_get_svc):
        """cancel action calls service.transition(task_id, 'cancelled')."""
        from services.innate_skills.persistent_task_skill import _cancel

        mock_service = MagicMock()
        mock_service.transition.return_value = (True, "Task cancelled.")
        mock_get_svc.return_value = mock_service

        result = _cancel('topic', {'task_id': 5})

        mock_service.transition.assert_called_once_with(5, 'cancelled')

    @patch('services.innate_skills.persistent_task_skill._get_service')
    @patch('services.innate_skills.persistent_task_skill._get_account_id', return_value=1)
    def test_pause_no_task_id_returns_error(self, _mock_account, mock_get_svc):
        """pause without a resolvable task_id returns an error message."""
        from services.innate_skills.persistent_task_skill import _pause

        mock_service = MagicMock()
        mock_service.get_active_tasks.return_value = []
        mock_get_svc.return_value = mock_service

        result = _pause('topic', {})

        assert 'could not identify' in result.lower()
        mock_service.transition.assert_not_called()

    def test_handle_unknown_action_returns_error(self):
        """handle_persistent_task returns an error string for unknown actions."""
        from services.innate_skills.persistent_task_skill import handle_persistent_task

        result = handle_persistent_task('topic', {'action': 'plan'})
        assert 'unknown' in result.lower() or 'plan' in result.lower()

    def test_tool_schema_does_not_include_dropped_actions(self):
        """The TOOL_SCHEMA enum must not include plan/expand/progress/priority actions."""
        from services.innate_skills.persistent_task_skill import TOOL_SCHEMA

        allowed_actions = set(
            TOOL_SCHEMA['input_schema']['properties']['action']['enum']
        )
        for dropped in ('plan', 'expand', 'progress', 'priority'):
            assert dropped not in allowed_actions, f"Dropped action '{dropped}' still in TOOL_SCHEMA"
