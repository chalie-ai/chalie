"""Tests for GoalPursuitProcessor — background goal execution subclass.

Verifies class-level constants, channel formation, tool exclusion, system prompt
loading, and the process() → send() delegation contract.
"""

import os
import pytest
from unittest.mock import MagicMock, patch, mock_open

from services.goal_pursuit_processor import GoalPursuitProcessor, _load_system_prompt

pytestmark = pytest.mark.unit


# ── Class-level constants ──────────────────────────────────────────────────────

class TestGoalPursuitProcessorConstants:
    def test_max_iterations_is_50(self):
        """GoalPursuitProcessor must allow 50 loop iterations (vs. base 30)."""
        assert GoalPursuitProcessor.MAX_ITERATIONS == 50

    def test_max_timeout_is_7200(self):
        """GoalPursuitProcessor must allow a 2-hour timeout (vs. base 900s)."""
        assert GoalPursuitProcessor.MAX_TIMEOUT == 7200

    def test_max_iterations_exceeds_base(self):
        """Subclass limit must exceed the base MessageProcessor default."""
        from services.message_processor import MAX_ITERATIONS as BASE_MAX
        assert GoalPursuitProcessor.MAX_ITERATIONS > BASE_MAX

    def test_max_timeout_exceeds_base(self):
        """Subclass timeout must exceed the base MessageProcessor default."""
        from services.message_processor import MAX_TIMEOUT as BASE_MAX
        assert GoalPursuitProcessor.MAX_TIMEOUT > BASE_MAX


# ── System prompt loading ──────────────────────────────────────────────────────

class TestLoadSystemPrompt:
    def test_loads_from_goal_pursuit_md(self):
        """_load_system_prompt reads the goal-pursuit.md file successfully."""
        prompt = _load_system_prompt()
        # The file exists in the repo — must not be the fallback
        assert prompt
        assert len(prompt) > 0

    def test_prompt_is_stripped(self):
        """Leading/trailing whitespace is stripped from the loaded prompt."""
        prompt = _load_system_prompt()
        assert prompt == prompt.strip()

    def test_fallback_prompt_used_when_file_missing(self):
        """When the prompt file cannot be read, a safe fallback string is returned."""
        with patch('builtins.open', side_effect=FileNotFoundError('no such file')):
            prompt = _load_system_prompt()

        assert prompt
        # Fallback must still be a meaningful instruction
        assert 'Chalie' in prompt or 'goal' in prompt.lower() or 'pursue' in prompt.lower()

    def test_fallback_prompt_used_on_permission_error(self):
        """When the prompt file raises PermissionError, fallback is returned."""
        with patch('builtins.open', side_effect=PermissionError('denied')):
            prompt = _load_system_prompt()

        assert prompt
        assert len(prompt) > 0

    def test_real_prompt_references_goal(self):
        """The real prompt file should give goal-pursuit context."""
        prompt = _load_system_prompt()
        # The file content is about pursuing a goal
        lower = prompt.lower()
        assert 'goal' in lower or 'pursue' in lower or 'accomplish' in lower


# ── Channel formation ──────────────────────────────────────────────────────────

class TestGoalPursuitProcessorChannel:
    def _make_processor_and_capture_send(self):
        """Return a processor instance with send() patched to capture kwargs."""
        processor = GoalPursuitProcessor()
        captured = {}

        def _fake_send(user_prompt, system_prompt, **kwargs):
            captured['user_prompt'] = user_prompt
            captured['system_prompt'] = system_prompt
            captured['kwargs'] = kwargs
            return {'response': 'done', 'tool_calls': None, 'actions': None}

        processor.send = _fake_send
        return processor, captured

    def test_channel_formatted_as_goal_pursuit_prefix(self):
        """Channel must be 'goal_pursuit:<pursuit_id>'."""
        processor, captured = self._make_processor_and_capture_send()

        with patch('services.tool_schema_service.get_skill_schemas', return_value=[]), \
             patch('services.innate_skills.registry.ALL_SKILL_NAMES', frozenset(['memory'])):
            processor.process('Build a report', 'abc123')

        assert captured['kwargs']['channel'] == 'goal_pursuit:abc123'

    def test_channel_uses_exact_pursuit_id(self):
        """The channel's pursuit_id segment must be the exact string passed in."""
        processor, captured = self._make_processor_and_capture_send()
        pursuit_id = 'deadbeef12345678deadbeef12345678'

        with patch('services.tool_schema_service.get_skill_schemas', return_value=[]), \
             patch('services.innate_skills.registry.ALL_SKILL_NAMES', frozenset(['memory'])):
            processor.process('Do a task', pursuit_id)

        assert captured['kwargs']['channel'] == f'goal_pursuit:{pursuit_id}'

    def test_job_is_frontal_cortex_unified(self):
        """Job must be 'frontal-cortex-unified' so the right LLM config is used."""
        processor, captured = self._make_processor_and_capture_send()

        with patch('services.tool_schema_service.get_skill_schemas', return_value=[]), \
             patch('services.innate_skills.registry.ALL_SKILL_NAMES', frozenset(['memory'])):
            processor.process('Investigate something', 'someid')

        assert captured['kwargs']['job'] == 'frontal-cortex-unified'

    def test_goal_passed_as_user_prompt(self):
        """The goal string must be forwarded as the user_prompt."""
        processor, captured = self._make_processor_and_capture_send()
        goal = 'Find all open pull requests and summarise them'

        with patch('services.tool_schema_service.get_skill_schemas', return_value=[]), \
             patch('services.innate_skills.registry.ALL_SKILL_NAMES', frozenset(['memory'])):
            processor.process(goal, 'anid')

        assert captured['user_prompt'] == goal


# ── Tool exclusion (anti-recursion) ───────────────────────────────────────────

class TestGoalPursuitToolExclusion:
    def test_goal_pursuit_excluded_from_tool_list(self):
        """'goal_pursuit' must NOT appear in the tools passed to send()."""
        processor = GoalPursuitProcessor()
        captured_tools = {}

        def _fake_send(user_prompt, system_prompt, **kwargs):
            captured_tools['tools'] = kwargs.get('tools', [])
            return {'response': 'ok', 'tool_calls': None, 'actions': None}

        processor.send = _fake_send

        skill_names = frozenset({'memory', 'goal_pursuit', 'find_tools', 'goals'})
        fake_schemas = [
            {'name': 'memory', 'description': '', 'input_schema': {}},
            {'name': 'find_tools', 'description': '', 'input_schema': {}},
            {'name': 'goals', 'description': '', 'input_schema': {}},
        ]

        with patch('services.innate_skills.registry.ALL_SKILL_NAMES', skill_names), \
             patch('services.tool_schema_service.get_skill_schemas', return_value=fake_schemas):
            processor.process('some goal', 'pid1')

        tool_names = [t['name'] for t in captured_tools['tools']]
        assert 'goal_pursuit' not in tool_names

    def test_other_skills_are_included(self):
        """Skills other than 'goal_pursuit' must still be included."""
        processor = GoalPursuitProcessor()
        captured_tools = {}

        def _fake_send(user_prompt, system_prompt, **kwargs):
            captured_tools['tools'] = kwargs.get('tools', [])
            return {'response': 'ok', 'tool_calls': None, 'actions': None}

        processor.send = _fake_send

        skill_names = frozenset({'memory', 'goal_pursuit', 'find_tools'})
        fake_schemas = [
            {'name': 'memory', 'description': '', 'input_schema': {}},
            {'name': 'find_tools', 'description': '', 'input_schema': {}},
        ]

        with patch('services.innate_skills.registry.ALL_SKILL_NAMES', skill_names), \
             patch('services.tool_schema_service.get_skill_schemas', return_value=fake_schemas):
            processor.process('some goal', 'pid2')

        tool_names = [t['name'] for t in captured_tools['tools']]
        assert 'memory' in tool_names
        assert 'find_tools' in tool_names

    def test_tool_schema_service_called_without_goal_pursuit(self):
        """get_skill_schemas is called with a list that excludes 'goal_pursuit'."""
        processor = GoalPursuitProcessor()
        processor.send = MagicMock(return_value={'response': 'ok', 'tool_calls': None, 'actions': None})

        skill_names = frozenset({'memory', 'goal_pursuit', 'schedule'})

        with patch('services.innate_skills.registry.ALL_SKILL_NAMES', skill_names), \
             patch('services.tool_schema_service.get_skill_schemas', return_value=[]) as mock_schemas:
            processor.process('my goal', 'xid')

        call_args, _ = mock_schemas.call_args
        names_passed = call_args[0]
        assert 'goal_pursuit' not in names_passed
        assert 'memory' in names_passed
        assert 'schedule' in names_passed


# ── System prompt propagation ──────────────────────────────────────────────────

class TestGoalPursuitSystemPrompt:
    def test_real_system_prompt_passed_to_send(self):
        """The loaded system prompt must be forwarded as the system_prompt arg."""
        processor = GoalPursuitProcessor()
        captured = {}

        def _fake_send(user_prompt, system_prompt, **kwargs):
            captured['system_prompt'] = system_prompt
            return {'response': 'ok', 'tool_calls': None, 'actions': None}

        processor.send = _fake_send

        with patch('services.tool_schema_service.get_skill_schemas', return_value=[]), \
             patch('services.innate_skills.registry.ALL_SKILL_NAMES', frozenset({'memory'})):
            processor.process('my goal', 'pid3')

        # Must match what _load_system_prompt returns
        expected = _load_system_prompt()
        assert captured['system_prompt'] == expected

    def test_fallback_prompt_used_and_forwarded(self):
        """When the prompt file is missing, the fallback is still forwarded to send()."""
        processor = GoalPursuitProcessor()
        captured = {}

        def _fake_send(user_prompt, system_prompt, **kwargs):
            captured['system_prompt'] = system_prompt
            return {'response': 'ok', 'tool_calls': None, 'actions': None}

        processor.send = _fake_send

        with patch('builtins.open', side_effect=FileNotFoundError('missing')), \
             patch('services.tool_schema_service.get_skill_schemas', return_value=[]), \
             patch('services.innate_skills.registry.ALL_SKILL_NAMES', frozenset({'memory'})):
            processor.process('my goal', 'pid4')

        assert captured['system_prompt']
        assert len(captured['system_prompt']) > 0


# ── process() return value ─────────────────────────────────────────────────────

class TestGoalPursuitProcessReturn:
    def test_process_returns_send_result(self):
        """process() must return whatever self.send() returns."""
        processor = GoalPursuitProcessor()
        expected = {'response': 'I did the thing.', 'tool_calls': None, 'actions': None}
        processor.send = MagicMock(return_value=expected)

        with patch('services.tool_schema_service.get_skill_schemas', return_value=[]), \
             patch('services.innate_skills.registry.ALL_SKILL_NAMES', frozenset({'memory'})):
            result = processor.process('do the thing', 'someid')

        assert result is expected

    def test_process_calls_send_exactly_once(self):
        """process() must invoke self.send() exactly once per call."""
        processor = GoalPursuitProcessor()
        processor.send = MagicMock(return_value={'response': 'done', 'tool_calls': None, 'actions': None})

        with patch('services.tool_schema_service.get_skill_schemas', return_value=[]), \
             patch('services.innate_skills.registry.ALL_SKILL_NAMES', frozenset({'memory'})):
            processor.process('task', 'id1')

        processor.send.assert_called_once()
