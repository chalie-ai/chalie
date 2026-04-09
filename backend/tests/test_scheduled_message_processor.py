"""Tests for ScheduledMessageProcessor."""

import pytest
from unittest.mock import MagicMock, patch

from services.scheduled_message_processor import (
    ScheduledMessageProcessor, _load_system_prompt, _EXCLUDED_SKILLS,
)

pytestmark = pytest.mark.unit


class TestProcess:
    """Core delegation contract — process() builds prompts and calls send()."""

    def _run(self, message='Check tasks', item_id='item1', skill_names=None):
        """Helper: run process() with mocked dependencies, return captured send() args."""
        processor = ScheduledMessageProcessor()
        captured = {}

        def _fake_send(user_prompt, system_prompt, **kwargs):
            captured.update(user_prompt=user_prompt, system_prompt=system_prompt, **kwargs)
            return {'response': 'done', 'tool_calls': None, 'actions': None}

        processor.send = _fake_send

        names = skill_names or frozenset({'memory', 'find_tools'})
        mock_upa = MagicMock()
        mock_upa.return_value.build.return_value.to_provider.return_value = 'assembled-prompt'

        with patch('services.tool_schema_service.get_skill_schemas', return_value=[]) as mock_schemas, \
             patch('services.innate_skills.registry.ALL_SKILL_NAMES', names), \
             patch('services.user_prompt_assembly_service.UserPromptAssemblyService', mock_upa):
            result = processor.process(message, item_id)

        return captured, result, mock_schemas, mock_upa

    def test_delegates_to_send(self):
        captured, result, _, _ = self._run()
        assert captured['user_prompt'] == 'assembled-prompt'
        assert captured['channel'] == 'scheduled:item1'
        assert captured['job'] == 'frontal-cortex-unified'
        assert result['response'] == 'done'

    def test_user_prompt_assembly_called_with_channel(self):
        _, _, _, mock_upa = self._run(item_id='abc')
        call_kwargs = mock_upa.return_value.build.call_args
        assert call_kwargs.kwargs['channel'] == 'scheduled:abc'

    def test_excludes_schedule_and_goal_pursuit(self):
        names = frozenset({'memory', 'schedule', 'goal_pursuit', 'goals'})
        _, _, mock_schemas, _ = self._run(skill_names=names)
        passed = mock_schemas.call_args[0][0]
        assert 'schedule' not in passed
        assert 'goal_pursuit' not in passed
        assert 'memory' in passed
        assert 'goals' in passed

    def test_excluded_skills_constant(self):
        assert _EXCLUDED_SKILLS == frozenset({'schedule', 'goal_pursuit'})


class TestLoadSystemPrompt:

    def test_loads_real_file(self):
        prompt = _load_system_prompt()
        assert prompt and len(prompt) > 0

    def test_fallback_on_missing_file(self):
        with patch('builtins.open', side_effect=FileNotFoundError):
            prompt = _load_system_prompt()
        assert 'Chalie' in prompt
