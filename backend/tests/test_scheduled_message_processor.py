"""Tests for ScheduledMessageProcessor."""

import pytest
from unittest.mock import MagicMock, patch

from services.scheduled_message_processor import (
    ScheduledMessageProcessor, _load_system_prompt, _EXCLUDED_SKILLS,
)

pytestmark = pytest.mark.unit


class TestProcess:
    """Core contract: process() assembles prompt, filters tools, delegates to send()."""

    def _run(self, message='Check tasks', item_id='item1', skill_names=None):
        processor = ScheduledMessageProcessor()
        captured = {}

        def _fake_send(user_prompt, system_prompt, **kwargs):
            captured.update(user_prompt=user_prompt, system_prompt=system_prompt, **kwargs)
            return {'response': 'done', 'tool_calls': None, 'actions': None}

        processor.send = _fake_send
        names = skill_names or frozenset({'memory', 'find_tools'})

        with patch.object(processor, '_assemble_user_prompt', return_value='assembled') as mock_asm, \
             patch('services.tool_schema_service.get_skill_schemas', return_value=[]) as mock_schemas, \
             patch('services.innate_skills.registry.ALL_SKILL_NAMES', names):
            result = processor.process(message, item_id)

        return captured, result, mock_schemas

    def test_send_receives_assembled_prompt_and_channel(self):
        captured, _, _ = self._run()
        assert captured['user_prompt'] == 'assembled'
        assert captured['channel'] == 'scheduled:item1'
        assert captured['job'] == 'frontal-cortex-unified'

    def test_excludes_recursive_skills(self):
        names = frozenset({'memory', 'schedule', 'goal_pursuit', 'goals'})
        _, _, mock_schemas = self._run(skill_names=names)
        passed = set(mock_schemas.call_args[0][0])
        assert passed == {'memory', 'goals'}

    def test_excluded_skills_constant(self):
        assert _EXCLUDED_SKILLS == frozenset({'schedule', 'goal_pursuit'})


class TestLoadSystemPrompt:

    def test_loads_real_file(self):
        assert len(_load_system_prompt()) > 0

    def test_fallback_on_missing_file(self):
        with patch('builtins.open', side_effect=FileNotFoundError):
            assert 'Chalie' in _load_system_prompt()
