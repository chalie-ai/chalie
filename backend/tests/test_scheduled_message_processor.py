"""Tests for ScheduledMessageProcessor.

Covers: process() delegation contract, system prompt loading/fallback,
channel formation, tool exclusion (schedule + goal_pursuit), class constants,
UserPromptAssemblyService context injection, and return value propagation.
"""

import pytest
from unittest.mock import MagicMock, patch

from services.scheduled_message_processor import (
    ScheduledMessageProcessor, _load_system_prompt, _EXCLUDED_SKILLS,
)

pytestmark = pytest.mark.unit


def _patch_user_prompt_assembly(passthrough=True):
    """Return a patch context manager for UserPromptAssemblyService.

    When passthrough=True, .build().to_provider() returns the user_message
    argument unchanged (so existing content assertions still work).

    Patches at the source module since process() uses a local import.
    Returns (patch_ctx, capture_dict) — capture_dict holds the last build() kwargs.
    """
    mock_cls = MagicMock()
    capture = {}

    def _build(user_message, channel, thread_id=None, metadata=None):
        capture['user_message'] = user_message
        capture['channel'] = channel
        capture['thread_id'] = thread_id
        builder = MagicMock()
        builder.to_provider.return_value = user_message if passthrough else 'assembled-prompt'
        return builder

    mock_cls.return_value.build = _build
    return patch(
        'services.user_prompt_assembly_service.UserPromptAssemblyService',
        mock_cls,
    ), capture


# ── Class-level constants ──────────────────────────────────────────────────────

class TestScheduledMessageProcessorConstants:
    def test_max_iterations_is_30(self):
        """MAX_ITERATIONS must match the standard tool loop cap."""
        assert ScheduledMessageProcessor.MAX_ITERATIONS == 30

    def test_max_timeout_is_900(self):
        """MAX_TIMEOUT must be 900 seconds (15 minutes) — same as the base limit."""
        assert ScheduledMessageProcessor.MAX_TIMEOUT == 900

    def test_max_iterations_matches_base(self):
        """ScheduledMessageProcessor inherits base limits — not elevated like GoalPursuit."""
        from services.message_processor import MAX_ITERATIONS as BASE_MAX
        assert ScheduledMessageProcessor.MAX_ITERATIONS == BASE_MAX

    def test_max_timeout_matches_base(self):
        """Timeout is identical to the base MessageProcessor default."""
        from services.message_processor import MAX_TIMEOUT as BASE_MAX
        assert ScheduledMessageProcessor.MAX_TIMEOUT == BASE_MAX


# ── System prompt loading ──────────────────────────────────────────────────────

class TestLoadSystemPrompt:
    def test_loads_from_scheduled_prompt_md(self):
        """_load_system_prompt reads scheduled-prompt.md from disk successfully."""
        prompt = _load_system_prompt()
        assert prompt
        assert len(prompt) > 0

    def test_prompt_is_stripped(self):
        """Leading/trailing whitespace is removed from the loaded prompt."""
        prompt = _load_system_prompt()
        assert prompt == prompt.strip()

    def test_real_prompt_references_scheduled_task(self):
        """The real prompt file gives scheduled-task execution context."""
        prompt = _load_system_prompt()
        lower = prompt.lower()
        assert 'scheduled' in lower or 'task' in lower or 'chalie' in lower

    def test_fallback_returned_on_file_not_found(self):
        """When scheduled-prompt.md is missing, a safe fallback string is returned."""
        with patch('builtins.open', side_effect=FileNotFoundError('no such file')):
            prompt = _load_system_prompt()

        assert prompt
        assert len(prompt) > 0
        assert 'Chalie' in prompt

    def test_fallback_returned_on_permission_error(self):
        """When the prompt file raises PermissionError, the fallback is still returned."""
        with patch('builtins.open', side_effect=PermissionError('denied')):
            prompt = _load_system_prompt()

        assert prompt
        assert len(prompt) > 0

    def test_fallback_is_action_oriented(self):
        """The fallback prompt must be a meaningful execution instruction."""
        with patch('builtins.open', side_effect=OSError('some io error')):
            prompt = _load_system_prompt()

        lower = prompt.lower()
        assert 'scheduled' in lower or 'task' in lower or 'execute' in lower


# ── Channel formation ──────────────────────────────────────────────────────────

class TestChannelFormation:
    def _make_processor_and_capture_send(self):
        """Return a processor with send() patched to capture call arguments."""
        processor = ScheduledMessageProcessor()
        captured = {}

        def _fake_send(user_prompt, system_prompt, **kwargs):
            captured['user_prompt'] = user_prompt
            captured['system_prompt'] = system_prompt
            captured['kwargs'] = kwargs
            return {'response': 'done', 'tool_calls': None, 'actions': None}

        processor.send = _fake_send
        return processor, captured

    def test_default_channel_is_scheduled_prefix_plus_item_id(self):
        """With no channel override, channel must be 'scheduled:<item_id>'."""
        processor, captured = self._make_processor_and_capture_send()
        upa_patch, _ = _patch_user_prompt_assembly()

        with patch('services.tool_schema_service.get_skill_schemas', return_value=[]), \
             patch('services.innate_skills.registry.ALL_SKILL_NAMES', frozenset(['memory'])), \
             upa_patch:
            processor.process('Do a thing', 'item42')

        assert captured['kwargs']['channel'] == 'scheduled:item42'

    def test_default_channel_uses_exact_item_id(self):
        """The item_id segment in the default channel must be the exact string provided."""
        processor, captured = self._make_processor_and_capture_send()
        item_id = 'deadbeef'
        upa_patch, _ = _patch_user_prompt_assembly()

        with patch('services.tool_schema_service.get_skill_schemas', return_value=[]), \
             patch('services.innate_skills.registry.ALL_SKILL_NAMES', frozenset(['memory'])), \
             upa_patch:
            processor.process('Do a thing', item_id)

        assert captured['kwargs']['channel'] == f'scheduled:{item_id}'

    def test_channel_override_is_used_when_provided(self):
        """An explicit channel argument must override the default scheduled:<id> pattern."""
        processor, captured = self._make_processor_and_capture_send()
        upa_patch, _ = _patch_user_prompt_assembly()

        with patch('services.tool_schema_service.get_skill_schemas', return_value=[]), \
             patch('services.innate_skills.registry.ALL_SKILL_NAMES', frozenset(['memory'])), \
             upa_patch:
            processor.process('Do a thing', 'item1', channel='my-custom-channel')

        assert captured['kwargs']['channel'] == 'my-custom-channel'

    def test_none_channel_falls_back_to_default(self):
        """Passing channel=None explicitly must still produce the scheduled:<id> default."""
        processor, captured = self._make_processor_and_capture_send()
        upa_patch, _ = _patch_user_prompt_assembly()

        with patch('services.tool_schema_service.get_skill_schemas', return_value=[]), \
             patch('services.innate_skills.registry.ALL_SKILL_NAMES', frozenset(['memory'])), \
             upa_patch:
            processor.process('Do a thing', 'item99', channel=None)

        assert captured['kwargs']['channel'] == 'scheduled:item99'

    def test_job_is_frontal_cortex_unified(self):
        """Job must be 'frontal-cortex-unified' so the correct LLM config is resolved."""
        processor, captured = self._make_processor_and_capture_send()
        upa_patch, _ = _patch_user_prompt_assembly()

        with patch('services.tool_schema_service.get_skill_schemas', return_value=[]), \
             patch('services.innate_skills.registry.ALL_SKILL_NAMES', frozenset(['memory'])), \
             upa_patch:
            processor.process('Do a thing', 'item1')

        assert captured['kwargs']['job'] == 'frontal-cortex-unified'


# ── User prompt framing ────────────────────────────────────────────────────────

class TestUserPromptFraming:
    def _capture_send(self, processor):
        captured = {}

        def _fake_send(user_prompt, system_prompt, **kwargs):
            captured['user_prompt'] = user_prompt
            return {'response': 'ok', 'tool_calls': None, 'actions': None}

        processor.send = _fake_send
        return captured

    def test_message_is_included_in_user_prompt(self):
        """The original message must appear within the user_prompt sent to send().

        With passthrough mock, UserPromptAssemblyService returns the raw instruction
        which contains the original message.
        """
        processor = ScheduledMessageProcessor()
        captured = self._capture_send(processor)
        message = 'Check on my open tasks'
        upa_patch, _ = _patch_user_prompt_assembly(passthrough=True)

        with patch('services.tool_schema_service.get_skill_schemas', return_value=[]), \
             patch('services.innate_skills.registry.ALL_SKILL_NAMES', frozenset(['memory'])), \
             upa_patch:
            processor.process(message, 'itemA')

        assert message in captured['user_prompt']

    def test_user_prompt_contains_scheduled_task_header(self):
        """The framing header '[SCHEDULED TASK' must be injected into user_prompt."""
        processor = ScheduledMessageProcessor()
        captured = self._capture_send(processor)
        upa_patch, _ = _patch_user_prompt_assembly(passthrough=True)

        with patch('services.tool_schema_service.get_skill_schemas', return_value=[]), \
             patch('services.innate_skills.registry.ALL_SKILL_NAMES', frozenset(['memory'])), \
             upa_patch:
            processor.process('Any message', 'itemB')

        assert '[SCHEDULED TASK' in captured['user_prompt']

    def test_empty_message_handled_without_error(self):
        """process() must not raise when given an empty message string."""
        processor = ScheduledMessageProcessor()
        processor.send = MagicMock(return_value={'response': '', 'tool_calls': None, 'actions': None})
        upa_patch, _ = _patch_user_prompt_assembly()

        with patch('services.tool_schema_service.get_skill_schemas', return_value=[]), \
             patch('services.innate_skills.registry.ALL_SKILL_NAMES', frozenset(['memory'])), \
             upa_patch:
            # Must not raise
            result = processor.process('', 'emptyitem')

        assert result is not None


# ── Tool exclusion — no recursive scheduling ──────────────────────────────────

class TestToolExclusion:
    def test_excluded_skills_constant(self):
        """_EXCLUDED_SKILLS must contain both 'schedule' and 'goal_pursuit'."""
        assert 'schedule' in _EXCLUDED_SKILLS
        assert 'goal_pursuit' in _EXCLUDED_SKILLS

    def test_schedule_skill_excluded_from_tools(self):
        """'schedule' must NOT appear in the tools list passed to send()."""
        processor = ScheduledMessageProcessor()
        captured_tools = {}

        def _fake_send(user_prompt, system_prompt, **kwargs):
            captured_tools['tools'] = kwargs.get('tools', [])
            return {'response': 'ok', 'tool_calls': None, 'actions': None}

        processor.send = _fake_send
        upa_patch, _ = _patch_user_prompt_assembly()

        skill_names = frozenset({'memory', 'schedule', 'find_tools', 'goals'})
        fake_schemas = [
            {'name': 'memory', 'description': '', 'input_schema': {}},
            {'name': 'find_tools', 'description': '', 'input_schema': {}},
            {'name': 'goals', 'description': '', 'input_schema': {}},
        ]

        with patch('services.innate_skills.registry.ALL_SKILL_NAMES', skill_names), \
             patch('services.tool_schema_service.get_skill_schemas', return_value=fake_schemas), \
             upa_patch:
            processor.process('Check my tasks', 'item1')

        tool_names = [t['name'] for t in captured_tools['tools']]
        assert 'schedule' not in tool_names

    def test_goal_pursuit_skill_excluded_from_tools(self):
        """'goal_pursuit' must NOT be passed to get_skill_schemas."""
        processor = ScheduledMessageProcessor()
        processor.send = MagicMock(return_value={'response': 'ok', 'tool_calls': None, 'actions': None})
        upa_patch, _ = _patch_user_prompt_assembly()

        skill_names = frozenset({'memory', 'schedule', 'goal_pursuit', 'goals'})

        with patch('services.innate_skills.registry.ALL_SKILL_NAMES', skill_names), \
             patch('services.tool_schema_service.get_skill_schemas', return_value=[]) as mock_schemas, \
             upa_patch:
            processor.process('my task', 'xid2')

        call_args, _ = mock_schemas.call_args
        names_passed = call_args[0]
        assert 'goal_pursuit' not in names_passed
        assert 'schedule' not in names_passed
        assert 'memory' in names_passed
        assert 'goals' in names_passed

    def test_other_skills_are_still_included(self):
        """Skills other than excluded ones must still be provided to the tool loop."""
        processor = ScheduledMessageProcessor()
        captured_tools = {}

        def _fake_send(user_prompt, system_prompt, **kwargs):
            captured_tools['tools'] = kwargs.get('tools', [])
            return {'response': 'ok', 'tool_calls': None, 'actions': None}

        processor.send = _fake_send
        upa_patch, _ = _patch_user_prompt_assembly()

        skill_names = frozenset({'memory', 'schedule', 'find_tools'})
        fake_schemas = [
            {'name': 'memory', 'description': '', 'input_schema': {}},
            {'name': 'find_tools', 'description': '', 'input_schema': {}},
        ]

        with patch('services.innate_skills.registry.ALL_SKILL_NAMES', skill_names), \
             patch('services.tool_schema_service.get_skill_schemas', return_value=fake_schemas), \
             upa_patch:
            processor.process('Do a task', 'item2')

        tool_names = [t['name'] for t in captured_tools['tools']]
        assert 'memory' in tool_names
        assert 'find_tools' in tool_names

    def test_get_skill_schemas_called_without_excluded(self):
        """get_skill_schemas must be called with a list that excludes all _EXCLUDED_SKILLS."""
        processor = ScheduledMessageProcessor()
        processor.send = MagicMock(return_value={'response': 'ok', 'tool_calls': None, 'actions': None})
        upa_patch, _ = _patch_user_prompt_assembly()

        skill_names = frozenset({'memory', 'schedule', 'goals', 'goal_pursuit'})

        with patch('services.innate_skills.registry.ALL_SKILL_NAMES', skill_names), \
             patch('services.tool_schema_service.get_skill_schemas', return_value=[]) as mock_schemas, \
             upa_patch:
            processor.process('my task', 'xid')

        call_args, _ = mock_schemas.call_args
        names_passed = call_args[0]
        assert 'schedule' not in names_passed
        assert 'goal_pursuit' not in names_passed
        assert 'memory' in names_passed
        assert 'goals' in names_passed


# ── System prompt propagation ──────────────────────────────────────────────────

class TestSystemPromptPropagation:
    def test_real_system_prompt_forwarded_to_send(self):
        """The loaded system prompt must be passed as the system_prompt arg of send()."""
        processor = ScheduledMessageProcessor()
        captured = {}

        def _fake_send(user_prompt, system_prompt, **kwargs):
            captured['system_prompt'] = system_prompt
            return {'response': 'ok', 'tool_calls': None, 'actions': None}

        processor.send = _fake_send
        upa_patch, _ = _patch_user_prompt_assembly()

        with patch('services.tool_schema_service.get_skill_schemas', return_value=[]), \
             patch('services.innate_skills.registry.ALL_SKILL_NAMES', frozenset({'memory'})), \
             upa_patch:
            processor.process('my task', 'pid3')

        expected = _load_system_prompt()
        assert captured['system_prompt'] == expected

    def test_fallback_prompt_forwarded_when_file_missing(self):
        """When scheduled-prompt.md is missing, the fallback is still forwarded to send()."""
        processor = ScheduledMessageProcessor()
        captured = {}

        def _fake_send(user_prompt, system_prompt, **kwargs):
            captured['system_prompt'] = system_prompt
            return {'response': 'ok', 'tool_calls': None, 'actions': None}

        processor.send = _fake_send
        upa_patch, _ = _patch_user_prompt_assembly()

        with patch('builtins.open', side_effect=FileNotFoundError('missing')), \
             patch('services.tool_schema_service.get_skill_schemas', return_value=[]), \
             patch('services.innate_skills.registry.ALL_SKILL_NAMES', frozenset({'memory'})), \
             upa_patch:
            processor.process('my task', 'pid4')

        assert captured['system_prompt']
        assert len(captured['system_prompt']) > 0


# ── process() → send() delegation contract ────────────────────────────────────

class TestProcessSendDelegation:
    def test_process_returns_send_result(self):
        """process() must return whatever self.send() returns — unmodified."""
        processor = ScheduledMessageProcessor()
        expected = {'response': 'Task complete.', 'tool_calls': None, 'actions': None}
        processor.send = MagicMock(return_value=expected)
        upa_patch, _ = _patch_user_prompt_assembly()

        with patch('services.tool_schema_service.get_skill_schemas', return_value=[]), \
             patch('services.innate_skills.registry.ALL_SKILL_NAMES', frozenset({'memory'})), \
             upa_patch:
            result = processor.process('do the task', 'someid')

        assert result is expected

    def test_process_calls_send_exactly_once(self):
        """process() must invoke self.send() exactly once per call."""
        processor = ScheduledMessageProcessor()
        processor.send = MagicMock(return_value={'response': 'done', 'tool_calls': None, 'actions': None})
        upa_patch, _ = _patch_user_prompt_assembly()

        with patch('services.tool_schema_service.get_skill_schemas', return_value=[]), \
             patch('services.innate_skills.registry.ALL_SKILL_NAMES', frozenset({'memory'})), \
             upa_patch:
            processor.process('task', 'id1')

        processor.send.assert_called_once()

    def test_send_receives_tools_kwarg(self):
        """send() must be called with a 'tools' keyword argument."""
        processor = ScheduledMessageProcessor()
        processor.send = MagicMock(return_value={'response': 'ok', 'tool_calls': None, 'actions': None})
        upa_patch, _ = _patch_user_prompt_assembly()

        with patch('services.tool_schema_service.get_skill_schemas', return_value=[{'name': 'memory'}]), \
             patch('services.innate_skills.registry.ALL_SKILL_NAMES', frozenset({'memory'})), \
             upa_patch:
            processor.process('task', 'id2')

        _, kwargs = processor.send.call_args
        assert 'tools' in kwargs

    def test_send_receives_channel_kwarg(self):
        """send() must receive 'channel' as a keyword argument."""
        processor = ScheduledMessageProcessor()
        processor.send = MagicMock(return_value={'response': 'ok', 'tool_calls': None, 'actions': None})
        upa_patch, _ = _patch_user_prompt_assembly()

        with patch('services.tool_schema_service.get_skill_schemas', return_value=[]), \
             patch('services.innate_skills.registry.ALL_SKILL_NAMES', frozenset({'memory'})), \
             upa_patch:
            processor.process('task', 'id3')

        _, kwargs = processor.send.call_args
        assert 'channel' in kwargs


# ── UserPromptAssemblyService context injection ─────────────────────────────

class TestUserPromptAssemblyInjection:
    def test_user_prompt_assembly_service_is_called(self):
        """process() must invoke UserPromptAssemblyService.build() to inject context."""
        processor = ScheduledMessageProcessor()
        processor.send = MagicMock(return_value={'response': 'ok', 'tool_calls': None, 'actions': None})
        upa_patch, capture = _patch_user_prompt_assembly(passthrough=False)

        with patch('services.tool_schema_service.get_skill_schemas', return_value=[]), \
             patch('services.innate_skills.registry.ALL_SKILL_NAMES', frozenset({'memory'})), \
             upa_patch:
            processor.process('Check calendar', 'ctx1')

        # Verify send() received the assembled prompt (not the raw instruction)
        call_args = processor.send.call_args
        user_prompt_sent = call_args[0][0]
        assert user_prompt_sent == 'assembled-prompt'

    def test_build_receives_channel(self):
        """UserPromptAssemblyService.build() must receive the resolved channel."""
        processor = ScheduledMessageProcessor()
        processor.send = MagicMock(return_value={'response': 'ok', 'tool_calls': None, 'actions': None})
        upa_patch, capture = _patch_user_prompt_assembly()

        with patch('services.tool_schema_service.get_skill_schemas', return_value=[]), \
             patch('services.innate_skills.registry.ALL_SKILL_NAMES', frozenset({'memory'})), \
             upa_patch:
            processor.process('Do thing', 'abc123')

        assert capture['channel'] == 'scheduled:abc123'
        assert capture['thread_id'] == 'scheduled:abc123'

    def test_build_receives_custom_channel(self):
        """When a custom channel is passed, build() receives that channel."""
        processor = ScheduledMessageProcessor()
        processor.send = MagicMock(return_value={'response': 'ok', 'tool_calls': None, 'actions': None})
        upa_patch, capture = _patch_user_prompt_assembly()

        with patch('services.tool_schema_service.get_skill_schemas', return_value=[]), \
             patch('services.innate_skills.registry.ALL_SKILL_NAMES', frozenset({'memory'})), \
             upa_patch:
            processor.process('Do thing', 'abc123', channel='my-channel')

        assert capture['channel'] == 'my-channel'

    def test_raw_instruction_contains_scheduled_header(self):
        """The raw instruction passed to build() must contain the SCHEDULED TASK header."""
        processor = ScheduledMessageProcessor()
        processor.send = MagicMock(return_value={'response': 'ok', 'tool_calls': None, 'actions': None})
        upa_patch, capture = _patch_user_prompt_assembly()

        with patch('services.tool_schema_service.get_skill_schemas', return_value=[]), \
             patch('services.innate_skills.registry.ALL_SKILL_NAMES', frozenset({'memory'})), \
             upa_patch:
            processor.process('Check my calendar', 'hdr1')

        assert '[SCHEDULED TASK' in capture['user_message']
        assert 'Check my calendar' in capture['user_message']
