# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Unit tests for PromptAssemblyService and _ONBOARDING_SCHEDULE.

Covers:
- :data:`~services.prompt_assembly_service._ONBOARDING_SCHEDULE` structure validation
- :class:`~services.prompt_assembly_service.PromptAssemblyService` constructor
- :meth:`~services.prompt_assembly_service.PromptAssemblyService.build_system_prompt`
  delegating to ``_inject_parameters`` with ``act_history=""``
- :meth:`~services.prompt_assembly_service.PromptAssemblyService._inject_parameters`
  placeholder substitution for key template variables

All external services (DB, embedding, LLM, world state) are either mocked or rely on
graceful try/except fallbacks built into the service.
"""

import pytest
from unittest.mock import MagicMock, patch, call

from services.prompt_assembly_service import PromptAssemblyService, _ONBOARDING_SCHEDULE

pytestmark = pytest.mark.unit


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def assembly_svc():
    """PromptAssemblyService with WorldStateService mocked.

    All sub-services that make DB or network calls are either mocked here or
    rely on their own try/except fallbacks (returning '' for each placeholder).

    Returns:
        PromptAssemblyService: Service instance ready for placeholder tests.
    """
    with patch('services.world_state_service.WorldStateService') as mock_ws_cls:
        mock_ws_instance = MagicMock()
        mock_ws_instance.get_world_state.return_value = ''
        mock_ws_cls.return_value = mock_ws_instance
        svc = PromptAssemblyService({'platform': 'test', 'append_mode': False})
    # Override the stored instance so individual tests can control world state output
    svc.world_state_service = MagicMock()
    svc.world_state_service.get_world_state.return_value = ''
    return svc


def _minimal_classification():
    """Return a minimal classification dict for testing.

    Returns:
        dict: Classification with only 'topic' and 'confidence'.
    """
    return {'topic': 'coding', 'confidence': 10}


# ─────────────────────────────────────────────────────────────────────────────
# TestOnboardingScheduleCanonical
# ─────────────────────────────────────────────────────────────────────────────


class TestOnboardingScheduleCanonical:
    """Validates the canonical :data:`_ONBOARDING_SCHEDULE` structure.

    These tests import from the canonical module, not the facade, ensuring
    the source-of-truth data structure is well-formed.
    """

    def test_schedule_is_a_non_empty_list(self):
        """_ONBOARDING_SCHEDULE must be a list with at least one entry."""
        assert isinstance(_ONBOARDING_SCHEDULE, list)
        assert len(_ONBOARDING_SCHEDULE) > 0

    def test_each_entry_has_all_required_keys(self):
        """Every schedule entry must have trait, min_turn, cooldown_turns, max_attempts, hint."""
        required_keys = {'trait', 'min_turn', 'cooldown_turns', 'max_attempts', 'hint'}
        for entry in _ONBOARDING_SCHEDULE:
            missing = required_keys - set(entry.keys())
            assert not missing, (
                f"Entry for '{entry.get('trait', '?')}' is missing keys: {missing}"
            )

    def test_min_turn_values_are_strictly_ascending(self):
        """Traits must be scheduled in ascending min_turn order."""
        min_turns = [entry['min_turn'] for entry in _ONBOARDING_SCHEDULE]
        assert min_turns == sorted(min_turns), (
            f"min_turn values are not ascending: {min_turns}"
        )

    def test_name_trait_is_first(self):
        """'name' must be the first trait elicited (lowest min_turn)."""
        assert _ONBOARDING_SCHEDULE[0]['trait'] == 'name'

    def test_all_hints_are_non_empty_strings(self):
        """Every schedule entry must have a non-empty hint string."""
        for entry in _ONBOARDING_SCHEDULE:
            assert isinstance(entry['hint'], str)
            assert len(entry['hint'].strip()) > 0, (
                f"Hint for trait '{entry['trait']}' is empty"
            )

    def test_all_max_attempts_are_positive(self):
        """max_attempts must be a positive integer for every entry."""
        for entry in _ONBOARDING_SCHEDULE:
            assert isinstance(entry['max_attempts'], int)
            assert entry['max_attempts'] >= 1, (
                f"max_attempts for '{entry['trait']}' must be >= 1"
            )

    def test_all_cooldown_turns_are_positive(self):
        """cooldown_turns must be positive for every entry."""
        for entry in _ONBOARDING_SCHEDULE:
            assert entry['cooldown_turns'] > 0, (
                f"cooldown_turns for '{entry['trait']}' must be > 0"
            )

    def test_all_trait_names_are_non_empty_strings(self):
        """Every trait name must be a non-empty string."""
        for entry in _ONBOARDING_SCHEDULE:
            assert isinstance(entry['trait'], str)
            assert len(entry['trait'].strip()) > 0


# ─────────────────────────────────────────────────────────────────────────────
# TestPromptAssemblyServiceInit
# ─────────────────────────────────────────────────────────────────────────────


class TestPromptAssemblyServiceInit:
    """Tests for :class:`PromptAssemblyService` constructor behaviour."""

    def test_init_stores_config(self, assembly_svc):
        """Constructor stores the provided config dict on ``self.config``."""
        assert assembly_svc.config == {'platform': 'test', 'append_mode': False}

    def test_init_creates_world_state_service_attribute(self, assembly_svc):
        """Constructor attaches a world_state_service attribute."""
        assert hasattr(assembly_svc, 'world_state_service')
        assert assembly_svc.world_state_service is not None

    def test_init_with_empty_config_does_not_raise(self):
        """An empty config dict is accepted without raising."""
        with patch('services.world_state_service.WorldStateService'):
            svc = PromptAssemblyService({})
        assert svc.config == {}

    def test_init_with_append_mode_true(self):
        """append_mode=True config is stored and accessible."""
        with patch('services.world_state_service.WorldStateService'):
            svc = PromptAssemblyService({'append_mode': True})
        assert svc.config['append_mode'] is True


# ─────────────────────────────────────────────────────────────────────────────
# TestBuildSystemPrompt
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildSystemPrompt:
    """Tests for :meth:`PromptAssemblyService.build_system_prompt`.

    Verifies that the method correctly delegates to ``_inject_parameters``
    with ``act_history=""`` (act_history travels in the message array, not
    the system prompt, to allow provider-side caching).
    """

    def test_returns_string(self, assembly_svc):
        """build_system_prompt always returns a string."""
        result = assembly_svc.build_system_prompt(
            system_prompt_template='Hello world',
            original_prompt='test',
            classification=_minimal_classification(),
            chat_history=[],
        )
        assert isinstance(result, str)

    def test_calls_inject_parameters_with_act_history_empty(self, assembly_svc):
        """build_system_prompt calls _inject_parameters with act_history=''."""
        with patch.object(assembly_svc, '_inject_parameters', wraps=assembly_svc._inject_parameters) as mock_inject:
            assembly_svc.build_system_prompt(
                system_prompt_template='template',
                original_prompt='hello',
                classification=_minimal_classification(),
                chat_history=[],
            )
        mock_inject.assert_called_once()
        call_kwargs = mock_inject.call_args.kwargs
        assert call_kwargs.get('act_history') == ''

    def test_original_prompt_placeholder_is_replaced(self, assembly_svc):
        """{{original_prompt}} is replaced with the original_prompt argument."""
        result = assembly_svc.build_system_prompt(
            system_prompt_template='User asked: {{original_prompt}}',
            original_prompt='What is Python?',
            classification=_minimal_classification(),
            chat_history=[],
        )
        assert 'What is Python?' in result
        assert '{{original_prompt}}' not in result

    def test_topic_placeholder_is_replaced(self, assembly_svc):
        """{{topic}} is replaced with the topic from the classification dict."""
        result = assembly_svc.build_system_prompt(
            system_prompt_template='Topic: {{topic}}',
            original_prompt='test',
            classification={'topic': 'data_science', 'confidence': 10},
            chat_history=[],
        )
        assert 'data_science' in result
        assert '{{topic}}' not in result

    def test_confidence_placeholder_is_replaced(self, assembly_svc):
        """{{confidence}} is replaced with the confidence value from classification."""
        result = assembly_svc.build_system_prompt(
            system_prompt_template='Confidence: {{confidence}}',
            original_prompt='test',
            classification={'topic': 'test', 'confidence': 42},
            chat_history=[],
        )
        assert '42' in result
        assert '{{confidence}}' not in result

    def test_act_history_placeholder_is_empty(self, assembly_svc):
        """{{act_history}} is replaced with an empty string in system prompt mode."""
        result = assembly_svc.build_system_prompt(
            system_prompt_template='History: {{act_history}}',
            original_prompt='test',
            classification=_minimal_classification(),
            chat_history=[],
        )
        assert '{{act_history}}' not in result
        assert result.strip() == 'History:'

    def test_similar_topic_takes_precedence_over_topic(self, assembly_svc):
        """When similar_topic is set, it takes precedence over topic for {{topic}} placeholder."""
        result = assembly_svc.build_system_prompt(
            system_prompt_template='{{topic}}',
            original_prompt='test',
            classification={
                'topic': 'original',
                'similar_topic': 'refined_topic',
                'confidence': 10,
            },
            chat_history=[],
        )
        assert 'refined_topic' in result

    def test_topic_update_is_used_when_similar_topic_absent(self, assembly_svc):
        """When topic_update is set and similar_topic is absent, topic_update fills {{topic}}."""
        result = assembly_svc.build_system_prompt(
            system_prompt_template='{{topic}}',
            original_prompt='test',
            classification={
                'topic': 'original',
                'topic_update': 'updated_topic',
                'confidence': 10,
            },
            chat_history=[],
        )
        assert 'updated_topic' in result

    def test_template_with_no_placeholders_returned_unchanged(self, assembly_svc):
        """A template containing no {{...}} placeholders is returned as-is."""
        template = 'You are a helpful assistant.'
        result = assembly_svc.build_system_prompt(
            system_prompt_template=template,
            original_prompt='test',
            classification=_minimal_classification(),
            chat_history=[],
        )
        assert result == template

    def test_inclusion_map_false_excludes_world_state(self, assembly_svc):
        """When inclusion_map disables world_state, {{world_state}} is replaced with ''."""
        assembly_svc.world_state_service.get_world_state.return_value = 'ACTIVE: buy groceries'
        result = assembly_svc.build_system_prompt(
            system_prompt_template='State: {{world_state}}',
            original_prompt='test',
            classification=_minimal_classification(),
            chat_history=[],
            inclusion_map={'world_state': False},
        )
        # world_state disabled → placeholder replaced with ''
        assert 'ACTIVE: buy groceries' not in result
        assert '{{world_state}}' not in result

    def test_inclusion_map_true_allows_world_state(self, assembly_svc):
        """When inclusion_map enables world_state, the service result is injected."""
        assembly_svc.world_state_service.get_world_state.return_value = 'ACTIVE: buy groceries'
        result = assembly_svc.build_system_prompt(
            system_prompt_template='State: {{world_state}}',
            original_prompt='test',
            classification=_minimal_classification(),
            chat_history=[],
            inclusion_map={'world_state': True},
        )
        assert 'ACTIVE: buy groceries' in result

    def test_passes_thread_id_to_inject_parameters(self, assembly_svc):
        """thread_id is forwarded to _inject_parameters for onboarding and focus lookup."""
        with patch.object(assembly_svc, '_inject_parameters', wraps=assembly_svc._inject_parameters) as mock_inject:
            assembly_svc.build_system_prompt(
                system_prompt_template='template',
                original_prompt='hello',
                classification=_minimal_classification(),
                chat_history=[],
                thread_id='thread-abc',
            )
        call_kwargs = mock_inject.call_args.kwargs
        assert call_kwargs.get('thread_id') == 'thread-abc'


# ─────────────────────────────────────────────────────────────────────────────
# TestInjectParameters
# ─────────────────────────────────────────────────────────────────────────────


class TestInjectParameters:
    """Tests for :meth:`PromptAssemblyService._inject_parameters` placeholder substitution.

    All assertions use templates containing exactly the placeholder under test, so
    the result cleanly demonstrates replacement without interference.
    """

    def test_original_prompt_replaced(self, assembly_svc):
        """{{original_prompt}} is substituted with the literal prompt text."""
        result = assembly_svc._inject_parameters(
            '{{original_prompt}}',
            original_prompt='hello world',
            classification=_minimal_classification(),
            chat_history=[],
        )
        assert 'hello world' in result

    def test_topic_replaced(self, assembly_svc):
        """{{topic}} is substituted with the topic from classification."""
        result = assembly_svc._inject_parameters(
            '{{topic}}',
            original_prompt='test',
            classification={'topic': 'machine_learning', 'confidence': 5},
            chat_history=[],
        )
        assert 'machine_learning' in result

    def test_confidence_replaced(self, assembly_svc):
        """{{confidence}} is substituted with the confidence value."""
        result = assembly_svc._inject_parameters(
            '{{confidence}}',
            original_prompt='test',
            classification={'topic': 'test', 'confidence': 99},
            chat_history=[],
        )
        assert '99' in result

    def test_act_history_replaced_with_given_value(self, assembly_svc):
        """{{act_history}} is substituted with the act_history argument."""
        result = assembly_svc._inject_parameters(
            '{{act_history}}',
            original_prompt='test',
            classification=_minimal_classification(),
            chat_history=[],
            act_history='[recall] Found: some data',
        )
        assert '[recall] Found: some data' in result

    def test_act_history_empty_by_default(self, assembly_svc):
        """{{act_history}} defaults to '' when not provided."""
        result = assembly_svc._inject_parameters(
            'H:{{act_history}}',
            original_prompt='test',
            classification=_minimal_classification(),
            chat_history=[],
        )
        assert result.strip() == 'H:'

    def test_world_state_excluded_by_inclusion_map(self, assembly_svc):
        """When inclusion_map disables world_state, {{world_state}} becomes ''."""
        assembly_svc.world_state_service.get_world_state.return_value = 'important task'
        result = assembly_svc._inject_parameters(
            'W:{{world_state}}',
            original_prompt='test',
            classification=_minimal_classification(),
            chat_history=[],
            inclusion_map={'world_state': False},
        )
        assert 'important task' not in result
        assert '{{world_state}}' not in result

    def test_episodic_memory_excluded_by_inclusion_map(self, assembly_svc):
        """When inclusion_map disables episodic_memory, {{episodic_memory}} becomes ''."""
        assembled_ctx = {'episodes': 'memory content here'}
        result = assembly_svc._inject_parameters(
            'E:{{episodic_memory}}',
            original_prompt='test',
            classification=_minimal_classification(),
            chat_history=[],
            assembled_context=assembled_ctx,
            inclusion_map={'episodic_memory': False},
        )
        assert 'memory content here' not in result

    def test_episodic_memory_included_from_assembled_context(self, assembly_svc):
        """{{episodic_memory}} is populated from assembled_context['episodes']."""
        assembled_ctx = {'episodes': 'user loves Python'}
        result = assembly_svc._inject_parameters(
            '{{episodic_memory}}',
            original_prompt='test',
            classification=_minimal_classification(),
            chat_history=[],
            assembled_context=assembled_ctx,
            inclusion_map={'episodic_memory': True},
        )
        assert 'user loves Python' in result

    def test_legacy_identity_context_placeholder_removed(self, assembly_svc):
        """Legacy {{identity_context}} placeholder is replaced with '' (backward compat)."""
        result = assembly_svc._inject_parameters(
            'X{{identity_context}}Y',
            original_prompt='test',
            classification=_minimal_classification(),
            chat_history=[],
        )
        assert '{{identity_context}}' not in result
        assert 'XY' in result

    def test_legacy_client_context_placeholder_removed(self, assembly_svc):
        """Legacy {{client_context}} placeholder is replaced with ''."""
        result = assembly_svc._inject_parameters(
            'A{{client_context}}B',
            original_prompt='test',
            classification=_minimal_classification(),
            chat_history=[],
        )
        assert '{{client_context}}' not in result
        assert 'AB' in result

    def test_legacy_available_skills_placeholder_removed(self, assembly_svc):
        """Legacy {{available_skills}} placeholder is replaced with ''."""
        result = assembly_svc._inject_parameters(
            'S:{{available_skills}}',
            original_prompt='test',
            classification=_minimal_classification(),
            chat_history=[],
        )
        assert '{{available_skills}}' not in result

    def test_visual_context_empty_when_not_in_assembled_context(self, assembly_svc):
        """{{visual_context}} is replaced with '' when assembled_context has no visual_context."""
        result = assembly_svc._inject_parameters(
            'V:{{visual_context}}',
            original_prompt='test',
            classification=_minimal_classification(),
            chat_history=[],
            assembled_context={},
        )
        assert '{{visual_context}}' not in result

    def test_contradiction_context_empty_when_absent(self, assembly_svc):
        """{{contradiction_context}} is replaced with '' when assembled_context has no contradiction."""
        result = assembly_svc._inject_parameters(
            'C:{{contradiction_context}}',
            original_prompt='test',
            classification=_minimal_classification(),
            chat_history=[],
            assembled_context={},
        )
        assert '{{contradiction_context}}' not in result
