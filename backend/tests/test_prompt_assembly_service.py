# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for PromptAssemblyService and _ONBOARDING_SCHEDULE."""

import pytest
from unittest.mock import MagicMock, patch

from services.prompt_assembly_service import PromptAssemblyService, _ONBOARDING_SCHEDULE

pytestmark = pytest.mark.unit


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def assembly_svc():
    """PromptAssemblyService with WorldStateService mocked."""
    with patch('services.world_state_service.WorldStateService') as mock_ws_cls:
        mock_ws_instance = MagicMock()
        mock_ws_instance.get_world_state.return_value = ''
        mock_ws_cls.return_value = mock_ws_instance
        svc = PromptAssemblyService({'platform': 'test', 'append_mode': False})
    svc.world_state_service = MagicMock()
    svc.world_state_service.get_world_state.return_value = ''
    return svc


def _minimal_classification():
    """Return a minimal classification dict for testing."""
    return {'topic': 'coding', 'confidence': 10}


# ─────────────────────────────────────────────────────────────────────────────
# TestOnboardingScheduleCanonical
# ─────────────────────────────────────────────────────────────────────────────


class TestOnboardingScheduleCanonical:
    """Validates the canonical _ONBOARDING_SCHEDULE structure."""

    def test_schedule_is_a_non_empty_list(self):
        assert isinstance(_ONBOARDING_SCHEDULE, list)
        assert len(_ONBOARDING_SCHEDULE) > 0

    def test_each_entry_has_all_required_keys(self):
        required_keys = {'trait', 'min_turn', 'cooldown_turns', 'max_attempts', 'hint'}
        for entry in _ONBOARDING_SCHEDULE:
            missing = required_keys - set(entry.keys())
            assert not missing, (
                f"Entry for '{entry.get('trait', '?')}' is missing keys: {missing}"
            )

    def test_min_turn_values_are_strictly_ascending(self):
        min_turns = [entry['min_turn'] for entry in _ONBOARDING_SCHEDULE]
        assert min_turns == sorted(min_turns), (
            f"min_turn values are not ascending: {min_turns}"
        )

    def test_name_trait_is_first(self):
        assert _ONBOARDING_SCHEDULE[0]['trait'] == 'name'


# ─────────────────────────────────────────────────────────────────────────────
# TestBuildSystemPrompt
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildSystemPrompt:
    """Tests for build_system_prompt delegation to _inject_parameters."""

    def test_calls_inject_parameters_with_act_history_empty(self, assembly_svc):
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

    def test_similar_topic_takes_precedence_over_topic(self, assembly_svc):
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
        template = 'You are a helpful assistant.'
        result = assembly_svc.build_system_prompt(
            system_prompt_template=template,
            original_prompt='test',
            classification=_minimal_classification(),
            chat_history=[],
        )
        assert result == template

    def test_inclusion_map_false_excludes_world_state(self, assembly_svc):
        assembly_svc.world_state_service.get_world_state.return_value = 'ACTIVE: buy groceries'
        result = assembly_svc.build_system_prompt(
            system_prompt_template='State: {{world_state}}',
            original_prompt='test',
            classification=_minimal_classification(),
            chat_history=[],
            inclusion_map={'world_state': False},
        )
        assert 'ACTIVE: buy groceries' not in result
        assert '{{world_state}}' not in result

    def test_inclusion_map_true_allows_world_state(self, assembly_svc):
        assembly_svc.world_state_service.get_world_state.return_value = 'ACTIVE: buy groceries'
        result = assembly_svc.build_system_prompt(
            system_prompt_template='State: {{world_state}}',
            original_prompt='test',
            classification=_minimal_classification(),
            chat_history=[],
            inclusion_map={'world_state': True},
        )
        assert 'ACTIVE: buy groceries' in result


# ─────────────────────────────────────────────────────────────────────────────
# TestInjectParameters
# ─────────────────────────────────────────────────────────────────────────────


class TestInjectParameters:
    """Tests for _inject_parameters placeholder substitution."""

    def test_original_prompt_replaced(self, assembly_svc):
        result = assembly_svc._inject_parameters(
            '{{original_prompt}}',
            original_prompt='hello world',
            classification=_minimal_classification(),
            chat_history=[],
        )
        assert 'hello world' in result

    def test_topic_replaced(self, assembly_svc):
        result = assembly_svc._inject_parameters(
            '{{topic}}',
            original_prompt='test',
            classification={'topic': 'machine_learning', 'confidence': 5},
            chat_history=[],
        )
        assert 'machine_learning' in result

    def test_confidence_replaced(self, assembly_svc):
        result = assembly_svc._inject_parameters(
            '{{confidence}}',
            original_prompt='test',
            classification={'topic': 'test', 'confidence': 99},
            chat_history=[],
        )
        assert '99' in result

    def test_act_history_replaced_with_given_value(self, assembly_svc):
        result = assembly_svc._inject_parameters(
            '{{act_history}}',
            original_prompt='test',
            classification=_minimal_classification(),
            chat_history=[],
            act_history='[recall] Found: some data',
        )
        assert '[recall] Found: some data' in result

    def test_act_history_empty_by_default(self, assembly_svc):
        result = assembly_svc._inject_parameters(
            'H:{{act_history}}',
            original_prompt='test',
            classification=_minimal_classification(),
            chat_history=[],
        )
        assert result.strip() == 'H:'

    @pytest.mark.parametrize("placeholder", [
        "identity_context", "client_context", "available_skills",
    ])
    def test_legacy_placeholder_removed(self, assembly_svc, placeholder):
        result = assembly_svc._inject_parameters(
            f'X{{{{{placeholder}}}}}Y',
            original_prompt='test',
            classification=_minimal_classification(),
            chat_history=[],
        )
        assert f'{{{{{placeholder}}}}}' not in result

    def test_visual_context_empty_when_not_in_assembled_context(self, assembly_svc):
        result = assembly_svc._inject_parameters(
            'V:{{visual_context}}',
            original_prompt='test',
            classification=_minimal_classification(),
            chat_history=[],
            assembled_context={},
        )
        assert '{{visual_context}}' not in result

    def test_contradiction_context_empty_when_absent(self, assembly_svc):
        result = assembly_svc._inject_parameters(
            'C:{{contradiction_context}}',
            original_prompt='test',
            classification=_minimal_classification(),
            chat_history=[],
            assembled_context={},
        )
        assert '{{contradiction_context}}' not in result
