"""Tests for innate skill registry — verifies authoritative constants are consistent."""

import pytest
from unittest.mock import patch, MagicMock

from services.innate_skills.registry import (
    ALL_SKILL_NAMES,
    PLANNING_SKILLS,
    REFLECTION_FILTER_SKILLS,
    COGNITIVE_PRIMITIVES,
    CONTEXTUAL_SKILLS,
    TRIAGE_VALID_SKILLS,
    PROCEDURAL_MEMORY_SKILLS,
    SKILL_DESCRIPTIONS,
    COGNITIVE_PRIMITIVES_ORDERED,
)
from services.act_action_categories import (
    READ_ACTIONS,
    DETERMINISTIC_ACTIONS,
    SAFE_ACTIONS,
    CRITIC_SKIP_READS,
    ACTION_FATIGUE_COSTS,
)


pytestmark = pytest.mark.unit


class TestSkillRegistry:

    def test_all_skill_names_matches_registered_handlers(self):
        """ALL_SKILL_NAMES must match the handler keys set by register_innate_skills()."""
        mock_dispatcher = MagicMock()
        mock_dispatcher.handlers = {}

        with patch('services.tool_registry_service.ToolRegistryService') as mock_registry_cls:
            mock_registry = mock_registry_cls.return_value
            mock_registry.get_on_demand_tools.return_value = []

            from services.innate_skills import register_innate_skills
            register_innate_skills(mock_dispatcher)

        # Exclude backward-compatibility aliases
        aliases = {'memory_query', 'memory_write', 'world_state_read', 'internal_reasoning', 'semantic_query'}
        registered = set(mock_dispatcher.handlers.keys()) - aliases
        assert registered == ALL_SKILL_NAMES

    def test_cognitive_primitives_subset_of_all(self):
        assert COGNITIVE_PRIMITIVES < ALL_SKILL_NAMES

    def test_planning_skills_excludes_internal(self):
        assert 'emit_card' not in PLANNING_SKILLS
        assert 'moment' not in PLANNING_SKILLS

    def test_contextual_plus_primitives_equals_planning(self):
        assert CONTEXTUAL_SKILLS | COGNITIVE_PRIMITIVES == PLANNING_SKILLS

    def test_safe_actions_subset_of_all(self):
        assert SAFE_ACTIONS <= ALL_SKILL_NAMES

    def test_read_actions_subset_of_all(self):
        assert READ_ACTIONS <= ALL_SKILL_NAMES

    def test_critic_skip_reads_subset_of_read_actions(self):
        assert CRITIC_SKIP_READS <= READ_ACTIONS

    def test_deterministic_actions_subset_of_all(self):
        assert DETERMINISTIC_ACTIONS <= ALL_SKILL_NAMES

    def test_procedural_memory_skills_equals_primitives(self):
        assert PROCEDURAL_MEMORY_SKILLS == COGNITIVE_PRIMITIVES

    def test_skill_descriptions_covers_all_skills(self):
        assert set(SKILL_DESCRIPTIONS.keys()) == ALL_SKILL_NAMES

    def test_ordered_primitives_matches_frozenset(self):
        assert set(COGNITIVE_PRIMITIVES_ORDERED) == COGNITIVE_PRIMITIVES

    def test_all_types_are_frozenset(self):
        """All sets must be frozenset (immutable)."""
        for name, value in [
            ('ALL_SKILL_NAMES', ALL_SKILL_NAMES),
            ('PLANNING_SKILLS', PLANNING_SKILLS),
            ('REFLECTION_FILTER_SKILLS', REFLECTION_FILTER_SKILLS),
            ('COGNITIVE_PRIMITIVES', COGNITIVE_PRIMITIVES),
            ('CONTEXTUAL_SKILLS', CONTEXTUAL_SKILLS),
            ('TRIAGE_VALID_SKILLS', TRIAGE_VALID_SKILLS),
            ('PROCEDURAL_MEMORY_SKILLS', PROCEDURAL_MEMORY_SKILLS),
            ('READ_ACTIONS', READ_ACTIONS),
            ('DETERMINISTIC_ACTIONS', DETERMINISTIC_ACTIONS),
            ('SAFE_ACTIONS', SAFE_ACTIONS),
            ('CRITIC_SKIP_READS', CRITIC_SKIP_READS),
        ]:
            assert isinstance(value, frozenset), f"{name} should be frozenset, got {type(value)}"


class TestActionFatigueCosts:

    def test_all_keys_in_all_skills(self):
        assert set(ACTION_FATIGUE_COSTS.keys()) <= ALL_SKILL_NAMES

    def test_is_immutable(self):
        with pytest.raises(TypeError):
            ACTION_FATIGUE_COSTS['new_key'] = 999
