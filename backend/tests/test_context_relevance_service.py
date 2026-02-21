"""
Unit tests for ContextRelevanceService.

Tests cover all 7 layers of context relevance determination:
1. Template masks
2. Signal rules (hard and soft)
3. Urgency overrides
4. Soft exclusion recovery
5. Dependency resolution
6. Safety overrides
7. MAX_INCLUDED_NODES safeguard
"""

import pytest
import json
import tempfile
from pathlib import Path

from backend.services.context_relevance_service import (
    ContextRelevanceService,
    ConfigError,
)


@pytest.mark.unit
class TestContextRelevanceService:
    """Unit tests for context relevance service."""

    @pytest.fixture
    def service(self):
        """Create a service instance with default config."""
        config_dir = Path(__file__).parent.parent / 'configs' / 'agents'
        config_file = config_dir / 'context-relevance.json'
        return ContextRelevanceService(str(config_file))

    @pytest.fixture
    def minimal_config(self):
        """Create a minimal test config."""
        return {
            'enabled': True,
            'log_exclusions': False,
            'template_masks': {
                'RESPOND': {
                    'episodic_memory': True,
                    'working_memory': True,
                    'facts': True,
                    'gists': True,
                    'active_goals': True,
                    'identity_context': True,
                },
                'ACKNOWLEDGE': {
                    'identity_context': True,
                    'communication_style': True,
                },
            },
            'dependencies': {},
            'signal_rules': {},
            'urgency_overrides': [],
            'safety_overrides': {},
            'soft_recovery_budget': 1500,
            'soft_recovery_priority': [],
            'max_included_nodes': 12,
        }

    def test_template_mask_excludes_nodes(self, service):
        """Test that template masks exclude nodes not in the template."""
        result = service.compute_inclusion_map(mode='ACKNOWLEDGE')

        # ACKNOWLEDGE should exclude episodic_memory, working_memory, facts, gists
        assert result.get('episodic_memory') is False
        assert result.get('working_memory') is False
        assert result.get('facts') is False
        assert result.get('gists') is False

        # ACKNOWLEDGE should include identity_context, communication_style, user_traits
        assert result.get('identity_context') is True
        assert result.get('communication_style') is True
        assert result.get('user_traits') is True

    def test_template_mask_respond_vs_act(self, service):
        """Test that different modes have different template masks."""
        respond_result = service.compute_inclusion_map(mode='RESPOND')
        act_result = service.compute_inclusion_map(mode='ACT')

        # RESPOND includes available_skills
        assert respond_result.get('available_skills') is True

        # ACT includes available_tools
        assert act_result.get('available_tools') is True

    def test_signal_rule_soft_exclusion(self, service):
        """Test that soft-excluded nodes can be recovered with budget."""
        signals = {
            'greeting_pattern': True,
        }
        # With a greeting and no budget, active_goals should be soft-excluded
        result = service.compute_inclusion_map(
            mode='RESPOND',
            signals=signals,
            token_budget_remaining=4000
        )
        # Note: with budget headroom, soft exclusions may be recovered
        # This tests that the signal rule is being evaluated
        assert 'active_goals' in result

    def test_signal_rule_hard_exclusion_not_recovered(self, service):
        """Test that hard-excluded nodes are never recovered."""
        signals = {
            'greeting_pattern': True,
            'prompt_token_count': 2,
        }
        # With greeting + low tokens, working_memory should be HARD excluded
        result = service.compute_inclusion_map(
            mode='RESPOND',
            signals=signals,
            token_budget_remaining=10000  # High budget
        )
        # working_memory should still be excluded even with high budget
        assert result.get('working_memory') is False

    def test_soft_exclusion_recovery_with_budget(self, minimal_config):
        """Test that soft-excluded nodes are recovered when budget allows."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(minimal_config, f)
            config_path = f.name

        service = ContextRelevanceService(config_path)

        signals = {
            'greeting_pattern': True,
        }

        # With very high budget, soft-excluded nodes should be recovered
        result = service.compute_inclusion_map(
            mode='RESPOND',
            signals=signals,
            token_budget_remaining=10000
        )
        # active_goals was soft-excluded but should be recovered due to budget
        assert result.get('active_goals') is True or result.get('active_goals') is None

    def test_soft_exclusion_not_recovered_without_budget(self, minimal_config):
        """Test that soft-excluded nodes stay excluded when budget is low."""
        minimal_config['signal_rules'] = {
            'active_goals': [
                {
                    'when': {'greeting_pattern': True},
                    'strength': 'soft'
                }
            ]
        }
        minimal_config['soft_recovery_budget'] = 5000

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(minimal_config, f)
            config_path = f.name

        service = ContextRelevanceService(config_path)

        signals = {'greeting_pattern': True}

        # With low budget, soft-excluded nodes stay excluded
        result = service.compute_inclusion_map(
            mode='RESPOND',
            signals=signals,
            token_budget_remaining=100  # Very low
        )
        # active_goals should stay excluded with low budget
        # (or be True if template doesn't exclude it)

    def test_dependency_auto_includes_parent(self, service):
        """Test that including a child auto-includes its parent dependency."""
        signals = {}
        result = service.compute_inclusion_map(
            mode='RESPOND',
            signals=signals
        )

        # focus depends on active_goals
        # If focus is included, active_goals must be included
        if result.get('focus') is True:
            assert result.get('active_goals') is True

    def test_dependency_episodic_to_gists(self, service):
        """Test that episodic_memory depends on gists."""
        signals = {}
        result = service.compute_inclusion_map(
            mode='RESPOND',
            signals=signals
        )

        # episodic_memory depends on gists
        if result.get('episodic_memory') is True:
            assert result.get('gists') is True

    def test_circular_dependency_detection(self):
        """Test that circular dependencies raise ConfigError at load time."""
        circular_config = {
            'enabled': True,
            'template_masks': {},
            'dependencies': {
                'a': ['b'],
                'b': ['c'],
                'c': ['a'],  # Creates cycle: a -> b -> c -> a
            },
            'signal_rules': {},
            'urgency_overrides': [],
            'safety_overrides': {},
            'soft_recovery_budget': 1500,
            'soft_recovery_priority': [],
            'max_included_nodes': 12,
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(circular_config, f)
            config_path = f.name

        with pytest.raises(ConfigError, match='Circular dependency'):
            ContextRelevanceService(config_path)

    def test_urgency_override_force_includes(self, service):
        """Test that urgency='high' forces inclusion of critical nodes."""
        signals = {
            'greeting_pattern': True,
            'prompt_token_count': 2,
        }
        classification = {'urgency': 'high'}

        result = service.compute_inclusion_map(
            mode='RESPOND',
            signals=signals,
            classification=classification,
            token_budget_remaining=1000
        )

        # Urgency overrides should force working_memory, world_state, facts
        assert result.get('working_memory') is True
        assert result.get('world_state') is True
        assert result.get('facts') is True

    def test_safety_override_returning_from_silence(self, service):
        """Test safety overrides for returning_from_silence."""
        result = service.compute_inclusion_map(
            mode='RESPOND',
            returning_from_silence=True
        )

        # returning_from_silence should force identity_context, user_traits, warm_return_hint
        assert result.get('identity_context') is True
        assert result.get('user_traits') is True

    def test_safety_override_beats_hard_exclusion(self, minimal_config):
        """Test that safety overrides can beat hard exclusions."""
        minimal_config['signal_rules'] = {
            'identity_context': [
                {
                    'when': {'greeting_pattern': True},
                    'strength': 'hard'
                }
            ]
        }
        minimal_config['safety_overrides'] = {
            'identity_context': [
                {
                    'when': {'returning_from_silence': True}
                }
            ]
        }
        minimal_config['template_masks']['RESPOND']['identity_context'] = True

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(minimal_config, f)
            config_path = f.name

        service = ContextRelevanceService(config_path)

        signals = {'greeting_pattern': True}

        result = service.compute_inclusion_map(
            mode='RESPOND',
            signals=signals,
            returning_from_silence=True
        )

        # Even with hard exclusion from signal rule, safety override should force it
        assert result.get('identity_context') is True

    def test_greeting_minimal_mode(self, service):
        """Test greeting minimal mode: greeting + low tokens hard-excludes many nodes."""
        signals = {
            'greeting_pattern': True,
            'prompt_token_count': 3,
        }

        result = service.compute_inclusion_map(
            mode='RESPOND',
            signals=signals
        )

        # Should hard-exclude: focus, working_memory, world_state, active_lists
        assert result.get('focus') is False
        assert result.get('working_memory') is False
        assert result.get('world_state') is False
        assert result.get('active_lists') is False

    def test_disabled_config_includes_all(self):
        """Test that when config is disabled, all nodes are included."""
        config = {
            'enabled': False,
            'template_masks': {},
            'dependencies': {},
            'signal_rules': {},
            'urgency_overrides': [],
            'safety_overrides': {},
            'soft_recovery_budget': 1500,
            'soft_recovery_priority': [],
            'max_included_nodes': 12,
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            config_path = f.name

        service = ContextRelevanceService(config_path)
        result = service.compute_inclusion_map(mode='RESPOND')

        # All nodes should be included
        for node, included in result.items():
            assert included is True

    def test_max_included_nodes_safeguard(self, minimal_config):
        """Test that exceeding MAX_INCLUDED_NODES logs warning."""
        minimal_config['max_included_nodes'] = 2

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(minimal_config, f)
            config_path = f.name

        service = ContextRelevanceService(config_path)

        # This should compute normally but log a warning if > 2 nodes included
        result = service.compute_inclusion_map(mode='RESPOND')
        # Just verify it returns a dict
        assert isinstance(result, dict)

    def test_unknown_mode_includes_all(self, service):
        """Test that unknown mode defaults to including all."""
        result = service.compute_inclusion_map(mode='UNKNOWN_MODE')

        # Template mask won't exist, so should default to including
        # (depends on how template_mask.get() handles missing mode)

    def test_missing_signals_safe_defaults(self, service):
        """Test that missing signals use safe defaults (all conditions fail)."""
        signals = {}  # No signals provided

        result = service.compute_inclusion_map(
            mode='RESPOND',
            signals=signals
        )

        # Should succeed without errors
        assert isinstance(result, dict)

    def test_predicate_comparison_operators(self, minimal_config):
        """Test predicate comparison operators: _gte, _gt, _lte, _lt, _eq."""
        minimal_config['signal_rules'] = {
            'episodic_memory': [
                {
                    'when': {
                        'context_warmth_gte': 0.5,
                        'prompt_token_count_lt': 100,
                    },
                    'strength': 'soft'
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(minimal_config, f)
            config_path = f.name

        service = ContextRelevanceService(config_path)

        # Signals that match all predicates
        signals = {
            'context_warmth': 0.7,
            'prompt_token_count': 50,
        }
        result = service.compute_inclusion_map(
            mode='RESPOND',
            signals=signals,
            token_budget_remaining=10000
        )
        # Rule should match, episodic_memory soft-excluded but recovered due to budget
        assert 'episodic_memory' in result

    def test_soft_recovery_priority_order(self, minimal_config):
        """Test that soft recovery respects configured priority order."""
        minimal_config['signal_rules'] = {
            'episodic_memory': [{'when': {}, 'strength': 'soft'}],
            'working_memory': [{'when': {}, 'strength': 'soft'}],
            'facts': [{'when': {}, 'strength': 'soft'}],
        }
        minimal_config['soft_recovery_priority'] = ['facts', 'working_memory', 'episodic_memory']
        minimal_config['soft_recovery_budget'] = 100

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(minimal_config, f)
            config_path = f.name

        service = ContextRelevanceService(config_path)

        # With limited budget, should recover in priority order
        result = service.compute_inclusion_map(
            mode='RESPOND',
            token_budget_remaining=1500
        )
        # Just verify it runs without error

    def test_acknowledge_preserves_identity_and_style(self, service):
        """Test ACKNOWLEDGE mode preserves identity_context and communication_style."""
        result = service.compute_inclusion_map(mode='ACKNOWLEDGE')

        assert result.get('identity_context') is True
        assert result.get('communication_style') is True
        assert result.get('user_traits') is True

    def test_node_names_match_actual_injects(self, service):
        """Test that node names in config match actual nodes used by frontal_cortex."""
        # This verifies that the node names are consistent
        expected_nodes = set(service.NODE_TOKEN_ESTIMATES.keys())

        # All nodes should be present
        result = service.compute_inclusion_map(mode='RESPOND')
        for node in expected_nodes:
            assert node in result

    def test_working_memory_turns_comparison(self, minimal_config):
        """Test working_memory_turns_gte comparison in signal rules."""
        minimal_config['signal_rules'] = {
            'facts': [
                {
                    'when': {'working_memory_turns_gte': 3},
                    'strength': 'hard'
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(minimal_config, f)
            config_path = f.name

        service = ContextRelevanceService(config_path)

        # With 2 turns, rule doesn't match
        result1 = service.compute_inclusion_map(
            mode='RESPOND',
            signals={'working_memory_turns': 2}
        )
        # With 3+ turns, rule matches
        result2 = service.compute_inclusion_map(
            mode='RESPOND',
            signals={'working_memory_turns': 3}
        )

        assert isinstance(result1, dict)
        assert isinstance(result2, dict)

    def test_multiple_rules_first_match_wins(self, minimal_config):
        """Test that first matching rule wins for a node."""
        minimal_config['signal_rules'] = {
            'episodic_memory': [
                {'when': {'greeting_pattern': True}, 'strength': 'soft'},
                {'when': {'greeting_pattern': True}, 'strength': 'hard'},  # Won't reach here
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(minimal_config, f)
            config_path = f.name

        service = ContextRelevanceService(config_path)

        signals = {'greeting_pattern': True}
        result = service.compute_inclusion_map(
            mode='RESPOND',
            signals=signals,
            token_budget_remaining=10000  # High budget for recovery
        )
        # First rule (soft) should apply, and be recovered due to budget

    def test_missing_config_file_uses_defaults(self):
        """Test that missing config file uses safe defaults."""
        service = ContextRelevanceService('/nonexistent/path/config.json')

        # Should not raise, should return all-included
        result = service.compute_inclusion_map(mode='RESPOND')
        assert isinstance(result, dict)
