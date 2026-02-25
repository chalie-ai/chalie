"""
Context Relevance Pre-Parser Service

Deterministic, rule-based service that gates context node inclusion based on mode
and routing signals. Executes in ~0.1ms with zero LLM calls.

Seven layers applied in order:
1. Template masks (static per-mode)
2. Signal rules (conditional)
3. Urgency overrides
4. Soft exclusion recovery (token budget-aware)
5. Dependency resolution (with circular detection)
6. Safety overrides
7. MAX_INCLUDED_NODES safeguard
"""

import json
import logging
from typing import Dict, Set, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class ContextRelevanceError(Exception):
    """Base exception for context relevance service."""
    pass


class ConfigError(ContextRelevanceError):
    """Configuration error (e.g., circular dependencies)."""
    pass


class ContextRelevanceService:
    """
    Deterministic context relevance pre-parser.
    ~0.1ms execution. No LLM calls.
    """

    # Estimated tokens per node (for soft recovery budget decisions)
    NODE_TOKEN_ESTIMATES = {
        'episodic_memory': 800,
        'working_memory': 600,
        'gists': 500,
        'facts': 300,
        'user_traits': 300,
        'world_state': 300,
        'active_lists': 400,
        'identity_context': 150,
        'focus': 150,
        'communication_style': 100,
        'client_context': 200,
        'available_skills': 200,
        'available_tools': 200,
        'identity_modulation': 100,
        'onboarding_nudge': 80,
        'warm_return_hint': 80,
        'adaptive_directives': 150,
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the service with config from JSON file.
        If config_path not provided, loads from default location.
        """
        if config_path is None:
            # Default: backend/configs/agents/context-relevance.json
            config_path = Path(__file__).parent.parent / 'configs' / 'agents' / 'context-relevance.json'

        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()

    def _load_config(self) -> Dict:
        """Load and parse JSON config file."""
        if not self.config_path.exists():
            # Return minimal safe default
            logger.warning(f"Config file not found at {self.config_path}, using minimal defaults")
            return {
                'enabled': False,
                'template_masks': {},
                'dependencies': {},
                'signal_rules': {},
                'urgency_overrides': [],
                'safety_overrides': [],
                'soft_recovery_budget': 1500,
                'soft_recovery_priority': [],
                'max_included_nodes': 12,
                'log_exclusions': True,
            }

        try:
            with open(self.config_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            raise ConfigError(f"Failed to load config from {self.config_path}: {e}")

    def _validate_config(self):
        """Validate config integrity (e.g., detect circular dependencies)."""
        deps = self.config.get('dependencies', {})
        self._detect_circular_dependencies(deps)

    def _detect_circular_dependencies(self, deps: Dict[str, List[str]]):
        """
        Detect circular dependencies in the dependency graph.
        Raises ConfigError if cycles found.
        """
        visited = set()
        rec_stack = set()

        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in deps.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for node in deps:
            if node not in visited:
                if has_cycle(node):
                    raise ConfigError(
                        f"Circular dependency detected in context-relevance config "
                        f"(node: {node})"
                    )

    def compute_inclusion_map(
        self,
        mode: str,
        signals: Optional[Dict] = None,
        classification: Optional[Dict] = None,
        returning_from_silence: bool = False,
        token_budget_remaining: int = 4000,
    ) -> Dict[str, bool]:
        """
        Compute which context nodes should be included.

        Pipeline:
        1. Start with all True
        2. Apply template_masks[mode] → hard-exclude nodes not in template
        3. Apply signal_rules → exclude (hard or soft) still-True nodes
        4. Apply urgency_overrides → force-include critical nodes when urgent
        5. Recover soft exclusions if budget has headroom
        6. Apply dependency resolution → auto-include deps of all included nodes
        7. Apply safety_overrides → force-include specific nodes
        8. Check MAX_INCLUDED_NODES safeguard
        9. Log structured exclusion report

        Args:
            mode: Cognitive mode (RESPOND, CLARIFY, ACT, ACKNOWLEDGE)
            signals: Routing signals dict (e.g., greeting_pattern, context_warmth, etc.)
            classification: Topic classification dict (includes urgency field)
            returning_from_silence: Whether returning after idle period
            token_budget_remaining: Estimated tokens remaining for soft recovery

        Returns:
            Dict[str, bool] with inclusion decision for each node
        """
        if not self.config.get('enabled', True):
            # If disabled, include everything
            all_nodes = set(self.NODE_TOKEN_ESTIMATES.keys())
            return {node: True for node in all_nodes}

        signals = signals or {}
        classification = classification or {}

        # Layer 1: Start with all nodes
        inclusion_map = {node: True for node in self.NODE_TOKEN_ESTIMATES.keys()}

        # Track exclusion reasons for logging
        excluded_hard = []
        excluded_soft = []

        # Layer 2: Apply template masks (hard exclusion)
        template_mask = self.config.get('template_masks', {}).get(mode, {})
        for node in list(inclusion_map.keys()):
            if not template_mask.get(node, False):
                inclusion_map[node] = False
                excluded_hard.append(node)

        # Layer 3: Apply signal rules (hard and soft)
        # Check all rules and apply strongest exclusion (hard > soft)
        signal_rules = self.config.get('signal_rules', {})
        for node, rules in signal_rules.items():
            if node not in inclusion_map:
                continue
            if not inclusion_map[node]:
                # Already excluded by template
                continue

            matched_hard = False
            matched_soft = False

            for rule in (rules if isinstance(rules, list) else [rules]):
                if self._matches_rule(rule.get('when', {}), signals, returning_from_silence):
                    strength = rule.get('strength', 'soft')
                    if strength == 'hard':
                        matched_hard = True
                    elif strength == 'soft':
                        matched_soft = True

            # Apply strongest exclusion: hard > soft
            if matched_hard:
                inclusion_map[node] = False
                if node not in excluded_hard:
                    excluded_hard.append(node)
            elif matched_soft:
                inclusion_map[node] = False
                if node not in excluded_soft:
                    excluded_soft.append(node)

        # Layer 4: Apply urgency overrides (force-include when urgent)
        urgency = classification.get('urgency', '')
        if urgency == 'high':
            urgency_nodes = self.config.get('urgency_overrides', [])
            for node in urgency_nodes:
                inclusion_map[node] = True
                # Remove from exclusion lists if present
                if node in excluded_hard:
                    excluded_hard.remove(node)
                if node in excluded_soft:
                    excluded_soft.remove(node)

        # Layer 5: Recover soft exclusions based on token budget
        recovered_soft = []
        already_included_estimate = sum(
            self.NODE_TOKEN_ESTIMATES.get(node, 0)
            for node in inclusion_map
            if inclusion_map[node]
        )
        soft_recovery_budget = self.config.get('soft_recovery_budget', 1500)
        headroom = token_budget_remaining - already_included_estimate

        if headroom > soft_recovery_budget:
            # Re-include soft-excluded nodes in priority order
            priority_list = self.config.get('soft_recovery_priority', [])
            for node in priority_list:
                if node in excluded_soft and inclusion_map.get(node) is False:
                    inclusion_map[node] = True
                    recovered_soft.append(node)
                    excluded_soft.remove(node)

        # Layer 6: Apply dependency resolution (auto-include dependencies)
        deps_added = []
        dependencies = self.config.get('dependencies', {})
        changed = True
        max_iterations = 10  # Prevent infinite loops (though circular deps are pre-checked)
        iteration = 0

        while changed and iteration < max_iterations:
            changed = False
            iteration += 1
            for node in list(inclusion_map.keys()):
                if inclusion_map[node]:
                    # Node is included; ensure its dependencies are too
                    for dep in dependencies.get(node, []):
                        if dep in inclusion_map and not inclusion_map[dep]:
                            inclusion_map[dep] = True
                            deps_added.append(dep)
                            changed = True
                            # Remove from exclusion lists
                            if dep in excluded_hard:
                                excluded_hard.remove(dep)
                            if dep in excluded_soft:
                                excluded_soft.remove(dep)

        # Layer 7: Apply safety overrides (force-include under specific conditions)
        safety_overrides = self.config.get('safety_overrides', {})
        overrides_applied = []

        for node, conditions in safety_overrides.items():
            if node not in inclusion_map:
                continue

            # Check if any condition matches
            for condition in (conditions if isinstance(conditions, list) else [conditions]):
                if self._matches_rule(condition.get('when', {}), signals, returning_from_silence):
                    inclusion_map[node] = True
                    overrides_applied.append(node)
                    # Remove from exclusion lists
                    if node in excluded_hard:
                        excluded_hard.remove(node)
                    if node in excluded_soft:
                        excluded_soft.remove(node)
                    break

        # Layer 8: Check MAX_INCLUDED_NODES safeguard
        total_included = sum(1 for v in inclusion_map.values() if v)
        max_nodes = self.config.get('max_included_nodes', 12)
        if total_included > max_nodes:
            logger.warning(
                f"[CONTEXT RELEVANCE] MAX_INCLUDED_NODES exceeded: "
                f"{total_included} > {max_nodes}"
            )

        # Layer 9: Log structured exclusion report
        if self.config.get('log_exclusions', True):
            est_tokens = sum(
                self.NODE_TOKEN_ESTIMATES.get(node, 0)
                for node in inclusion_map
                if inclusion_map[node]
            )
            logger.info(
                f"[CONTEXT RELEVANCE] mode={mode} | "
                f"excluded_hard={excluded_hard} | "
                f"excluded_soft={excluded_soft} | "
                f"recovered_soft={recovered_soft} | "
                f"deps_added={deps_added} | "
                f"overrides_applied={overrides_applied} | "
                f"total_included={total_included} | "
                f"est_tokens={est_tokens}"
            )

        return inclusion_map

    def _matches_rule(
        self,
        when_dict: Dict,
        signals: Dict,
        returning_from_silence: bool,
    ) -> bool:
        """
        Check if all predicates in when_dict match.
        AND logic: all must be true.

        Supports:
        - Exact match: "key": value
        - Comparisons: "key_gte": threshold, "key_lt": threshold, etc.
        - Special predicate: "returning_from_silence": True/False
        """
        for key, expected in when_dict.items():
            if key == 'returning_from_silence':
                if expected != returning_from_silence:
                    return False
                continue

            # Parse comparison operators
            if key.endswith('_gte'):
                actual_key = key[:-4]
                actual = signals.get(actual_key)
                if actual is None or actual < expected:
                    return False
            elif key.endswith('_gt'):
                actual_key = key[:-3]
                actual = signals.get(actual_key)
                if actual is None or actual <= expected:
                    return False
            elif key.endswith('_lte'):
                actual_key = key[:-4]
                actual = signals.get(actual_key)
                if actual is None or actual > expected:
                    return False
            elif key.endswith('_lt'):
                actual_key = key[:-3]
                actual = signals.get(actual_key)
                if actual is None or actual >= expected:
                    return False
            elif key.endswith('_eq'):
                actual_key = key[:-3]
                actual = signals.get(actual_key)
                if actual != expected:
                    return False
            else:
                # Exact match
                actual = signals.get(key)
                if actual != expected:
                    return False

        return True
