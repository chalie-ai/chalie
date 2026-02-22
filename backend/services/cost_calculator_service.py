"""
Cost Calculator Service - ACT loop budget tracking.

Tracks action costs and iteration budgets for the ACT loop.
Mode selection costs are no longer needed (handled by deterministic mode router).
"""

import logging
from typing import Dict, List


class CostCalculatorService:
    """Tracks costs for ACT loop budget management."""

    # Action type complexity multipliers for ACT mode
    ACTION_COMPLEXITY = {
        # Current skill names
        'recall': 2.0,
        'memorize': 1.5,
        'introspect': 1.0,
        'associate': 2.0,
        # Backward-compat aliases
        'memory_query': 2.0,
        'memory_write': 1.5,
        'world_state_read': 1.0,
        'internal_reasoning': 2.5,
        'semantic_query': 2.0,
    }

    def __init__(self, config: dict):
        """
        Initialize cost calculator.

        Args:
            config: Configuration dict with cost parameters
        """
        self.cost_base = config.get('cost_base', 1.0)
        self.cost_growth_factor = config.get('cost_growth_factor', 1.5)

        # Track path efficiency history
        self.path_efficiency_history = []

        # Load procedural memory weights
        self._procedural_weights = None
        self._load_procedural_weights()

    def _load_procedural_weights(self):
        """Load action weights from procedural memory."""
        try:
            from services.procedural_memory_service import ProceduralMemoryService
            from services.database_service import get_shared_db_service
            from services.config_service import ConfigService

            db_service = get_shared_db_service()
            proc_config = ConfigService.get_agent_config("procedural-memory")
            proc_memory = ProceduralMemoryService(db_service, proc_config)
            self._procedural_weights = proc_memory.get_all_policy_weights()

            if self._procedural_weights:
                logging.info(
                    f"[COST CALC] Loaded {len(self._procedural_weights)} "
                    f"procedural weights"
                )
        except Exception as e:
            logging.debug(f"[COST CALC] Procedural memory not available: {e}")
            self._procedural_weights = None

    def get_action_complexity(self, action_type: str) -> float:
        """
        Get action complexity multiplier, preferring procedural memory weights.

        Args:
            action_type: Action type name

        Returns:
            Complexity multiplier
        """
        if self._procedural_weights and action_type in self._procedural_weights:
            return self._procedural_weights[action_type]
        return self.ACTION_COMPLEXITY.get(action_type, 1.5)

    def calculate_iteration_cost(self, iteration_number: int) -> float:
        """
        Calculate base iteration cost with exponential growth.

        Formula: base × (growth_factor ^ iteration_number)
        """
        return self.cost_base * (self.cost_growth_factor ** iteration_number)

    def calculate_action_cost(self, actions: List[Dict]) -> float:
        """
        Calculate total cost for a set of actions.

        Args:
            actions: List of action dicts with 'type' key

        Returns:
            Total action cost
        """
        if not actions:
            return 0.0

        total = 0.0
        for action in actions:
            action_type = action.get('type', 'unknown')
            complexity = self.get_action_complexity(action_type)
            total += complexity * 0.2  # Scaled down

        return total

    def record_path_efficiency(self, efficiency: float) -> None:
        """Record efficiency of chosen path for tracking."""
        self.path_efficiency_history.append(efficiency)

    def map_effort_to_multiplier(self, effort_estimate: str) -> float:
        """Map effort estimate to cognitive effort multiplier."""
        effort_map = {
            'low': 0.8,
            'medium': 1.0,
            'high': 1.2,
            'very_high': 1.5
        }
        return effort_map.get(effort_estimate, 1.0)
