"""
Cortex Iteration Service - Batch logging of ACT loop iterations.

Responsibility: Write iteration data to PostgreSQL in batches for reflection analysis.
"""

import uuid
import json
from typing import List, Dict, Any
from services.database_service import DatabaseService
import logging


class CortexIterationService:
    """Manages batch logging of cortex iteration data."""

    def __init__(self, database_service: DatabaseService):
        """
        Initialize iteration logging service.

        Args:
            database_service: DatabaseService instance for connection management
        """
        self.db_service = database_service

    def calculate_exploration_bonus(self, topic: str, current_actions: List[str]) -> Dict[str, float]:
        """
        Calculate exploration bonus for action types.

        Bonuses:
        - 0.01 per iteration an action was NOT used (existing)
        - 0.05 for first-time action types (NEW)
        - 0.02 for novel action sequences (NEW)

        Returns dict mapping action types to bonus values.
        """
        action_types = ['memory_query', 'memory_write', 'world_state_read',
                       'internal_reasoning', 'background_job', 'schedule', 'semantic_query']

        bonus_dict = {action: 0.0 for action in current_actions}

        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                # Query last 20 iterations for this topic
                cursor.execute("""
                    SELECT actions_executed
                    FROM cortex_iterations
                    WHERE topic = %s
                    ORDER BY created_at DESC
                    LIMIT 20
                """, (topic,))

                rows = cursor.fetchall()

                # Count not-used iterations
                not_used_count = {action: 0 for action in action_types}
                all_historical_actions = set()

                for row in rows:
                    actions_executed = row[0] if row[0] else []
                    executed_actions = set()

                    for action in actions_executed:
                        if isinstance(action, dict) and 'action_type' in action:
                            action_type = action['action_type']
                            executed_actions.add(action_type)
                            all_historical_actions.add(action_type)

                    # Count which actions were NOT executed
                    for action_type in action_types:
                        if action_type not in executed_actions:
                            not_used_count[action_type] += 1

                # Calculate bonuses
                for action_type in current_actions:
                    # Base: 0.01 per unused iteration
                    bonus = not_used_count.get(action_type, 0) * 0.01

                    # NEW: 0.05 bonus for first-time actions
                    if action_type not in all_historical_actions:
                        bonus += 0.05
                        logging.info(f"[EXPLORATION] First-time action: {action_type} (+0.05 bonus)")

                    bonus_dict[action_type] = bonus

                # Cap total bonus at 0.30 (per architecture spec)
                total_bonus = sum(bonus_dict.values())
                if total_bonus > 0.30:
                    scale = 0.30 / total_bonus
                    bonus_dict = {k: v * scale for k, v in bonus_dict.items()}

        except Exception as e:
            logging.error(f"[EXPLORATION BONUS] Failed to calculate: {e}")
            return {action: 0.0 for action in action_types}

        return bonus_dict

    def create_loop_id(self) -> str:
        """
        Generate unique ID for ACT loop execution.

        Returns:
            UUID string for loop_id
        """
        return str(uuid.uuid4())

    def log_iterations_batch(
        self,
        loop_id: str,
        topic: str,
        exchange_id: str,
        session_id: str,
        iterations: List[Dict[str, Any]]
    ) -> None:
        """
        Batch write all iterations from a single ACT loop.

        Args:
            loop_id: Unique ID for this ACT loop execution
            topic: Conversation topic
            exchange_id: Exchange ID from conversation
            session_id: Session ID from session service
            iterations: List of iteration data dicts with all fields

        Raises:
            Exception: Logs error but does not re-raise (non-critical failure)
        """
        if not iterations:
            logging.info(f"[ITERATION LOG] No iterations to log for loop {loop_id}")
            return

        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                for iteration in iterations:
                    cursor.execute("""
                        INSERT INTO cortex_iterations (
                            loop_id, topic, exchange_id, session_id,
                            iteration_number, started_at, completed_at, execution_time_ms,
                            chosen_mode, chosen_confidence, alternative_paths,
                            iteration_cost, diminishing_cost, uncertainty_cost,
                            action_base_cost, total_cost, cumulative_cost,
                            efficiency_score, expected_confidence_gain, task_value, future_leverage, effort_estimate, effort_multiplier, iteration_penalty, exploration_bonus, net_value,
                            decision_override, overridden_mode, termination_reason,
                            actions_executed, action_count, action_success_count,
                            frontal_cortex_response, config_snapshot
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s
                        )
                    """, (
                        loop_id,
                        topic,
                        exchange_id,
                        session_id,
                        iteration['iteration_number'],
                        iteration['started_at'],
                        iteration['completed_at'],
                        iteration['execution_time_ms'],
                        iteration['chosen_mode'],
                        iteration.get('chosen_confidence', 0.5),
                        json.dumps(iteration.get('alternative_paths', [])),
                        iteration.get('iteration_cost', 0.0),
                        iteration.get('diminishing_cost', 0.0),
                        iteration.get('uncertainty_cost', 0.0),
                        iteration.get('action_base_cost', 0.0),
                        iteration.get('total_cost', 0.0),
                        iteration.get('cumulative_cost', 0.0),
                        iteration.get('efficiency_score', 0.0),
                        iteration.get('expected_confidence_gain', 0.0),
                        iteration.get('task_value', 0.0),
                        iteration.get('future_leverage', 0.0),
                        iteration.get('effort_estimate', 'medium'),
                        iteration.get('effort_multiplier', 1.2),
                        iteration.get('iteration_penalty', 1.0),
                        iteration.get('exploration_bonus', 0.0),
                        iteration.get('net_value', 0.0),
                        iteration.get('decision_override', False),
                        iteration.get('overridden_mode'),
                        iteration.get('termination_reason'),
                        json.dumps(iteration.get('actions_executed', [])),
                        iteration.get('action_count', 0),
                        iteration.get('action_success_count', 0),
                        json.dumps(iteration.get('frontal_cortex_response', {})),
                        json.dumps(iteration.get('config_snapshot', {}))
                    ))

                # Note: commit handled by context manager
                logging.info(f"[ITERATION LOG] Logged {len(iterations)} iterations for loop {loop_id}")

        except Exception as e:
            logging.error(f"[ITERATION LOG] Failed to log iterations: {e}")
            # Don't re-raise - logging failure shouldn't crash the loop
