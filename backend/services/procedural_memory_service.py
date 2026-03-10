"""
Procedural Memory Service - Policy weights and skill success stats.

Tracks action outcomes to learn which actions work well over time.
Follows EpisodicStorageService pattern (DatabaseService injection).
"""

import json
import time
import logging
from typing import Dict, Any, Optional, List
from services.database_service import DatabaseService


class ProceduralMemoryService:
    """Manages procedural memory: action weights learned from experience."""

    def __init__(self, database_service: DatabaseService, config: dict = None):
        """
        Initialize procedural memory service.

        Args:
            database_service: DatabaseService instance
            config: Optional config with:
                - reward_history_max: max entries in reward history (default 100)
                - default_action_weight: weight for unknown actions (default 1.0)
                - learning_rate: how fast weights update (default 0.1)
        """
        self.db_service = database_service
        config = config or {}
        self.reward_history_max = config.get('reward_history_max', 100)
        self.default_action_weight = config.get('default_action_weight', 1.0)
        self.learning_rate = config.get('learning_rate', 0.1)

    def record_action_outcome(
        self,
        action_name: str,
        success: bool,
        reward: float = 0.0,
        topic: str = None,
        failure_class: str = None,
    ) -> bool:
        """
        Record the outcome of an action execution.

        Args:
            action_name: Name of the action (e.g., 'memory_query')
            success: Whether the action succeeded
            reward: Reward signal (-1.0 to 1.0)
            topic: Optional topic for context-specific stats
            failure_class: 'internal' or 'external'; logged in reward_history for observability

        Returns:
            True if recorded successfully
        """
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                # Upsert action record
                cursor.execute("""
                    INSERT INTO procedural_memory (action_name, total_attempts, total_successes, weight)
                    VALUES (?, 0, 0, ?)
                    ON CONFLICT (action_name) DO NOTHING
                """, (action_name, self.default_action_weight))

                # Update stats
                cursor.execute("""
                    UPDATE procedural_memory
                    SET total_attempts = total_attempts + 1,
                        total_successes = total_successes + CASE WHEN ? THEN 1 ELSE 0 END,
                        success_rate = CAST(
                            (total_successes + CASE WHEN ? THEN 1 ELSE 0 END) AS REAL
                        ) / CAST((total_attempts + 1) AS REAL),
                        avg_reward = (avg_reward * total_attempts + ?) / (total_attempts + 1),
                        updated_at = datetime('now')
                    WHERE action_name = ?
                """, (success, success, reward, action_name))

                # Update reward history (append, trim to max)
                # In SQLite, we read the current history, append in Python, trim, and write back
                cursor.execute(
                    "SELECT reward_history FROM procedural_memory WHERE action_name = ?",
                    (action_name,)
                )
                row = cursor.fetchone()
                current_history = []
                if row:
                    raw = row[0] if not isinstance(row, dict) else row['reward_history']
                    if raw:
                        if isinstance(raw, str):
                            current_history = json.loads(raw)
                        elif isinstance(raw, list):
                            current_history = raw

                new_entry = {
                    'reward': reward,
                    'success': success,
                    'failure_class': failure_class,
                    'timestamp': time.time()
                }
                current_history.append(new_entry)
                # Sort by timestamp descending and trim to max
                current_history.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
                current_history = current_history[:self.reward_history_max]

                cursor.execute("""
                    UPDATE procedural_memory
                    SET reward_history = ?
                    WHERE action_name = ?
                """, (json.dumps(current_history), action_name))

                # Update context stats if topic provided
                if topic:
                    cursor.execute(
                        "SELECT context_stats FROM procedural_memory WHERE action_name = ?",
                        (action_name,)
                    )
                    row = cursor.fetchone()
                    context_stats = {}
                    if row:
                        raw = row[0] if not isinstance(row, dict) else row['context_stats']
                        if raw:
                            if isinstance(raw, str):
                                context_stats = json.loads(raw)
                            elif isinstance(raw, dict):
                                context_stats = raw

                    topic_data = context_stats.get(topic, {'attempts': 0, 'successes': 0})
                    topic_data['attempts'] = topic_data.get('attempts', 0) + 1
                    if success:
                        topic_data['successes'] = topic_data.get('successes', 0) + 1
                    context_stats[topic] = topic_data

                    cursor.execute("""
                        UPDATE procedural_memory
                        SET context_stats = ?
                        WHERE action_name = ?
                    """, (json.dumps(context_stats), action_name))

                # Recalculate weight using learning rate
                # Target weight: success_rate * (1 + clamp(avg_reward))
                cursor.execute(
                    "SELECT success_rate, avg_reward, weight FROM procedural_memory WHERE action_name = ?",
                    (action_name,)
                )
                row = cursor.fetchone()
                if row:
                    sr = (row[0] if not isinstance(row, dict) else row['success_rate']) or 0.0
                    ar = (row[1] if not isinstance(row, dict) else row['avg_reward']) or 0.0
                    w = (row[2] if not isinstance(row, dict) else row['weight']) or self.default_action_weight
                    clamped_reward = max(-0.5, min(0.5, ar))
                    target = sr * (1.0 + clamped_reward)
                    new_weight = w + self.learning_rate * (target - w)
                    new_weight = max(0.1, min(5.0, new_weight))

                    cursor.execute("""
                        UPDATE procedural_memory
                        SET weight = ?
                        WHERE action_name = ?
                    """, (new_weight, action_name))

                cursor.close()

                fc_tag = f", failure_class={failure_class}" if failure_class else ""
                logging.info(
                    f"[PROCEDURAL] Recorded outcome for '{action_name}': "
                    f"success={success}, reward={reward:.2f}{fc_tag}"
                )
                return True

        except Exception as e:
            logging.error(f"[PROCEDURAL] Failed to record outcome: {e}")
            return False

    def record_gate_rejection(self, action_name: str, reason: str = '') -> bool:
        """
        Record a gate rejection as a soft failure in procedural memory.

        Gate rejections are pre-execution blocks (the action was considered
        but a deterministic gate prevented it). These count as lighter
        failures than execution failures (reward=-0.3 vs -1.0) so the
        action's weight decreases gradually as rejections accumulate.
        """
        return self.record_action_outcome(
            action_name=action_name,
            success=False,
            reward=-0.3,
            failure_class='gate_rejection',
        )

    def get_action_weight(self, action_name: str) -> float:
        """
        Get the current policy weight for an action.

        Args:
            action_name: Name of the action

        Returns:
            Policy weight (default if action unknown)
        """
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    "SELECT weight FROM procedural_memory WHERE action_name = ?",
                    (action_name,)
                )
                row = cursor.fetchone()
                cursor.close()

                if row:
                    return row[0] if not isinstance(row, dict) else row['weight']
                return self.default_action_weight

        except Exception as e:
            logging.error(f"[PROCEDURAL] Failed to get weight for '{action_name}': {e}")
            return self.default_action_weight

    def get_all_policy_weights(self) -> Dict[str, float]:
        """
        Get all action weights as a dict.

        Returns:
            Dict mapping action_name to weight
        """
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                cursor.execute("SELECT action_name, weight FROM procedural_memory")
                rows = cursor.fetchall()
                cursor.close()

                return {
                    (row[0] if not isinstance(row, dict) else row['action_name']):
                    (row[1] if not isinstance(row, dict) else row['weight'])
                    for row in rows
                }

        except Exception as e:
            logging.error(f"[PROCEDURAL] Failed to get all weights: {e}")
            return {}

    def get_ranked_skills(self, topic: str = None) -> List[Dict[str, Any]]:
        """
        Return skills ranked by expected value for a topic.

        Expected value = success_rate * (1 + clamp(avg_reward)) * topic_affinity
        where topic_affinity = per-topic success rate from context_stats, or 1.0 if no data.

        Args:
            topic: Optional topic for context-specific ranking

        Returns:
            List of dicts: [{name, weight, success_rate, avg_reward, attempts, topic_affinity, expected_value}]
            sorted by expected_value descending
        """
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT action_name, weight, success_rate, avg_reward,
                           total_attempts, context_stats
                    FROM procedural_memory
                    ORDER BY weight DESC
                """)

                rows = cursor.fetchall()
                cursor.close()

                ranked = []
                for row in rows:
                    if isinstance(row, dict):
                        action_name = row['action_name']
                        weight = row['weight'] or self.default_action_weight
                        success_rate = row['success_rate'] or 0.0
                        avg_reward = row['avg_reward'] or 0.0
                        total_attempts = row['total_attempts'] or 0
                        context_stats = row['context_stats'] or {}
                    else:
                        action_name = row[0]
                        weight = row[1] or self.default_action_weight
                        success_rate = row[2] or 0.0
                        avg_reward = row[3] or 0.0
                        total_attempts = row[4] or 0
                        context_stats = row[5] or {}

                    # Parse context_stats if it is a JSON string
                    if isinstance(context_stats, str):
                        try:
                            context_stats = json.loads(context_stats)
                        except (json.JSONDecodeError, TypeError):
                            context_stats = {}

                    # Calculate topic affinity
                    topic_affinity = 1.0
                    if topic and topic in context_stats:
                        topic_data = context_stats[topic]
                        topic_attempts = topic_data.get('attempts', 0)
                        topic_successes = topic_data.get('successes', 0)
                        if topic_attempts > 0:
                            topic_affinity = topic_successes / topic_attempts

                    # Expected value formula
                    clamped_reward = max(-0.5, min(0.5, avg_reward))
                    expected_value = success_rate * (1.0 + clamped_reward) * topic_affinity

                    ranked.append({
                        'name': action_name,
                        'weight': weight,
                        'success_rate': success_rate,
                        'avg_reward': avg_reward,
                        'attempts': total_attempts,
                        'topic_affinity': topic_affinity,
                        'expected_value': expected_value,
                    })

                # Sort by expected value descending
                ranked.sort(key=lambda x: x['expected_value'], reverse=True)
                return ranked

        except Exception as e:
            logging.error(f"[PROCEDURAL] Failed to get ranked skills: {e}")
            return []

    def get_action_stats(self, action_name: str) -> Optional[Dict[str, Any]]:
        """
        Get full stats for an action.

        Args:
            action_name: Name of the action

        Returns:
            Stats dict or None
        """
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT action_name, total_attempts, total_successes, success_rate,
                           avg_reward, weight, context_stats, created_at, updated_at
                    FROM procedural_memory
                    WHERE action_name = ?
                """, (action_name,))

                row = cursor.fetchone()
                cursor.close()

                if not row:
                    return None

                if isinstance(row, dict):
                    context_stats = row['context_stats']
                    if isinstance(context_stats, str):
                        try:
                            context_stats = json.loads(context_stats)
                        except (json.JSONDecodeError, TypeError):
                            context_stats = {}
                    return {
                        'action_name': row['action_name'],
                        'total_attempts': row['total_attempts'],
                        'total_successes': row['total_successes'],
                        'success_rate': row['success_rate'],
                        'avg_reward': row['avg_reward'],
                        'weight': row['weight'],
                        'context_stats': context_stats,
                        'created_at': row['created_at'],
                        'updated_at': row['updated_at']
                    }

                context_stats = row[6]
                if isinstance(context_stats, str):
                    try:
                        context_stats = json.loads(context_stats)
                    except (json.JSONDecodeError, TypeError):
                        context_stats = {}
                return {
                    'action_name': row[0],
                    'total_attempts': row[1],
                    'total_successes': row[2],
                    'success_rate': row[3],
                    'avg_reward': row[4],
                    'weight': row[5],
                    'context_stats': context_stats,
                    'created_at': row[7],
                    'updated_at': row[8]
                }

        except Exception as e:
            logging.error(f"[PROCEDURAL] Failed to get stats for '{action_name}': {e}")
            return None
