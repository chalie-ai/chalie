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
        topic: str = None
    ) -> bool:
        """
        Record the outcome of an action execution.

        Args:
            action_name: Name of the action (e.g., 'memory_query')
            success: Whether the action succeeded
            reward: Reward signal (-1.0 to 1.0)
            topic: Optional topic for context-specific stats

        Returns:
            True if recorded successfully
        """
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                # Upsert action record
                cursor.execute("""
                    INSERT INTO procedural_memory (action_name, total_attempts, total_successes, weight)
                    VALUES (%s, 0, 0, %s)
                    ON CONFLICT (action_name) DO NOTHING
                """, (action_name, self.default_action_weight))

                # Update stats
                cursor.execute("""
                    UPDATE procedural_memory
                    SET total_attempts = total_attempts + 1,
                        total_successes = total_successes + CASE WHEN %s THEN 1 ELSE 0 END,
                        success_rate = (total_successes + CASE WHEN %s THEN 1 ELSE 0 END)::FLOAT
                                       / (total_attempts + 1)::FLOAT,
                        avg_reward = (avg_reward * total_attempts + %s) / (total_attempts + 1),
                        updated_at = NOW()
                    WHERE action_name = %s
                """, (success, success, reward, action_name))

                # Update reward history (append, trim to max)
                cursor.execute("""
                    UPDATE procedural_memory
                    SET reward_history = (
                        SELECT jsonb_agg(elem)
                        FROM (
                            SELECT elem
                            FROM jsonb_array_elements(
                                reward_history || %s::jsonb
                            ) AS elem
                            ORDER BY elem->>'timestamp' DESC
                            LIMIT %s
                        ) sub
                    )
                    WHERE action_name = %s
                """, (
                    json.dumps([{'reward': reward, 'success': success, 'timestamp': time.time()}]),
                    self.reward_history_max,
                    action_name
                ))

                # Update context stats if topic provided
                if topic:
                    cursor.execute("""
                        UPDATE procedural_memory
                        SET context_stats = jsonb_set(
                            COALESCE(context_stats, '{}'::jsonb),
                            %s,
                            COALESCE(context_stats->%s, '{"attempts": 0, "successes": 0}'::jsonb)
                            || jsonb_build_object(
                                'attempts',
                                (COALESCE((context_stats->%s->>'attempts')::int, 0) + 1),
                                'successes',
                                (COALESCE((context_stats->%s->>'successes')::int, 0) + CASE WHEN %s THEN 1 ELSE 0 END)
                            )
                        )
                        WHERE action_name = %s
                    """, (
                        '{' + topic + '}',
                        topic, topic, topic,
                        success,
                        action_name
                    ))

                # Recalculate weight using learning rate
                cursor.execute("""
                    UPDATE procedural_memory
                    SET weight = GREATEST(0.1, LEAST(5.0,
                        weight + %s * (%s - weight)
                    ))
                    WHERE action_name = %s
                """, (
                    self.learning_rate,
                    # Target weight: success_rate * (1 + avg_reward)
                    0.0,  # placeholder, calculated below
                    action_name
                ))

                # Actually compute the target weight properly
                cursor.execute("""
                    UPDATE procedural_memory
                    SET weight = GREATEST(0.1, LEAST(5.0,
                        weight + %s * (
                            (success_rate * (1.0 + GREATEST(-0.5, LEAST(0.5, avg_reward))))
                            - weight
                        )
                    ))
                    WHERE action_name = %s
                """, (self.learning_rate, action_name))

                cursor.close()

                logging.info(
                    f"[PROCEDURAL] Recorded outcome for '{action_name}': "
                    f"success={success}, reward={reward:.2f}"
                )
                return True

        except Exception as e:
            logging.error(f"[PROCEDURAL] Failed to record outcome: {e}")
            return False

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
                    "SELECT weight FROM procedural_memory WHERE action_name = %s",
                    (action_name,)
                )
                row = cursor.fetchone()
                cursor.close()

                if row:
                    return row[0]
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

                return {row[0]: row[1] for row in rows}

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
                    action_name = row[0]
                    weight = row[1] or self.default_action_weight
                    success_rate = row[2] or 0.0
                    avg_reward = row[3] or 0.0
                    total_attempts = row[4] or 0
                    context_stats = row[5] or {}

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
                    WHERE action_name = %s
                """, (action_name,))

                row = cursor.fetchone()
                cursor.close()

                if not row:
                    return None

                return {
                    'action_name': row[0],
                    'total_attempts': row[1],
                    'total_successes': row[2],
                    'success_rate': row[3],
                    'avg_reward': row[4],
                    'weight': row[5],
                    'context_stats': row[6],
                    'created_at': row[7],
                    'updated_at': row[8]
                }

        except Exception as e:
            logging.error(f"[PROCEDURAL] Failed to get stats for '{action_name}': {e}")
            return None
