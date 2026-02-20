"""
Routing Decision Service — Logs routing decisions to PostgreSQL.

Lightweight service following CortexIterationService pattern.
Handles logging, feedback updates, and reflection storage.
"""

import json
import uuid
import logging
from typing import Dict, Any, Optional, List

from services.database_service import DatabaseService

logger = logging.getLogger(__name__)

LOG_PREFIX = "[ROUTING DECISION]"


class RoutingDecisionService:
    """Manages routing decision audit trail in PostgreSQL."""

    def __init__(self, db_service: DatabaseService):
        self.db_service = db_service

    def log_decision(
        self,
        topic: str,
        exchange_id: str,
        routing_result: Dict[str, Any],
        previous_mode: Optional[str] = None,
    ) -> str:
        """
        Log a routing decision.

        Args:
            topic: Conversation topic
            exchange_id: Exchange ID
            routing_result: Dict from ModeRouterService.route()
            previous_mode: Mode from last exchange

        Returns:
            UUID of the logged decision
        """
        decision_id = str(uuid.uuid4())

        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO routing_decisions (
                        id, topic, exchange_id, selected_mode,
                        router_confidence, scores, tiebreaker_used,
                        tiebreaker_candidates, margin, effective_margin,
                        signal_snapshot, weight_snapshot, routing_time_ms,
                        previous_mode
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    decision_id,
                    topic,
                    exchange_id,
                    routing_result['mode'],
                    routing_result['router_confidence'],
                    json.dumps(routing_result['scores']),
                    routing_result['tiebreaker_used'],
                    json.dumps(routing_result.get('tiebreaker_candidates')),
                    routing_result['margin'],
                    routing_result['effective_margin'],
                    json.dumps(routing_result['signal_snapshot']),
                    json.dumps(routing_result.get('weight_snapshot')),
                    routing_result['routing_time_ms'],
                    previous_mode,
                ))
                cursor.close()

            logger.debug(f"{LOG_PREFIX} Logged decision {decision_id} for topic '{topic}'")
            return decision_id

        except Exception as e:
            logger.error(f"{LOG_PREFIX} Failed to log decision: {e}")
            return decision_id

    def update_feedback(
        self,
        decision_id: str,
        feedback: Dict[str, Any],
    ):
        """
        Update routing decision with post-exchange feedback.

        Args:
            decision_id: UUID of the routing decision
            feedback: Dict with misroute info, suggested_mode, reward
        """
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE routing_decisions
                    SET feedback = %s
                    WHERE id = %s
                """, (json.dumps(feedback), decision_id))
                cursor.close()

            logger.debug(f"{LOG_PREFIX} Updated feedback for {decision_id}")

        except Exception as e:
            logger.error(f"{LOG_PREFIX} Failed to update feedback: {e}")

    def update_reflection(
        self,
        decision_id: str,
        reflection: Dict[str, Any],
    ):
        """
        Update routing decision with reflection verdict.

        Args:
            decision_id: UUID of the routing decision
            reflection: Dict with ambiguity analysis from reflection service
        """
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE routing_decisions
                    SET reflection = %s
                    WHERE id = %s
                """, (json.dumps(reflection), decision_id))
                cursor.close()

            logger.debug(f"{LOG_PREFIX} Updated reflection for {decision_id}")

        except Exception as e:
            logger.error(f"{LOG_PREFIX} Failed to update reflection: {e}")

    def get_recent_decisions(
        self,
        hours: int = 24,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get recent routing decisions for analysis.

        Args:
            hours: Lookback window in hours
            limit: Maximum number of decisions to return

        Returns:
            List of decision dicts
        """
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, topic, exchange_id, selected_mode,
                           router_confidence, scores, tiebreaker_used,
                           tiebreaker_candidates, margin, effective_margin,
                           signal_snapshot, feedback, reflection,
                           previous_mode, created_at
                    FROM routing_decisions
                    WHERE created_at > NOW() - INTERVAL '%s hours'
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (hours, limit))

                rows = cursor.fetchall()
                cursor.close()

                decisions = []
                for row in rows:
                    decisions.append({
                        'id': str(row[0]),
                        'topic': row[1],
                        'exchange_id': row[2],
                        'selected_mode': row[3],
                        'router_confidence': row[4],
                        'scores': row[5],
                        'tiebreaker_used': row[6],
                        'tiebreaker_candidates': row[7],
                        'margin': row[8],
                        'effective_margin': row[9],
                        'signal_snapshot': row[10],
                        'feedback': row[11],
                        'reflection': row[12],
                        'previous_mode': row[13],
                        'created_at': row[14],
                    })

                return decisions

        except Exception as e:
            logger.error(f"{LOG_PREFIX} Failed to get recent decisions: {e}")
            return []

    def get_unreflected_decisions(
        self,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Get decisions that haven't been reviewed by the reflection service.

        Args:
            limit: Maximum number to return

        Returns:
            List of unreflected decision dicts
        """
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, topic, exchange_id, selected_mode,
                           router_confidence, scores, tiebreaker_used,
                           signal_snapshot, feedback, previous_mode, created_at
                    FROM routing_decisions
                    WHERE reflection IS NULL
                    ORDER BY created_at ASC
                    LIMIT %s
                """, (limit,))

                rows = cursor.fetchall()
                cursor.close()

                decisions = []
                for row in rows:
                    decisions.append({
                        'id': str(row[0]),
                        'topic': row[1],
                        'exchange_id': row[2],
                        'selected_mode': row[3],
                        'router_confidence': row[4],
                        'scores': row[5],
                        'tiebreaker_used': row[6],
                        'signal_snapshot': row[7],
                        'feedback': row[8],
                        'previous_mode': row[9],
                        'created_at': row[10],
                    })

                return decisions

        except Exception as e:
            logger.error(f"{LOG_PREFIX} Failed to get unreflected decisions: {e}")
            return []

    def get_previous_mode(self, topic: str) -> Optional[str]:
        """
        Get the mode from the most recent routing decision for a topic.

        Args:
            topic: Topic name

        Returns:
            Previous mode string or None
        """
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT selected_mode
                    FROM routing_decisions
                    WHERE topic = %s
                    ORDER BY created_at DESC
                    LIMIT 1
                """, (topic,))
                row = cursor.fetchone()
                cursor.close()
                return row[0] if row else None

        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Failed to get previous mode: {e}")
            return None

    def get_mode_distribution(self, hours: int = 168) -> Dict[str, float]:
        """
        Get mode distribution over a time window.

        Args:
            hours: Lookback window (default 168 = 7 days)

        Returns:
            Dict mapping mode to proportion (0.0-1.0)
        """
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT selected_mode, COUNT(*) as cnt
                    FROM routing_decisions
                    WHERE created_at > NOW() - INTERVAL '%s hours'
                    GROUP BY selected_mode
                """, (hours,))

                rows = cursor.fetchall()
                cursor.close()

                total = sum(row[1] for row in rows)
                if total == 0:
                    return {}

                return {row[0]: row[1] / total for row in rows}

        except Exception as e:
            logger.error(f"{LOG_PREFIX} Failed to get mode distribution: {e}")
            return {}

    def get_tiebreaker_rate(self, hours: int = 24) -> float:
        """
        Get tie-breaker invocation rate.

        Args:
            hours: Lookback window

        Returns:
            Proportion of decisions that used tie-breaker (0.0-1.0)
        """
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT
                        COUNT(*) FILTER (WHERE tiebreaker_used = TRUE) as tb_count,
                        COUNT(*) as total
                    FROM routing_decisions
                    WHERE created_at > NOW() - INTERVAL '%s hours'
                """, (hours,))

                row = cursor.fetchone()
                cursor.close()

                if row[1] == 0:
                    return 0.0
                return row[0] / row[1]

        except Exception as e:
            logger.error(f"{LOG_PREFIX} Failed to get tiebreaker rate: {e}")
            return 0.0
