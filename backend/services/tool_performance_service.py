"""
Tool Performance Service — Records, aggregates, and uses tool invocation metrics.

Tracks success rate, latency, cost, and user preferences per tool.
Provides ranked candidates from triage selection for the ACT dispatch.

Ranking formula (weights sum to 1.0):
  0.40 * success_rate
  0.25 * (1 - normalized_latency)
  0.15 * reliability_score
  0.10 * (1 - normalized_cost)
  0.10 * normalized_preference

Filter bubble prevention:
- Preferences are only 10% of ranking score
- Triage LLM never sees preferences (decides on capability)
- 30-day decay toward neutral for implicit preferences
- New tools start at preference=0 (neutral)
"""

import logging
import time
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

LOG_PREFIX = "[TOOL PERF]"

MAX_LATENCY_NORMALIZATION = 5000.0  # 5s = maximum expected latency
MAX_COST_NORMALIZATION = 1.0        # $1 = maximum expected cost per invocation
PREFERENCE_DECAY_FACTOR = 0.8       # 20% toward neutral every 30 days


class ToolPerformanceService:
    """Records invocations, updates preferences, ranks candidates."""

    def __init__(self, db_service=None):
        self._db = db_service

    def _get_db(self):
        if self._db:
            return self._db
        from services.database_service import DatabaseService, get_merged_db_config
        return DatabaseService(get_merged_db_config())

    def record_invocation(
        self,
        tool_name: str,
        exchange_id: str,
        success: bool,
        latency_ms: float,
        cost: float = 0.0,
        user_id: str = 'default',
    ) -> None:
        """Called from tool_worker after each tool execution."""
        db = self._get_db()
        try:
            # Insert performance metric
            db.execute(
                """
                INSERT INTO tool_performance_metrics
                    (tool_name, exchange_id, invocation_success, latency_ms, cost_estimate)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (tool_name, exchange_id or '', success, latency_ms, cost)
            )

            # Update user preferences: usage_count++, success_count if success
            db.execute(
                """
                INSERT INTO user_tool_preferences (user_id, tool_name, usage_count, success_count, last_used_at)
                VALUES (%s, %s, 1, %s, NOW())
                ON CONFLICT (user_id, tool_name) DO UPDATE SET
                    usage_count = user_tool_preferences.usage_count + 1,
                    success_count = user_tool_preferences.success_count + EXCLUDED.success_count,
                    last_used_at = NOW(),
                    updated_at = NOW()
                """,
                (user_id, tool_name, 1 if success else 0)
            )

            # Positive implicit preference update (+0.05 for successful invocation)
            if success:
                db.execute(
                    """
                    UPDATE user_tool_preferences
                    SET implicit_preference = LEAST(1.0, implicit_preference + 0.05),
                        updated_at = NOW()
                    WHERE user_id = %s AND tool_name = %s
                    """,
                    (user_id, tool_name)
                )

            logger.debug(
                f"{LOG_PREFIX} Recorded: {tool_name} success={success} latency={latency_ms:.0f}ms"
            )
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} record_invocation failed for {tool_name}: {e}")
        finally:
            if not self._db:
                db.close_pool()

    def record_user_correction(self, exchange_id: str, tool_name: str, user_id: str = 'default') -> None:
        """Called when next message indicates user correction."""
        if not exchange_id:
            return
        db = self._get_db()
        try:
            db.execute(
                """
                UPDATE tool_performance_metrics
                SET user_correction = TRUE
                WHERE exchange_id = %s AND tool_name = %s
                """,
                (exchange_id, tool_name)
            )
            # Decrease implicit preference
            db.execute(
                """
                UPDATE user_tool_preferences
                SET implicit_preference = GREATEST(-1.0, implicit_preference - 0.10),
                    updated_at = NOW()
                WHERE user_id = %s AND tool_name = %s
                """,
                (user_id, tool_name)
            )
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} record_user_correction failed: {e}")
        finally:
            if not self._db:
                db.close_pool()

    def get_tool_stats(self, tool_name: str, days: int = 30) -> dict:
        """Aggregate: success_rate, avg_latency, total_cost."""
        db = self._get_db()
        try:
            rows = db.fetch_all(
                """
                SELECT
                    COUNT(*) AS total,
                    COUNT(*) FILTER (WHERE invocation_success) AS successes,
                    AVG(latency_ms) AS avg_latency,
                    SUM(cost_estimate) AS total_cost,
                    AVG(cost_estimate) AS avg_cost
                FROM tool_performance_metrics
                WHERE tool_name = %s
                AND created_at > NOW() - INTERVAL '%s days'
                """,
                (tool_name, days)
            )
            if not rows or not rows[0]['total']:
                return {'success_rate': 0.5, 'avg_latency': 0, 'avg_cost': 0, 'total': 0}

            row = rows[0]
            total = row['total'] or 1
            return {
                'success_rate': (row['successes'] or 0) / total,
                'avg_latency': float(row['avg_latency'] or 0),
                'avg_cost': float(row['avg_cost'] or 0),
                'total': row['total'],
            }
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} get_tool_stats failed: {e}")
            return {'success_rate': 0.5, 'avg_latency': 0, 'avg_cost': 0, 'total': 0}
        finally:
            if not self._db:
                db.close_pool()

    def rank_candidates(self, candidates: List[str], user_id: str = 'default') -> List[dict]:
        """
        Re-rank triage tool candidates by performance + preference.
        Returns list sorted by score descending.
        """
        if not candidates:
            return []

        ranked = []
        for tool_name in candidates:
            stats = self.get_tool_stats(tool_name)
            pref = self._get_user_preference(tool_name, user_id)
            reliability = self._get_reliability(tool_name)

            score = (
                0.40 * stats.get('success_rate', 0.5)
                + 0.25 * (1.0 - self._normalize_latency(stats.get('avg_latency', 0)))
                + 0.15 * reliability
                + 0.10 * (1.0 - self._normalize_cost(stats.get('avg_cost', 0)))
                + 0.10 * self._normalize_preference(pref)
            )

            ranked.append({
                'name': tool_name,
                'score': round(score, 4),
                'stats': stats,
                'preference': pref,
            })

        return sorted(ranked, key=lambda x: x['score'], reverse=True)

    def apply_preference_decay(self, user_id: str = 'default') -> None:
        """
        Apply 30-day preference decay (20% toward neutral).
        Called by the enrichment service or as a scheduled task.
        """
        db = self._get_db()
        try:
            db.execute(
                """
                UPDATE user_tool_preferences
                SET implicit_preference = implicit_preference * %s,
                    updated_at = NOW()
                WHERE user_id = %s
                AND last_used_at < NOW() - INTERVAL '30 days'
                AND ABS(implicit_preference) > 0.01
                """,
                (PREFERENCE_DECAY_FACTOR, user_id)
            )
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} preference decay failed: {e}")
        finally:
            if not self._db:
                db.close_pool()

    def _get_user_preference(self, tool_name: str, user_id: str) -> float:
        """Get combined preference score (explicit + implicit) for a tool."""
        db = self._get_db()
        try:
            rows = db.fetch_all(
                "SELECT explicit_preference, implicit_preference FROM user_tool_preferences WHERE user_id = %s AND tool_name = %s",
                (user_id, tool_name)
            )
            if rows:
                return float(rows[0]['explicit_preference'] or 0) + float(rows[0]['implicit_preference'] or 0)
            return 0.0
        except Exception:
            return 0.0
        finally:
            if not self._db:
                db.close_pool()

    def _get_reliability(self, tool_name: str) -> float:
        """Get reliability_score from tool_capability_profiles."""
        db = self._get_db()
        try:
            rows = db.fetch_all(
                "SELECT reliability_score FROM tool_capability_profiles WHERE tool_name = %s",
                (tool_name,)
            )
            if rows:
                return float(rows[0]['reliability_score'] or 1.0)
            return 1.0
        except Exception:
            return 1.0
        finally:
            if not self._db:
                db.close_pool()

    def _normalize_latency(self, latency_ms: float) -> float:
        """Normalize latency to [0, 1]. 0 = fast, 1 = slow."""
        return min(1.0, latency_ms / MAX_LATENCY_NORMALIZATION)

    def _normalize_cost(self, cost: float) -> float:
        """Normalize cost to [0, 1]. 0 = free, 1 = max cost."""
        return min(1.0, cost / MAX_COST_NORMALIZATION)

    def _normalize_preference(self, pref: float) -> float:
        """Normalize preference [-2, 2] to [0, 1]."""
        return min(1.0, max(0.0, (pref + 2.0) / 4.0))
