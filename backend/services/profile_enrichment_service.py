"""
Profile Enrichment Service — Idle-time tool profile refresh.

Runs every 6 hours in background. Finds the tool profile with the oldest
last_enriched_at timestamp and enriches it with recent related episodes.

Also maintains reliability_score freshness: decays toward 0.5 if no
performance data arrives in 30 days (prevents stale reliability from
over-influencing ranking).
"""

import time
import logging
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

LOG_PREFIX = "[PROFILE ENRICHMENT]"

CYCLE_INTERVAL_SECONDS = 6 * 3600  # 6 hours
RELIABILITY_DECAY_DAYS = 30  # Decay reliability if no new data in 30 days


class ProfileEnrichmentService:
    """Background service for idle-time profile refresh."""

    def __init__(self, db_service=None):
        self._db = db_service

    def _get_db(self):
        if self._db:
            return self._db
        from services.database_service import get_lightweight_db_service
        return get_lightweight_db_service()

    def run(self, shared_state=None):
        """Background loop. Runs enrichment cycle every 6 hours."""
        logger.info(f"{LOG_PREFIX} Starting 6h idle enrichment service...")
        while True:
            try:
                self._enrichment_cycle()
            except Exception as e:
                logger.error(f"{LOG_PREFIX} Enrichment cycle failed: {e}", exc_info=True)
            time.sleep(CYCLE_INTERVAL_SECONDS)

    def _enrichment_cycle(self):
        """One enrichment cycle: find stalest profile, enrich it, update reliability."""
        db = self._get_db()
        try:
            # 1. Find profile with oldest last_enriched_at
            rows = db.fetch_all(
                """
                SELECT tool_name, tool_type, full_profile, embedding IS NOT NULL AS has_embedding,
                       last_enriched_at, enrichment_count
                FROM tool_capability_profiles
                ORDER BY last_enriched_at NULLS FIRST
                LIMIT 1
                """
            )
            if not rows:
                logger.debug(f"{LOG_PREFIX} No profiles to enrich")
                return

            profile_row = rows[0]
            tool_name = profile_row['tool_name']
            logger.info(f"{LOG_PREFIX} Enriching profile: {tool_name} (last_enriched={profile_row['last_enriched_at']})")

            # 2. Pull recent episodes since last enrichment with cosine > 0.5
            since = profile_row['last_enriched_at'] or datetime.now(timezone.utc) - timedelta(days=30)

            from services.embedding_service import get_embedding_service
            emb_service = get_embedding_service()
            profile_embedding = emb_service.generate_embedding(profile_row.get('full_profile', tool_name))
            embedding_str = f"[{','.join(str(x) for x in profile_embedding)}]"

            episode_rows = db.fetch_all(
                """
                SELECT id::text AS id, outcome, gist,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM episodes
                WHERE embedding IS NOT NULL
                AND created_at > %s
                AND 1 - (embedding <=> %s::vector) > 0.5
                ORDER BY similarity DESC
                LIMIT 20
                """,
                (embedding_str, since, embedding_str)
            )

            if episode_rows:
                episode_ids = [r['id'] for r in episode_rows]
                from services.tool_profile_service import ToolProfileService
                added = ToolProfileService().enrich_from_episodes(tool_name, episode_ids)
                logger.info(f"{LOG_PREFIX} {tool_name}: added {added} new scenarios from {len(episode_ids)} episodes")
            else:
                # Still update last_enriched_at to avoid being selected next cycle
                db.execute(
                    "UPDATE tool_capability_profiles SET last_enriched_at = NOW() WHERE tool_name = %s",
                    (tool_name,)
                )
                logger.debug(f"{LOG_PREFIX} {tool_name}: no new relevant episodes found")

            # 3. Re-aggregate performance stats
            self._update_reliability(db, tool_name)

        except Exception as e:
            logger.error(f"{LOG_PREFIX} Cycle error: {e}", exc_info=True)
        finally:
            if not self._db:
                db.close_pool()

    def _update_reliability(self, db, tool_name: str):
        """Update reliability_score and avg_latency from tool_performance_metrics."""
        try:
            rows = db.fetch_all(
                """
                SELECT
                    COUNT(*) FILTER (WHERE invocation_success) AS success_count,
                    COUNT(*) AS total_count,
                    AVG(latency_ms) AS avg_latency,
                    MAX(created_at) AS last_used
                FROM tool_performance_metrics
                WHERE tool_name = %s
                AND created_at > NOW() - INTERVAL '30 days'
                """,
                (tool_name,)
            )

            if not rows or not rows[0]['total_count']:
                # No recent performance data — decay reliability toward 0.5
                db.execute(
                    """
                    UPDATE tool_capability_profiles
                    SET reliability_score = GREATEST(0.5, reliability_score * 0.9),
                        last_enriched_at = NOW()
                    WHERE tool_name = %s
                    """,
                    (tool_name,)
                )
                return

            row = rows[0]
            total = row['total_count'] or 1
            success_rate = (row['success_count'] or 0) / total
            avg_latency = float(row['avg_latency'] or 0)

            db.execute(
                """
                UPDATE tool_capability_profiles
                SET reliability_score = %s,
                    avg_latency_ms = %s,
                    last_enriched_at = NOW()
                WHERE tool_name = %s
                """,
                (round(success_rate, 4), round(avg_latency, 2), tool_name)
            )
            logger.debug(
                f"{LOG_PREFIX} {tool_name}: reliability={success_rate:.2%}, avg_latency={avg_latency:.0f}ms"
            )
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} reliability update failed for {tool_name}: {e}")


def profile_enrichment_worker(shared_state=None):
    """Entry point for consumer.py service registration."""
    service = ProfileEnrichmentService()
    service.run(shared_state)
