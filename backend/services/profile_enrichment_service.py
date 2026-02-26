"""
Profile Enrichment Service — Idle-time tool profile refresh.

Runs every 6 hours in background. Finds up to 3 tool profiles with the oldest
last_enriched_at timestamps and enriches them with recent related episodes.

Also maintains reliability_score freshness: decays toward 0.5 if no
performance data arrives in 30 days (prevents stale reliability from
over-influencing ranking). Applies preference decay for stale tools and
checks for usage-triggered full profile rebuilds.
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
        """One enrichment cycle: find stalest profiles (up to 3), enrich, update reliability."""
        db = self._get_db()
        try:
            # 1. Find profiles with oldest last_enriched_at (up to 3 per cycle)
            rows = db.fetch_all(
                """
                SELECT tool_name, tool_type, full_profile, embedding IS NOT NULL AS has_embedding,
                       last_enriched_at, enrichment_count
                FROM tool_capability_profiles
                ORDER BY last_enriched_at NULLS FIRST
                LIMIT 3
                """
            )
            if not rows:
                logger.debug(f"{LOG_PREFIX} No profiles to enrich")
                return

            for profile_row in rows:
                try:
                    self._enrich_single_profile(db, profile_row)
                except Exception as e:
                    logger.error(f"{LOG_PREFIX} Single profile enrichment failed for {profile_row.get('tool_name')}: {e}")

            # Apply preference decay for stale tools
            try:
                from services.tool_performance_service import ToolPerformanceService
                ToolPerformanceService(db_service=db).apply_preference_decay()
            except Exception as e:
                logger.debug(f"{LOG_PREFIX} Preference decay failed: {e}")

        except Exception as e:
            logger.error(f"{LOG_PREFIX} Cycle error: {e}", exc_info=True)
        finally:
            if not self._db:
                db.close_pool()

    def _enrich_single_profile(self, db, profile_row: dict):
        """Enrich a single tool profile with episodes, update reliability, check rebuild triggers."""
        tool_name = profile_row['tool_name']
        logger.info(f"{LOG_PREFIX} Enriching profile: {tool_name} (last_enriched={profile_row['last_enriched_at']})")

        # Pull recent episodes since last enrichment with cosine > 0.5
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

        # Re-aggregate performance stats
        updated_reliability = self._update_reliability(db, tool_name)

        # Check for usage-triggered full profile rebuild
        if updated_reliability is not None:
            self._check_rebuild_triggers(db, tool_name, updated_reliability)

    def _update_reliability(self, db, tool_name: str) -> float:
        """Update reliability_score and avg_latency from tool_performance_metrics.

        Returns the updated reliability score, or None if no data.
        """
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
                return None

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
            return success_rate
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} reliability update failed for {tool_name}: {e}")
            return None

    def _check_rebuild_triggers(self, db, tool_name: str, reliability_score: float):
        """Check if tool profile needs a full rebuild based on usage signals."""
        try:
            rows = db.fetch_all(
                """SELECT enrichment_count,
                          (SELECT COUNT(*) FROM tool_performance_metrics
                           WHERE tool_name = %s) AS total_uses,
                          (SELECT COUNT(*) FILTER (WHERE invocation_success)
                           FROM tool_performance_metrics
                           WHERE tool_name = %s) AS successful_uses
                   FROM tool_capability_profiles WHERE tool_name = %s""",
                (tool_name, tool_name, tool_name)
            )
            if not rows:
                return

            total_uses = rows[0].get('total_uses', 0)
            successful_uses = rows[0].get('successful_uses', 0)
            enrichment_count = rows[0].get('enrichment_count', 0)

            should_rebuild = False
            reason = ''

            # Trigger A: After 15 successful uses (enough data for informed rebuild)
            if successful_uses >= 15 and enrichment_count < 2:
                should_rebuild = True
                reason = f'usage_milestone ({successful_uses} successes)'

            # Trigger B: Reliability dropped below 0.5 (profile may be misleading triage)
            if reliability_score < 0.5 and total_uses >= 5:
                should_rebuild = True
                reason = f'low_reliability ({reliability_score:.2f} over {total_uses} uses)'

            if should_rebuild:
                logger.info(
                    f"{LOG_PREFIX} Triggering full profile rebuild for {tool_name}: {reason}"
                )
                from services.tool_registry_service import ToolRegistryService
                manifest = ToolRegistryService().get_tool_full_description(tool_name)
                if manifest:
                    from services.tool_profile_service import ToolProfileService
                    ToolProfileService().build_profile(tool_name, manifest, force=True)
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Rebuild trigger check failed for {tool_name}: {e}")


def profile_enrichment_worker(shared_state=None):
    """Entry point for consumer.py service registration."""
    service = ProfileEnrichmentService()
    service.run(shared_state)
