"""
Decay Engine Service - Unified periodic decay scheduler across all memory types.

Background service that periodically decays episodic activation scores and
semantic concept strength. Follows IdleConsolidationService pattern.
"""

import time
import math
import logging
from typing import Optional

from .memory_client import MemoryClientService
from .config_service import ConfigService

logger = logging.getLogger(__name__)


class DecayEngineService:
    """Background service that applies decay to all memory types periodically."""

    def __init__(self, decay_interval: int = 1800):
        """
        Initialize decay engine.

        Args:
            decay_interval: Seconds between decay cycles (default: 1800 = 30 minutes)
        """
        self.decay_interval = decay_interval

        # Load decay rates from config
        try:
            episodic_config = ConfigService.get_agent_config("episodic-memory")
            self.episodic_decay_rate = episodic_config.get('episodic_decay_rate', 0.05)
            self.semantic_decay_rate = episodic_config.get('semantic_decay_rate', 0.03)
        except Exception:
            self.episodic_decay_rate = 0.05
            self.semantic_decay_rate = 0.03

        logger.info(
            f"[DECAY ENGINE] Initialized "
            f"(interval={decay_interval}s, "
            f"episodic_rate={self.episodic_decay_rate}, "
            f"semantic_rate={self.semantic_decay_rate})"
        )

    def run(self, shared_state: Optional[dict] = None) -> None:
        """
        Main service loop - periodically runs decay cycles.

        Args:
            shared_state: Optional shared state dict (for consumer integration)
        """
        logger.info("[DECAY ENGINE] Service started")

        while True:
            try:
                time.sleep(self.decay_interval)

                logger.info("[DECAY ENGINE] Running decay cycle...")
                self.run_decay_cycle()

            except KeyboardInterrupt:
                logger.info("[DECAY ENGINE] Service shutting down...")
                break
            except Exception as e:
                logger.error(f"[DECAY ENGINE] Error: {e}", exc_info=True)
                logger.info("[DECAY ENGINE] Waiting 1 minute before retry...")
                time.sleep(60)

    def run_decay_cycle(self):
        """Run one full decay cycle across all memory types."""
        episodic_count = self._decay_episodic()
        semantic_count = self._decay_semantic()
        identity_count = self._apply_identity_inertia()
        external_count = self._decay_external_knowledge()
        trait_stats = self._decay_user_traits()
        thread_dormancy = self._apply_thread_dormancy()

        logger.info(
            f"[DECAY ENGINE] Cycle complete: "
            f"episodic={episodic_count} updated, "
            f"semantic={semantic_count} updated, "
            f"identity={identity_count} inertia-adjusted, "
            f"external_knowledge={external_count} accelerated, "
            f"traits={trait_stats.get('decayed', 0)} decayed/{trait_stats.get('deleted', 0)} deleted, "
            f"threads={thread_dormancy} dormancy-applied"
        )

    def _decay_episodic(self) -> int:
        """
        Apply exponential decay to episodic activation scores.

        Formula: activation_score = activation_score * exp(-decay_rate * hours_since_access)

        SQLite lacks EXP(), so we fetch eligible rows, compute decay in Python,
        then batch-UPDATE.

        Returns:
            Number of episodes updated
        """
        try:
            from .database_service import get_lightweight_db_service

            db_service = get_lightweight_db_service()

            try:
                with db_service.connection() as conn:
                    cursor = conn.cursor()

                    # Fetch episodes eligible for decay (activation > floor, older than 1 hour)
                    cursor.execute("""
                        SELECT id, activation_score,
                               (CAST(strftime('%s', 'now') AS REAL) - CAST(strftime('%s', COALESCE(last_accessed_at, created_at)) AS REAL)) / 3600.0 AS hours_since,
                               json_extract(salience_factors, '$.source') AS sf_source,
                               json_extract(salience_factors, '$.durability') AS sf_durability
                        FROM episodes
                        WHERE deleted_at IS NULL
                          AND activation_score > 0.1
                          AND COALESCE(last_accessed_at, created_at) < datetime('now', '-1 hour')
                    """)
                    rows = cursor.fetchall()

                    updated = 0
                    durability_updated = 0
                    cron_tool_updated = 0

                    for row in rows:
                        episode_id, activation_score, hours_since, sf_source, sf_durability = row

                        # Determine effective decay rate
                        rate = self.episodic_decay_rate

                        # Durability-based accelerated decay for tool_reflection episodes
                        if sf_source == 'tool_reflection':
                            if sf_durability == 'transient':
                                rate = self.episodic_decay_rate * 2.0
                                durability_updated += 1
                            elif sf_durability == 'evolving':
                                rate = self.episodic_decay_rate * 1.5
                                durability_updated += 1

                        # 3x accelerated decay for cron_tool episodes
                        if sf_durability == 'cron_tool':
                            rate = self.episodic_decay_rate * 3.0
                            cron_tool_updated += 1

                        # Compute new activation: activation * exp(-rate * hours)
                        new_activation = max(0.1, activation_score * math.exp(-rate * hours_since))

                        if abs(new_activation - activation_score) > 0.0001:
                            cursor.execute("""
                                UPDATE episodes
                                SET activation_score = ?
                                WHERE id = ?
                            """, (new_activation, episode_id))
                            updated += 1

                    if durability_updated > 0:
                        logger.info(
                            f"[DECAY ENGINE] Applied durability-based decay to "
                            f"{durability_updated} tool_reflection episodes"
                        )

                    if cron_tool_updated > 0:
                        logger.info(
                            f"[DECAY ENGINE] Applied 3x decay to "
                            f"{cron_tool_updated} cron_tool episodes"
                        )

                    cursor.close()

                    if updated > 0:
                        logger.info(f"[DECAY ENGINE] Decayed {updated} episodic activation scores")
                    return updated

            except Exception as e:
                logger.error(f"[DECAY ENGINE] Episodic decay failed: {e}")
                return 0
            finally:
                db_service.close_pool()

        except Exception as e:
            logger.error(f"[DECAY ENGINE] Could not initialize DB for episodic decay: {e}")
            return 0

    def _decay_semantic(self) -> int:
        """
        Apply decay to semantic concept strength, respecting decay_resistance.

        Formula: strength = MAX(0.2, strength - (decay_rate * (1 - decay_resistance)))

        Returns:
            Number of concepts updated
        """
        try:
            from .database_service import get_lightweight_db_service

            db_service = get_lightweight_db_service()

            try:
                with db_service.connection() as conn:
                    cursor = conn.cursor()

                    # Batch update: decay strength respecting decay_resistance, floor at 0.2
                    cursor.execute("""
                        UPDATE semantic_concepts
                        SET strength = MAX(
                            0.2,
                            strength - (? * (1.0 - COALESCE(decay_resistance, 0.5)))
                        ),
                        updated_at = datetime('now')
                        WHERE deleted_at IS NULL
                          AND strength > 0.2
                          AND last_accessed_at < datetime('now', '-1 hour')
                    """, (self.semantic_decay_rate,))

                    updated = cursor.rowcount
                    cursor.close()

                    if updated > 0:
                        logger.info(f"[DECAY ENGINE] Decayed {updated} semantic concept strengths")
                    return updated

            except Exception as e:
                logger.error(f"[DECAY ENGINE] Semantic decay failed: {e}")
                return 0
            finally:
                db_service.close_pool()

        except Exception as e:
            logger.error(f"[DECAY ENGINE] Could not initialize DB for semantic decay: {e}")
            return 0


    # Sources that qualify for accelerated external knowledge decay
    EXTERNAL_KNOWLEDGE_PREFIXES = ("external_specialist:",)

    def _decay_external_knowledge(self) -> int:
        """
        Apply accelerated decay to knowledge tagged as from external sources.

        External knowledge (specialist facts, web search results) in MemoryStore get
        their TTL reduced by the decay multiplier. This ensures external knowledge
        decays 1.5x faster until reinforced by direct experience.

        Returns:
            Number of facts with accelerated decay
        """
        try:
            from .memory_client import MemoryClientService

            store = MemoryClientService.create_connection()

            multiplier = 1.5

            # Scan for fact keys with external source tags
            count = 0
            cursor = 0
            while True:
                cursor, keys = store.scan(cursor, match="fact:*", count=100)
                for key in keys:
                    try:
                        fact_json = store.get(key)
                        if not fact_json:
                            continue

                        import json
                        fact = json.loads(fact_json)
                        source = fact.get('source', '')

                        if source and any(
                            source.startswith(prefix)
                            for prefix in self.EXTERNAL_KNOWLEDGE_PREFIXES
                        ):
                            ttl = store.ttl(key)
                            if ttl > 0:
                                # Reduce TTL by multiplier
                                new_ttl = max(60, int(ttl / multiplier))
                                if new_ttl < ttl:
                                    store.expire(key, new_ttl)
                                    count += 1
                    except Exception:
                        continue

                if cursor == 0:
                    break

            if count > 0:
                logger.info(
                    f"[DECAY ENGINE] Accelerated decay for {count} external knowledge facts "
                    f"(multiplier={multiplier}x)"
                )
            return count

        except Exception as e:
            logger.error(f"[DECAY ENGINE] External knowledge decay failed: {e}")
            return 0

    def _decay_user_traits(self) -> dict:
        """
        Apply confidence decay to user traits via UserTraitService.

        Returns:
            dict: {decayed: int, deleted: int}
        """
        try:
            from .database_service import get_lightweight_db_service
            from .user_trait_service import UserTraitService

            db_service = get_lightweight_db_service()
            try:
                trait_service = UserTraitService(db_service)
                return trait_service.apply_decay()
            finally:
                db_service.close_pool()
        except ImportError:
            return {'decayed': 0, 'deleted': 0}
        except Exception as e:
            logger.error(f"[DECAY ENGINE] User trait decay failed: {e}")
            return {'decayed': 0, 'deleted': 0}

    def _apply_thread_dormancy(self) -> int:
        """
        Apply dormancy rules to curiosity threads.

        Active threads not explored in 45 days → dormant.
        Dormant + engagement < 0.2 + dormant > 60 days → abandoned.

        Returns:
            Number of threads transitioned
        """
        try:
            from .curiosity_thread_service import CuriosityThreadService
            return CuriosityThreadService().apply_dormancy()
        except ImportError:
            return 0
        except Exception as e:
            logger.error(f"[DECAY ENGINE] Thread dormancy failed: {e}")
            return 0

    def _apply_identity_inertia(self) -> int:
        """Apply inertia: pull identity activations toward baselines."""
        try:
            from .database_service import get_lightweight_db_service
            db_service = get_lightweight_db_service()
            try:
                from .identity_service import IdentityService
                identity = IdentityService(db_service)
                return identity.apply_inertia()
            finally:
                db_service.close_pool()
        except Exception as e:
            logger.error(f"[DECAY ENGINE] Identity inertia failed: {e}")
            return 0


def decay_engine_worker(shared_state=None):
    """
    Module-level wrapper for threading.
    Instantiates the service inside the child process.
    """
    # Read config inside child process
    try:
        episodic_config = ConfigService.get_agent_config("episodic-memory")
        decay_interval = episodic_config.get('decay_interval_seconds', 1800)
    except Exception:
        decay_interval = 1800

    service = DecayEngineService(decay_interval=decay_interval)
    service.run(shared_state)
