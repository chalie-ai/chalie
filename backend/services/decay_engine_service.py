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

try:
    from services.telemetry_service import (
        get_telemetry_collector,
        GOAL_LIFECYCLE,
    )
    _TELEMETRY_AVAILABLE = True
except Exception as e:  # pragma: no cover
    _TELEMETRY_AVAILABLE = False
    logging.getLogger(__name__).debug(f"Telemetry import unavailable: {e}")

logger = logging.getLogger(__name__)

RELIABILITY_DECAY_MULTIPLIER = {
    'reliable':    1.0,
    'uncertain':   1.5,
    'contradicted': 2.0,
    'superseded':  3.0,
}


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
        except Exception as e:
            logger.warning(f"[DECAY ENGINE] Failed to load decay rates from config, using defaults: {e}")
            self.episodic_decay_rate = 0.05

        logger.info(
            f"[DECAY ENGINE] Initialized "
            f"(interval={decay_interval}s, "
            f"episodic_rate={self.episodic_decay_rate})"
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

                # Self-regulation: check memory richness before decaying
                try:
                    from services.self_model_service import SelfModelService
                    richness = SelfModelService().get_memory_richness()
                    if richness < 0.1:
                        logger.debug(f"[DECAY ENGINE] Richness {richness:.2f} < 0.1, skipping cycle")
                        continue
                except Exception as e:
                    logger.warning(f"[DECAY ENGINE] Memory richness check failed, running decay anyway: {e}")
                    richness = 1.0  # fail-open: run decay if telemetry unavailable

                logger.info("[DECAY ENGINE] Running decay cycle...")
                self.run_decay_cycle(richness=richness)

            except KeyboardInterrupt:
                logger.info("[DECAY ENGINE] Service shutting down...")
                break
            except Exception as e:
                logger.error(f"[DECAY ENGINE] Error: {e}", exc_info=True)
                logger.info("[DECAY ENGINE] Waiting 1 minute before retry...")
                time.sleep(60)

    def run_decay_cycle(self, richness: float = 1.0):
        """Run one full decay cycle across all memory types.

        When richness < 0.3, only essential sub-cycles run (episodic + knowledge).
        Non-essential sub-cycles (identity, external knowledge) are skipped to
        conserve resources on sparse memory systems.

        Args:
            richness: Current memory richness score in [0.0, 1.0].  Values below
                0.3 cause non-essential sub-cycles to be skipped.
        """
        episodic_count = self._decay_episodic()
        knowledge_count = self._decay_knowledge()
        goal_decay_count = self._decay_goals()

        # Non-essential sub-cycles gated on sufficient memory richness
        if richness >= 0.3:
            identity_count = self._apply_identity_inertia()
            external_count = self._decay_external_knowledge()
        else:
            identity_count = external_count = 0
            logger.debug(f"[DECAY ENGINE] Richness {richness:.2f} < 0.3, ran essential sub-cycles only")

        logger.info(
            f"[DECAY ENGINE] Cycle complete: "
            f"episodic={episodic_count} updated, "
            f"knowledge={knowledge_count} updated, "
            f"identity={identity_count} inertia-adjusted, "
            f"external_knowledge={external_count} accelerated, "
            f"goals={goal_decay_count} decayed"
        )

    def _decay_goals(self) -> int:
        """Decay unreinforced goals via :class:`~services.goal_ecology_service.GoalEcologyService`.

        Delegates to :meth:`~services.goal_ecology_service.GoalEcologyService.decay_unreinforced`
        to identify goals whose salience has not been reinforced within their
        configured timescale window.  After the sub-service returns, emits a
        single ``GOAL_LIFECYCLE`` telemetry event summarising the number of goals
        transitioned to *decayed* status during this cycle.

        Returns:
            Number of goals whose status was set to ``'decayed'``, or ``0`` on
            any error (failure is non-fatal; the rest of the decay cycle
            continues regardless).
        """
        try:
            from services.goal_ecology_service import GoalEcologyService

            service = GoalEcologyService()
            decayed_count = service.decay_unreinforced()

            if _TELEMETRY_AVAILABLE:
                try:
                    get_telemetry_collector().record(
                        GOAL_LIFECYCLE,
                        {
                            "action": "decay",
                            "decayed_count": decayed_count,
                            "source": "decay_engine",
                        },
                    )
                except Exception as _tel_err:
                    logger.debug(
                        "[DECAY ENGINE] GOAL_LIFECYCLE 'decay' telemetry emit failed (non-fatal): %s",
                        _tel_err,
                    )

            return decayed_count
        except Exception as e:
            logger.error(f"[DECAY ENGINE] Goal decay sub-cycle failed: {e}", exc_info=True)
            return 0

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
            from .database_service import get_shared_db_service

            db_service = get_shared_db_service()

            try:
                with db_service.connection() as conn:
                    cursor = conn.cursor()

                    # Fetch episodes eligible for decay (activation > floor, older than 1 hour)
                    cursor.execute("""
                        SELECT id, activation_score,
                               (CAST(strftime('%s', 'now') AS REAL) - CAST(strftime('%s', COALESCE(last_accessed_at, created_at)) AS REAL)) / 3600.0 AS hours_since,
                               json_extract(salience_factors, '$.source') AS sf_source,
                               json_extract(salience_factors, '$.durability') AS sf_durability,
                               COALESCE(reliability, 'reliable') AS reliability
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
                        episode_id, activation_score, hours_since, sf_source, sf_durability, reliability = row

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

                        # Reliability multiplier: contradicted/uncertain memories decay faster
                        rate *= RELIABILITY_DECAY_MULTIPLIER.get(reliability, 1.0)

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

    def _decay_knowledge(self) -> int:
        """Apply decay to unified knowledge table via KnowledgeService.

        Delegates to :meth:`KnowledgeService.decay_cycle` which handles
        per-decay-class rates, reliability multipliers, soft-deletion at
        confidence floor, and memory_pressure signal emission.

        Returns:
            Number of knowledge entries updated, or 0 on any error.
        """
        try:
            from .database_service import get_shared_db_service
            from .knowledge_service import KnowledgeService

            db = get_shared_db_service()
            try:
                svc = KnowledgeService(db)
                return svc.decay_cycle()
            finally:
                db.close_pool()
        except Exception as e:
            logger.error(f"[DECAY ENGINE] Knowledge decay failed: {e}", exc_info=True)
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
                    except Exception as e:
                        logger.debug(f"[DECAY ENGINE] Failed to process external knowledge key '{key}': {e}")
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

    def _apply_identity_inertia(self) -> int:
        """Pull identity activations toward their baselines via the inertia mechanism.

        Returns:
            Number of identity vectors whose activation was adjusted.
        """
        try:
            from .database_service import get_shared_db_service
            db_service = get_shared_db_service()
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
    except Exception as e:
        logger.warning(f"[DECAY ENGINE] Failed to load decay_interval from config, using default 1800s: {e}")
        decay_interval = 1800

    service = DecayEngineService(decay_interval=decay_interval)
    service.run(shared_state)
