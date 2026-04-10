"""
Decay Engine Service - Unified periodic decay scheduler across all memory types.

Background service that periodically decays episodic activation scores and
semantic concept strength. Follows IdleConsolidationService pattern.
"""

import time
import math
import logging
from typing import Optional

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
            self.retrieval_decay_exponent = episodic_config.get('retrieval_decay_exponent', 0.5)
        except Exception as e:
            logger.warning(f"[DECAY ENGINE] Failed to load decay rates from config, using defaults: {e}")
            self.retrieval_decay_exponent = 0.5

        logger.info(
            f"[DECAY ENGINE] Initialized "
            f"(interval={decay_interval}s, "
            f"retrieval_decay_exponent={self.retrieval_decay_exponent})"
        )

    def run(self, shared_state: Optional[dict] = None) -> None:
        """
        Main service loop - periodically runs decay cycles.

        Args:
            shared_state: Optional shared state dict (for consumer integration)
        """
        logger.info("[DECAY ENGINE] Service started")
        self._cleanup_legacy_store_keys()

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
        reconsolidation_count = self._process_pending_reconsolidation()
        consolidation_count = self._run_episode_consolidation()
        knowledge_count = self._decay_knowledge()
        goal_decay_count = self._decay_goals()
        transcript_cleaned = self._cleanup_transcript()
        tool_calls_purged = self._purge_tool_calls()

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
            f"reconsolidation={reconsolidation_count} processed, "
            f"consolidation={consolidation_count} super-episodes, "
            f"knowledge={knowledge_count} updated, "
            f"transcript_cleaned={transcript_cleaned}, "
            f"tool_calls_purged={tool_calls_purged}, "
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
        Apply power-law decay to episodic retrieval_weight.

        Formula: rw = rw * (1 + hours_since_access)^(-exponent)

        storage_strength is never modified by decay — only increases on access.
        No hard-delete at any threshold.

        Reliability multipliers still apply (contradicted memories decay faster).

        Returns:
            Number of episodes updated
        """
        try:
            from .database_service import get_shared_db_service

            db_service = get_shared_db_service()

            try:
                with db_service.connection() as conn:
                    cursor = conn.cursor()

                    cursor.execute("""
                        SELECT id, retrieval_weight,
                               (CAST(strftime('%s', 'now') AS REAL) - CAST(strftime('%s', COALESCE(last_accessed_at, created_at)) AS REAL)) / 3600.0 AS hours_since,
                               json_extract(salience_factors, '$.source') AS sf_source,
                               json_extract(salience_factors, '$.durability') AS sf_durability
                        FROM episodes
                        WHERE deleted_at IS NULL
                          AND retrieval_weight > 0.01
                          AND COALESCE(last_accessed_at, created_at) < datetime('now', '-1 hour')
                    """)
                    rows = cursor.fetchall()

                    updated = 0
                    durability_updated = 0

                    for row in rows:
                        episode_id, retrieval_weight, hours_since, sf_source, sf_durability = row

                        exponent = self.retrieval_decay_exponent

                        # Durability-based accelerated decay for tool_reflection episodes
                        if sf_source == 'tool_reflection':
                            if sf_durability == 'transient':
                                exponent = self.retrieval_decay_exponent * 2.0
                                durability_updated += 1
                            elif sf_durability == 'evolving':
                                exponent = self.retrieval_decay_exponent * 1.5
                                durability_updated += 1

                        # Power-law decay: rw = rw * (1 + hours)^(-exponent)
                        new_rw = max(0.01, retrieval_weight * math.pow(1.0 + hours_since, -exponent))

                        if abs(new_rw - retrieval_weight) > 0.0001:
                            cursor.execute("""
                                UPDATE episodes
                                SET retrieval_weight = ?
                                WHERE id = ?
                            """, (new_rw, episode_id))
                            updated += 1

                    if durability_updated > 0:
                        logger.info(
                            f"[DECAY ENGINE] Applied durability-based decay to "
                            f"{durability_updated} tool_reflection episodes"
                        )

                    cursor.close()

                    if updated > 0:
                        logger.info(f"[DECAY ENGINE] Decayed {updated} episodic retrieval weights")
                    return updated

            except Exception as e:
                logger.error(f"[DECAY ENGINE] Episodic decay failed: {e}")
                return 0
            finally:
                db_service.close_pool()

        except Exception as e:
            logger.error(f"[DECAY ENGINE] Could not initialize DB for episodic decay: {e}")
            return 0

    def _process_pending_reconsolidation(self) -> int:
        try:
            from .database_service import get_shared_db_service
            from .episodic_service import EpisodicService

            db_service = get_shared_db_service()
            try:
                svc = EpisodicService(db_service)
                return svc.process_pending_reconsolidation()
            finally:
                db_service.close_pool()
        except Exception as e:
            logger.error(f"[DECAY ENGINE] Pending reconsolidation failed: {e}", exc_info=True)
            return 0

    def _run_episode_consolidation(self) -> int:
        """Consolidate clusters of similar episodes into super episodes.

        Returns:
            Number of super episodes created, or 0 on any error.
        """
        try:
            from .database_service import get_shared_db_service
            from .episode_consolidation_service import EpisodeConsolidationService

            db_service = get_shared_db_service()
            try:
                svc = EpisodeConsolidationService(db_service)
                return svc.run_consolidation_cycle()
            finally:
                db_service.close_pool()
        except Exception as e:
            logger.error(f"[DECAY ENGINE] Episode consolidation failed: {e}", exc_info=True)
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

    def _cleanup_legacy_store_keys(self) -> None:
        """Remove stale MemoryStore keys left by previous pipeline versions."""
        try:
            from .memory_client import MemoryClientService
            store = MemoryClientService.create_connection()
            for key in store.keys("semantic_consolidation:*"):
                store.delete(key)
        except Exception as e:
            logger.debug(f"[DECAY ENGINE] Legacy store key cleanup non-fatal: {e}")

    def _cleanup_transcript(self) -> int:
        """Delete unlinked transcript entries below compaction watermark."""
        try:
            from services import transcript_service
            return transcript_service.cleanup_unlinked_entries()
        except Exception as e:
            logger.debug(f"[DECAY ENGINE] Transcript cleanup non-fatal: {e}")
            return 0

    def _purge_tool_calls(self, max_rows: int = 25000) -> int:
        """Purge tool_calls rows beyond the newest max_rows entries."""
        try:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()
            with db.connection() as conn:
                total = conn.execute("SELECT COUNT(*) FROM tool_calls").fetchone()[0]
                if total <= max_rows:
                    return 0
                excess = total - max_rows
                conn.execute("""
                    DELETE FROM tool_calls WHERE rowid IN (
                        SELECT rowid FROM tool_calls ORDER BY created_at ASC LIMIT ?
                    )
                """, (excess,))
                logger.info(f"[DECAY ENGINE] Purged {excess} tool_calls rows (kept {max_rows})")
                return excess
        except Exception as e:
            logger.debug(f"[DECAY ENGINE] Tool calls purge non-fatal: {e}")
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
