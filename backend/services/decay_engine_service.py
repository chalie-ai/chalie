"""
Decay Engine Service - Unified periodic decay scheduler across all memory types.

Background service that periodically decays episodic activation scores and
semantic concept strength. Follows IdleConsolidationService pattern.
"""

import math
import logging

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
            self.retrieval_decay_exponent = episodic_config.get('retrieval_decay_exponent', 0.5)
        except Exception as e:
            logger.warning(f"[DECAY ENGINE] Failed to load decay rates from config, using defaults: {e}")
            self.retrieval_decay_exponent = 0.5

        logger.info(
            f"[DECAY ENGINE] Initialized "
            f"(interval={decay_interval}s, "
            f"retrieval_decay_exponent={self.retrieval_decay_exponent})"
        )

    def run_once(self) -> None:
        """Single tick for SubconsciousWorker. Delegates to run_decay_cycle with current richness.

        Resolves richness from SelfModelService; falls back to 1.0 if unavailable.
        Engine logic itself is unchanged — only the trigger surface is new.
        """
        try:
            from services.self_model_service import SelfModelService
            richness = SelfModelService().get_memory_richness()
        except Exception as exc:
            logger.warning(f"[DecayEngine] richness lookup failed, defaulting to 1.0: {exc}")
            richness = 1.0
        self.run_decay_cycle(richness=richness)

    def run_decay_cycle(self, richness: float = 1.0):
        """Run one full decay cycle across all memory types.

        Sub-cycles: _decay_episodic, _decay_data_graph, _cleanup_transcript,
        _purge_tool_calls.

        Args:
            richness: Current memory richness score in [0.0, 1.0]. Reserved for
                future sub-cycle gating; all current sub-cycles run unconditionally.
        """
        episodic_count = self._decay_episodic()
        data_graph_count = self._decay_data_graph()
        transcript_cleaned = self._cleanup_transcript()
        tool_calls_purged = self._purge_tool_calls()

        logger.info(
            f"[DECAY ENGINE] Cycle complete: "
            f"episodic={episodic_count} updated, "
            f"data_graph={data_graph_count} updated, "
            f"transcript_cleaned={transcript_cleaned}, "
            f"tool_calls_purged={tool_calls_purged}"
        )

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
                               (CAST(strftime('%s', 'now') AS REAL) - CAST(strftime('%s', COALESCE(last_accessed_at, created_at)) AS REAL)) / 3600.0 AS hours_since
                        FROM episodes
                        WHERE deleted_at IS NULL
                          AND retrieval_weight > 0.01
                          AND COALESCE(last_accessed_at, created_at) < datetime('now', '-1 hour')
                    """)
                    rows = cursor.fetchall()

                    updated = 0

                    for row in rows:
                        episode_id, retrieval_weight, hours_since = row

                        exponent = self.retrieval_decay_exponent

                        # Power-law decay: rw = rw * (1 + hours)^(-exponent)
                        new_rw = max(0.01, retrieval_weight * math.pow(1.0 + hours_since, -exponent))

                        if abs(new_rw - retrieval_weight) > 0.0001:
                            cursor.execute("""
                                UPDATE episodes
                                SET retrieval_weight = ?
                                WHERE id = ?
                            """, (new_rw, episode_id))
                            updated += 1

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

    def _decay_data_graph(self) -> int:
        """Apply decay to data_graph via DataGraphService."""
        try:
            from .database_service import get_shared_db_service
            from .data_graph_service import DataGraphService
            db = get_shared_db_service()
            try:
                svc = DataGraphService(db)
                return svc.decay_cycle()
            finally:
                db.close_pool()
        except Exception as e:
            logger.error(f"[DECAY ENGINE] Data graph decay failed: {e}", exc_info=True)
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

