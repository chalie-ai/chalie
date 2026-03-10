import json
import time
import logging
from typing import Optional

from .memory_client import MemoryClientService
from .config_service import ConfigService
from .semantic_consolidation_tracker import SemanticConsolidationTracker


logger = logging.getLogger(__name__)

# MemoryStore key for constraint consolidation cooldown
_CONSTRAINT_CONSOLIDATION_KEY = "constraint_consolidation:last_run"
_CONSTRAINT_CONSOLIDATION_COOLDOWN = 86400  # 24 hours in seconds


class IdleConsolidationService:
    """
    Background service that triggers batch consolidation during idle periods.

    Monitors memory worker queues and triggers consolidation when:
    - All memory queues are empty (idle state)
    - 1 hour has passed since last consolidation
    """

    def __init__(self, check_interval: int = 300):
        """
        Initialize the idle consolidation service.

        Args:
            check_interval: Seconds between idle checks (default: 300 = 5 minutes)
        """
        self.store = MemoryClientService.create_connection()
        self.tracker = SemanticConsolidationTracker()
        self.check_interval = check_interval

        # Load queue names from config
        config = ConfigService.connections()
        topics = config.get("memory", {}).get("topics", {})

        self.prompt_queue = topics.get("prompt_queue", "prompt-queue")
        self.memory_queue = topics.get("memory_chunker", "memory-chunker-queue")
        self.episodic_queue = topics.get("episodic_memory", "episodic-memory-queue")
        self.semantic_queue = config.get("memory", {}).get("queues", {}).get(
            "semantic_consolidation_queue", {}
        ).get("name", "semantic_consolidation_queue")

        logger.info(
            f"[IDLE CONSOLIDATION] Service initialized "
            f"(check_interval={check_interval}s, idle_threshold=900s)"
        )

    def run(self, shared_state: Optional[dict] = None) -> None:
        """
        Main service loop - checks for idle state and triggers consolidation.

        Args:
            shared_state: Optional shared state dict (for consumer integration)
        """
        logger.info("[IDLE CONSOLIDATION] Service started")

        while True:
            try:
                # Sleep for check interval
                time.sleep(self.check_interval)

                # Check if workers are idle
                if not self._are_workers_idle():
                    logger.debug("[IDLE CONSOLIDATION] Workers not idle, skipping")
                    continue

                # Check if 1 hour has passed since last consolidation
                state = self.tracker.get_state()
                time_since_last = time.time() - state['last_consolidation_time']

                if time_since_last >= 900:  # 15 minutes (fallback; primary trigger is episode-count)
                    logger.info(
                        f"[IDLE CONSOLIDATION] System idle for {time_since_last/60:.1f} minutes, "
                        f"triggering batch consolidation"
                    )
                    self._trigger_consolidation()
                else:
                    logger.debug(
                        f"[IDLE CONSOLIDATION] Idle but only {time_since_last/60:.1f} minutes "
                        f"since last consolidation (need 15 min)"
                    )

            except KeyboardInterrupt:
                logger.info("[IDLE CONSOLIDATION] Service shutting down...")
                break
            except Exception as e:
                logger.error(f"[IDLE CONSOLIDATION] Error: {e}", exc_info=True)
                logger.info("[IDLE CONSOLIDATION] Waiting 1 minute before retry...")
                time.sleep(60)

    def _are_workers_idle(self) -> bool:
        """
        Check if all memory worker queues are empty.

        Returns:
            bool: True if all queues are empty (idle), False otherwise
        """
        queues = [
            self.prompt_queue,
            self.memory_queue,
            self.episodic_queue
        ]

        for queue_name in queues:
            queue_length = self.store.llen(queue_name)
            if queue_length > 0:
                logger.debug(
                    f"[IDLE CONSOLIDATION] Queue '{queue_name}' has {queue_length} items, "
                    f"not idle"
                )
                return False

        return True

    def _trigger_consolidation(self) -> None:
        """Trigger batch consolidation by enqueuing a batch job."""
        try:
            from services.prompt_queue import PromptQueue
            from workers.semantic_consolidation_worker import semantic_consolidation_worker

            batch_job = {
                "type": "batch_consolidation",
                "trigger": "idle_sleep",
                "timestamp": time.time()
            }

            queue = PromptQueue(
                queue_name=self.semantic_queue,
                worker_func=semantic_consolidation_worker,
            )
            queue.enqueue(batch_job)

            logger.info("[IDLE CONSOLIDATION] Enqueued batch consolidation job")

            # Reset tracker after successful enqueue
            self.tracker.reset_episode_count()

        except Exception as e:
            logger.error(f"[IDLE CONSOLIDATION] Failed to enqueue batch job: {e}", exc_info=True)

        # Run constraint consolidation (independent of semantic consolidation)
        self._consolidate_constraints()

    def _consolidate_constraints(self) -> None:
        """
        Convert recurring gate rejection patterns into episodic memories.

        Queries ConstraintMemoryService for patterns with 10+ rejections over
        7 days, creates episodes for each, deduplicates against existing
        constraint_learning episodes via sqlite-vec similarity.

        Runs at most once per 24h (MemoryStore cooldown flag).
        """
        # Check cooldown
        last_run = self.store.get(_CONSTRAINT_CONSOLIDATION_KEY)
        if last_run:
            logger.debug("[IDLE CONSOLIDATION] Constraint consolidation on cooldown")
            return

        try:
            from services.constraint_memory_service import ConstraintMemoryService
            from services.database_service import get_shared_db_service
            from services.episodic_storage_service import EpisodicStorageService
            from services.embedding_service import get_embedding_service

            cms = ConstraintMemoryService()
            patterns = cms.get_blocked_action_patterns(hours=168)  # 7 days

            # Filter to 10+ rejections (consolidation threshold)
            significant = [p for p in patterns if p.get('total_rejections', 0) >= 10]

            if not significant:
                logger.debug("[IDLE CONSOLIDATION] No significant constraint patterns to consolidate")
                # Set cooldown even when nothing to consolidate (avoid re-checking every cycle)
                self.store.setex(
                    _CONSTRAINT_CONSOLIDATION_KEY,
                    _CONSTRAINT_CONSOLIDATION_COOLDOWN,
                    str(int(time.time())),
                )
                return

            db = get_shared_db_service()
            episodic = EpisodicStorageService(db)
            emb_service = get_embedding_service()

            created = 0
            boosted = 0

            for pattern in significant:
                action = pattern['action']
                total = pattern['total_rejections']
                top_reason = pattern['top_reason']

                gist = (
                    f"Attempted {action} {total} times over 7 days; "
                    f"blocked because {top_reason}"
                )

                # Dedup: search existing episodes for semantic similarity
                try:
                    embedding = emb_service.generate_embedding(gist)
                except Exception as e:
                    logger.warning(
                        f"[IDLE CONSOLIDATION] Failed to generate embedding "
                        f"for constraint gist: {e}"
                    )
                    continue

                duplicate = self._find_similar_constraint_episode(
                    db, embedding, threshold=0.85
                )

                if duplicate:
                    # Boost activation count instead of creating duplicate
                    self._boost_episode_activation(db, duplicate['id'])
                    boosted += 1
                    logger.debug(
                        f"[IDLE CONSOLIDATION] Boosted existing constraint episode "
                        f"{duplicate['id']} for '{action}'"
                    )
                    continue

                # Create new episode
                episode_data = {
                    'intent': {
                        'type': 'constraint_learning',
                        'action': action,
                    },
                    'context': {
                        'total_rejections': total,
                        'top_reason': top_reason,
                        'reason_breakdown': pattern.get('reason_breakdown', {}),
                    },
                    'action': f"learned constraint: {action} blocked by {top_reason}",
                    'emotion': {'valence': 0.0, 'label': 'neutral'},
                    'outcome': 'constraint_learned',
                    'gist': gist,
                    'salience': 3,  # Low — background observation
                    'freshness': 1.0,
                    'topic': 'self_reflection',
                    'embedding': embedding,
                }

                try:
                    episode_id = episodic.store_episode(episode_data)
                    created += 1
                    logger.info(
                        f"[IDLE CONSOLIDATION] Created constraint episode "
                        f"{episode_id} for '{action}' ({total} rejections)"
                    )
                except Exception as e:
                    logger.warning(
                        f"[IDLE CONSOLIDATION] Failed to store constraint episode "
                        f"for '{action}': {e}"
                    )

            logger.info(
                f"[IDLE CONSOLIDATION] Constraint consolidation complete: "
                f"{created} created, {boosted} boosted"
            )

        except Exception as e:
            logger.error(
                f"[IDLE CONSOLIDATION] Constraint consolidation failed: {e}",
                exc_info=True,
            )

        # Set cooldown regardless of outcome
        try:
            self.store.setex(
                _CONSTRAINT_CONSOLIDATION_KEY,
                _CONSTRAINT_CONSOLIDATION_COOLDOWN,
                str(int(time.time())),
            )
        except Exception:
            pass

    @staticmethod
    def _find_similar_constraint_episode(
        db_service, query_embedding, threshold: float = 0.85
    ) -> Optional[dict]:
        """
        Search existing constraint_learning episodes for semantic duplicates.

        Uses sqlite-vec cosine distance. Returns the most similar episode
        if similarity >= threshold, else None.
        """
        import struct

        try:
            blob = struct.pack(f'{len(query_embedding)}f', *query_embedding)

            with db_service.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT e.id, e.gist, v.distance
                    FROM episodes e
                    JOIN episodes_vec v ON v.rowid = e.rowid
                    WHERE v.embedding MATCH ? AND k = 5
                      AND e.deleted_at IS NULL
                      AND e.outcome = 'constraint_learned'
                    ORDER BY v.distance
                    LIMIT 1
                """, (blob,))

                row = cursor.fetchone()
                cursor.close()

                if not row:
                    return None

                # sqlite-vec cosine distance: 0 = identical, 2 = opposite
                # similarity = 1 - (distance / 2)
                distance = row[2] if not isinstance(row, dict) else row['distance']
                similarity = 1.0 - (distance / 2.0)

                if similarity >= threshold:
                    return {
                        'id': row[0] if not isinstance(row, dict) else row['id'],
                        'gist': row[1] if not isinstance(row, dict) else row['gist'],
                        'similarity': similarity,
                    }

                return None

        except Exception as e:
            logger.warning(
                f"[IDLE CONSOLIDATION] Failed to search for similar "
                f"constraint episodes: {e}"
            )
            return None

    @staticmethod
    def _boost_episode_activation(db_service, episode_id: str) -> None:
        """Increment activation_score for an existing episode."""
        try:
            with db_service.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE episodes
                    SET activation_score = activation_score + 1,
                        last_accessed_at = datetime('now')
                    WHERE id = ?
                """, (episode_id,))
                cursor.close()
        except Exception as e:
            logger.warning(
                f"[IDLE CONSOLIDATION] Failed to boost episode {episode_id}: {e}"
            )


def idle_consolidation_process(shared_state):
    """Top-level entry point for thread spawn.

    Creates the service instance inside the child process to avoid
    pickling MemoryStore connections (which contain _thread.lock objects).
    """
    service = IdleConsolidationService()
    service.run(shared_state)
