"""
Constraint Consolidation Service — Convert recurring gate rejection patterns into episodic memories.

Called from the decay engine on each cycle. Runs at most once per 24 hours (MemoryStore cooldown).
"""

import logging
import time
from typing import Optional

from .memory_client import MemoryClientService

logger = logging.getLogger(__name__)

_CONSTRAINT_CONSOLIDATION_KEY = "constraint_consolidation:last_run"
_CONSTRAINT_CONSOLIDATION_COOLDOWN = 86400  # 24 hours in seconds


def run_constraint_consolidation() -> None:
    """
    Convert recurring gate rejection patterns into episodic memories.

    Queries ConstraintMemoryService for patterns with 10+ rejections over
    7 days, creates episodes for each, deduplicates against existing
    constraint_learning episodes via sqlite-vec similarity.

    Runs at most once per 24h (MemoryStore cooldown flag).
    """
    store = MemoryClientService.create_connection()

    last_run = store.get(_CONSTRAINT_CONSOLIDATION_KEY)
    if last_run:
        logger.debug("[CONSTRAINT CONSOLIDATION] On cooldown, skipping")
        return

    try:
        from services.constraint_memory_service import ConstraintMemoryService
        from services.database_service import get_shared_db_service
        from services.episodic_service import EpisodicService
        from services.embedding_service import get_embedding_service

        cms = ConstraintMemoryService()
        patterns = cms.get_blocked_action_patterns(hours=168)  # 7 days

        significant = [p for p in patterns if p.get('total_rejections', 0) >= 10]

        if not significant:
            logger.debug("[CONSTRAINT CONSOLIDATION] No significant constraint patterns to consolidate")
            store.setex(
                _CONSTRAINT_CONSOLIDATION_KEY,
                _CONSTRAINT_CONSOLIDATION_COOLDOWN,
                str(int(time.time())),
            )
            return

        db = get_shared_db_service()
        episodic = EpisodicService(db)
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

            try:
                embedding = emb_service.generate_embedding(gist)
            except Exception as e:
                logger.warning(
                    f"[CONSTRAINT CONSOLIDATION] Failed to generate embedding "
                    f"for constraint gist: {e}"
                )
                continue

            duplicate = _find_similar_constraint_episode(db, embedding, threshold=0.85)

            if duplicate:
                _boost_episode_activation(db, duplicate['id'])
                boosted += 1
                logger.debug(
                    f"[CONSTRAINT CONSOLIDATION] Boosted existing constraint episode "
                    f"{duplicate['id']} for '{action}'"
                )
                continue

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
                'salience': 3,
                'topic': 'self_reflection',
                'embedding': embedding,
            }

            try:
                episode_id = episodic.store_episode(episode_data)
                created += 1
                logger.info(
                    f"[CONSTRAINT CONSOLIDATION] Created constraint episode "
                    f"{episode_id} for '{action}' ({total} rejections)"
                )
            except Exception as e:
                logger.warning(
                    f"[CONSTRAINT CONSOLIDATION] Failed to store constraint episode "
                    f"for '{action}': {e}"
                )

        logger.info(
            f"[CONSTRAINT CONSOLIDATION] Complete: "
            f"{created} created, {boosted} boosted"
        )

    except Exception as e:
        logger.error(
            f"[CONSTRAINT CONSOLIDATION] Failed: {e}",
            exc_info=True,
        )

    try:
        store.setex(
            _CONSTRAINT_CONSOLIDATION_KEY,
            _CONSTRAINT_CONSOLIDATION_COOLDOWN,
            str(int(time.time())),
        )
    except Exception:
        pass


def _find_similar_constraint_episode(
    db_service, query_embedding, threshold: float = 0.85
) -> Optional[dict]:
    """
    Search existing constraint_learning episodes for semantic duplicates.

    Uses sqlite-vec cosine distance. Returns the most similar episode
    if similarity >= threshold, else None.
    """
    from services.embedding_utils import pack_embedding

    try:
        blob = pack_embedding(query_embedding)

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
            f"[CONSTRAINT CONSOLIDATION] Failed to search for similar "
            f"constraint episodes: {e}"
        )
        return None


def _boost_episode_activation(db_service, episode_id: str) -> None:
    """Boost retrieval_weight for an existing episode on re-encounter."""
    try:
        with db_service.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE episodes
                SET retrieval_weight = MIN(retrieval_weight + 0.3, 1.0),
                    last_accessed_at = datetime('now')
                WHERE id = ?
            """, (episode_id,))
            cursor.close()
    except Exception as e:
        logger.warning(
            f"[CONSTRAINT CONSOLIDATION] Failed to boost episode {episode_id}: {e}"
        )
