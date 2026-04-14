"""
Tracks trait contradictions that have been surfaced to the user and are awaiting resolution.

Lifecycle:
- Created when a true_contradiction is detected at trait storage time
- Deleted when user responds (resolved via normal ingestion → temporal_change path)
- Cleaned up hourly: both traits get confidence *= 0.5, hard-deleted if <= 0.10
"""
import logging
import uuid
from services.time_utils import utc_now, parse_utc

logger = logging.getLogger(__name__)
LOG_PREFIX = "[PENDING_CONTRADICTION]"


class PendingContradictionService:
    def __init__(self, db_service):
        self.db = db_service

    def create(self, trait_a_id: int, trait_b_id: int, question: str, source: str) -> str:
        """Record a pending contradiction. Returns the new record id."""
        record_id = str(uuid.uuid4())
        with self.db.connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    """INSERT OR IGNORE INTO pending_contradictions
                       (id, trait_a_id, trait_b_id, question, surfaced_at, source)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (record_id, trait_a_id, trait_b_id, question, utc_now().isoformat(), source)
                )
            finally:
                cursor.close()
        logger.info(f"{LOG_PREFIX} Created id={record_id} traits=({trait_a_id},{trait_b_id})")
        return record_id

    def delete(self, record_id: str) -> None:
        with self.db.connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "DELETE FROM pending_contradictions WHERE id = ?", (record_id,)
                )
            finally:
                cursor.close()

    def get_all(self) -> list:
        rows = self.db.fetch_all(
            "SELECT id, trait_a_id, trait_b_id, question, surfaced_at, source FROM pending_contradictions"
        )
        if not rows:
            return []
        return [
            {'id': r['id'], 'trait_a_id': r['trait_a_id'], 'trait_b_id': r['trait_b_id'],
             'question': r['question'], 'surfaced_at': r['surfaced_at'], 'source': r['source']}
            for r in rows
        ]

    def cleanup_stale(self) -> int:
        """
        Hourly cleanup. For each pending contradiction older than 1 hour:
        - Multiply both traits' confidence by 0.5
        - Hard-delete any trait whose confidence drops to <= 0.10
        - Remove the pending record if either trait is deleted

        Returns count of records processed.
        """
        from services.data_graph_service import get_data_graph_service
        dgs = get_data_graph_service()

        records = self.db.fetch_all(
            "SELECT id, trait_a_id, trait_b_id, surfaced_at FROM pending_contradictions"
        )
        if not records:
            return 0

        processed = 0
        for row in records:
            record_id = row['id']
            trait_a_id = row['trait_a_id']
            trait_b_id = row['trait_b_id']
            surfaced_at_str = row['surfaced_at']

            try:
                surfaced_at = parse_utc(surfaced_at_str)
            except Exception:
                continue

            age_seconds = (utc_now() - surfaced_at).total_seconds()
            if age_seconds < 3600:
                continue

            # Apply 50% retrieval_weight decay to both traits
            should_delete_record = False
            for trait_id in (trait_a_id, trait_b_id):
                rw_rows = self.db.fetch_all(
                    "SELECT retrieval_weight FROM data_graph WHERE id = ? AND deleted_at IS NULL",
                    (trait_id,)
                )
                if not rw_rows:
                    should_delete_record = True
                    continue
                current_rw = rw_rows[0]['retrieval_weight']
                new_rw = current_rw * 0.5
                if new_rw <= 0.10:
                    dgs.hard_delete_by_id(trait_id)
                    logger.info(f"{LOG_PREFIX} Hard-deleted trait id={trait_id} retrieval_weight={new_rw:.3f}")
                    should_delete_record = True
                else:
                    dgs.demote(trait_id, factor=0.5)
                    logger.debug(f"{LOG_PREFIX} Decayed trait id={trait_id} retrieval_weight {current_rw:.3f} → {new_rw:.3f}")

            if should_delete_record:
                self.delete(record_id)
                logger.info(f"{LOG_PREFIX} Removed pending record id={record_id}")

            processed += 1

        return processed
