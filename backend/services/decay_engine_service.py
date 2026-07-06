
import logging

from models.episode import Episode
from services.database import Database

logger = logging.getLogger(__name__)


class DecayEngineService:

    def __init__(self) -> None:
        logger.info("[DECAY ENGINE] Initialized (absolute exponential decay)")

    def run_once(self) -> None:
        self.run_decay_cycle()

    def run_decay_cycle(self) -> None:
        fossils_tombstoned = self._janitor_fossil_episodes()
        episodic_count = self._decay_episodic()
        episodes_deleted = self._delete_expired_episodes()
        data_graph_count = self._decay_data_graph()
        transcript_cleaned = self._cleanup_transcript()

        logger.info(
            f"[DECAY ENGINE] Cycle complete: "
            f"fossils_tombstoned={fossils_tombstoned}, "
            f"episodic={episodic_count} updated, "
            f"episodes_deleted={episodes_deleted}, "
            f"data_graph={data_graph_count} updated, "
            f"transcript_cleaned={transcript_cleaned}"
        )

    def _decay_episodic(self) -> int:
        try:
            with Database.transaction():
                return Episode.decay_weights()
        except Exception as e:
            logger.exception(f"[DECAY ENGINE] Episodic decay failed: {e}")
            return 0

    def _delete_expired_episodes(self) -> int:
        try:
            with Database.transaction():
                return Episode.delete_expired()
        except Exception as e:
            logger.exception(f"[DECAY ENGINE] Episode deletion failed: {e}")
            return 0

    def _janitor_fossil_episodes(self) -> int:
        # Bidirectional dependency: the per-source allowlist lives in
        # services/source_profiles.py; this is the janitor-protection consumer.
        from .source_profiles import janitor_protected_sql

        try:
            with Database.transaction():
                return Episode.tombstone_fossils(janitor_protected_sql())
        except Exception as e:
            logger.exception(f"[DECAY ENGINE] Fossil janitor failed: {e}")
            return 0

    def _decay_data_graph(self) -> int:
        try:
            from .data_graph_service import DataGraphService
            svc = DataGraphService()
            return svc.decay_cycle()
        except Exception as e:
            logger.error(f"[DECAY ENGINE] Data graph decay failed: {e}")
            return 0

    def _cleanup_transcript(self) -> int:
        try:
            from services.transcript_service import Transcript
            return Transcript.cleanup_unlinked_entries()
        except Exception as e:
            logger.warning(f"[DECAY ENGINE] Transcript cleanup failed: {e}")
            return 0
