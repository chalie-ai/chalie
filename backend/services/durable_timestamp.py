"""
DurableTimestamp — single home for the persist + hydrate dual-write idiom.

A worker-gate timestamp must survive a process/container restart. The pattern,
established by SubconsciousWorker for ``subconscious_last_fired_at``, is a
dual-write to two stores:

  - MemoryStore (fast in-process read path).
  - data_graph ``kind='system'`` row (durable across MemoryStore eviction and
    process restarts; hydrated back on boot).

This class owns that mechanism once so consumers (SubconsciousWorker's
last-fired clock, WorldState's last-user-message clock) parameterise it by their
own key pair + provenance label rather than duplicating the read/write plumbing.

Bidirectional dependency note:
  - services/subconscious_worker.py uses this for ``subconscious_last_fired_at``.
  - services/world_state.py uses this for ``world_state:last_user_message_at``.
  The data_graph *write/read* surface is reached lazily (alongside memory_client)
  to avoid import cycles at module load; only the ``KIND_SYSTEM`` constant — a
  bare string defined at the top of data_graph_service before any heavy machinery
  — is safe to bind at module level.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from services.data_graph_service import KIND_SYSTEM
from services.time_utils import parse_utc

logger = logging.getLogger(__name__)

# parse_utc returns this when given unparseable input. A persisted timestamp
# that decodes to the sentinel is corruption, not a value — we surface it loudly
# rather than handing a year-0001 datetime to a gate comparison.
_PARSE_SENTINEL = datetime.min.replace(tzinfo=timezone.utc)


class DurableTimestamp:
    """Reads/writes one named UTC timestamp across MemoryStore + data_graph.

    Instances are immutable configuration objects: the key pair and provenance
    label are fixed at construction. ``persist`` and ``load`` carry no state.
    """

    # data_graph kind under which all durable system timestamps are stored.
    # Bound to the canonical constant so this class and data_graph_service never
    # drift on the kind string.
    _DG_KIND = KIND_SYSTEM

    def __init__(self, memory_key: str, data_graph_key: str, source: str):
        """Bind this timestamp to its storage keys.

        Args:
            memory_key: MemoryStore key for the fast read path.
            data_graph_key: data_graph ``key`` (under ``kind='system'``) for the
                durable copy.
            source: Provenance label recorded on the data_graph write.
        """
        self._memory_key = memory_key
        self._data_graph_key = data_graph_key
        self._source = source

    def persist(self, when: datetime) -> None:
        """Write ``when`` to MemoryStore and data_graph.

        Best-effort across both stores; either failure is logged at WARNING but
        does not raise. Split-brain (one written, one not) is real state
        divergence operators need to see, so neither failure is silenced to
        DEBUG.
        """
        iso = when.isoformat()
        self._persist_memory(iso)
        self._persist_data_graph(iso)

    def load(self) -> Optional[datetime]:
        """Hydrate the timestamp: MemoryStore first, then data_graph.

        MemoryStore is the fast path; data_graph survives MemoryStore eviction
        and process restarts.

        Returns:
            The persisted timestamp, or ``None`` when it has never been written.
            ``None`` is the genuine "unset" state, distinct from corruption: a
            value that decodes to the parse sentinel is logged and treated as
            absent so a year-0001 datetime never reaches a gate comparison.
        """
        raw = self._read_memory()
        if raw is None:
            raw = self._read_data_graph()
        if raw is None:
            return None
        return self._decode(raw)

    # ── MemoryStore path ──────────────────────────────────────────────────────

    def _persist_memory(self, iso: str) -> None:
        try:
            from services.memory_client import MemoryClientService
            MemoryClientService.create_connection().set(self._memory_key, iso)
        except Exception as exc:
            logger.warning(
                "[DurableTimestamp] memory persist skipped for %s: %s",
                self._memory_key, exc,
            )

    def _read_memory(self) -> Optional[str]:
        try:
            from services.memory_client import MemoryClientService
            raw = MemoryClientService.create_connection().get(self._memory_key)
            if not raw:
                return None
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            return raw
        except Exception as exc:
            logger.debug(
                "[DurableTimestamp] memory hydrate skipped for %s: %s",
                self._memory_key, exc,
            )
            return None

    # ── data_graph path ───────────────────────────────────────────────────────

    def _persist_data_graph(self, iso: str) -> None:
        try:
            from services.data_graph_service import get_data_graph_service
            get_data_graph_service().store(
                kind=self._DG_KIND,
                key=self._data_graph_key,
                value=iso,
                source=self._source,
            )
        except Exception as exc:
            logger.warning(
                "[DurableTimestamp] data_graph persist skipped for %s: %s",
                self._data_graph_key, exc,
            )

    def _read_data_graph(self) -> Optional[str]:
        try:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()
            with db.connection() as conn:
                row = conn.execute(
                    "SELECT value FROM data_graph "
                    "WHERE kind=? AND key=? AND active=1 AND deleted_at IS NULL "
                    "LIMIT 1",
                    (self._DG_KIND, self._data_graph_key),
                ).fetchone()
            if row and row[0]:
                return row[0]
            return None
        except Exception as exc:
            logger.debug(
                "[DurableTimestamp] data_graph hydrate skipped for %s: %s",
                self._data_graph_key, exc,
            )
            return None

    # ── Decode ────────────────────────────────────────────────────────────────

    def _decode(self, raw: str) -> Optional[datetime]:
        """Parse a stored value, treating the parse sentinel as corruption."""
        dt = parse_utc(raw)
        if dt == _PARSE_SENTINEL:
            logger.warning(
                "[DurableTimestamp] unparseable stored value for %s (raw=%r); "
                "treating as unset",
                self._data_graph_key, raw,
            )
            return None
        return dt
