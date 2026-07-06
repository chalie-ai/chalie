
import logging
from datetime import datetime, timezone
from typing import Optional

from services.time_utils import parse_utc

logger = logging.getLogger(__name__)

# parse_utc returns this when given unparseable input. A persisted timestamp
# that decodes to the sentinel is corruption, not a value — we surface it loudly
# rather than handing a year-0001 datetime to a gate comparison.
_PARSE_SENTINEL = datetime.min.replace(tzinfo=timezone.utc)


class DurableTimestamp:

    def __init__(self, memory_key: str, data_graph_key: str, source: str):
        self._memory_key = memory_key
        self._data_graph_key = data_graph_key
        self._source = source

    def persist(self, when: datetime) -> None:
        iso = when.isoformat()
        self._persist_memory(iso)
        self._persist_data_graph(iso)

    def load(self) -> Optional[datetime]:
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
            from models.system import SystemRow
            SystemRow.store(
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
            from models.system import SystemRow
            row = SystemRow.active_by_key(self._data_graph_key)
            return row.value if row and row.value else None
        except Exception as exc:
            logger.debug(
                "[DurableTimestamp] data_graph hydrate skipped for %s: %s",
                self._data_graph_key, exc,
            )
            return None

    # ── Decode ────────────────────────────────────────────────────────────────

    def _decode(self, raw: str) -> Optional[datetime]:
        dt = parse_utc(raw)
        if dt == _PARSE_SENTINEL:
            logger.warning(
                "[DurableTimestamp] unparseable stored value for %s (raw=%r); "
                "treating as unset",
                self._data_graph_key, raw,
            )
            return None
        return dt
