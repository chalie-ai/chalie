"""``ExactKeyRow`` — the shared exact-key upsert base for the two ``data_graph``
verticals that address rows by an exact ``key`` rather than by semantic search:
:class:`~models.system_memory.SystemMemoryRow` (searchable knowledge memories)
and :class:`~models.machine_state.MachineStateRow` (operational cursors/clocks/
summary). Abstract — it leaves :attr:`KIND` unset (each subclass declares it) and
adds only the exact-key store/forget/supersede lifecycle on top of the shared
:class:`~models.data_graph.DataGraphRow` shell. Sole home of this lane's SQL (I6).
"""

from __future__ import annotations

from typing import ClassVar, Self

from models.data_graph import DataGraphRow
from services.time_utils import utc_now


class ExactKeyRow(DataGraphRow):
    """Exact-key upsert with temporal supersession. Abstract: subclasses set
    :attr:`KIND`; every read/write below scopes to it via the inherited
    kind-bound helpers."""

    # Retrieval-weight haircut on the demoted (superseded) row (_SUPERSEDE_RW_FACTOR).
    _SUPERSEDE_RW_FACTOR: ClassVar[float] = 0.5

    @classmethod
    def store(cls, key: str, value: str, source: str | None = None) -> tuple[Self, str, str | None]:
        """Exact-key upsert. Returns ``(row, status, old_value)`` where status is
        ``created`` | ``reinforced`` | ``superseded`` (this lane always temporally
        supersedes on a contradicting value)."""
        now = utc_now().isoformat()
        existing = cls.active_by_key(key)
        if existing is None:
            row = cls(kind=cls.KIND, key=key, value=value, source=source,
                      first_seen_at=now, last_confirmed_at=now).save()
            return row, "created", None
        old_value = existing.value
        if (value or "").lower().strip() == (existing.value or "").lower().strip():
            return existing.reinforce(), "reinforced", old_value
        existing.demote()
        row = cls(kind=cls.KIND, key=key, value=value, source=source,
                  first_seen_at=now, last_confirmed_at=now, valid_from=now).save()
        return row, "superseded", old_value

    def demote(self) -> Self:
        """Close this row out on supersession: ``active = 0``, halve
        ``retrieval_weight``, stamp ``valid_to`` so it drops out of live lanes and
        enters the fast-decay window."""
        self.active = 0
        self.retrieval_weight *= self._SUPERSEDE_RW_FACTOR
        self.valid_to = utc_now().isoformat()
        return self.save()

    @classmethod
    def forget(cls, key: str, value: str | None = None) -> int:
        """Bi-temporally invalidate every live row for this kind's exact ``key``
        (optionally only the row whose value matches, case-folded/trimmed):
        ``active = 0``, ``valid_to = now``, ``retrieval_weight`` halved. Matches
        VERBATIM (no canonicalisation). Returns the number of rows closed."""
        now = utc_now().isoformat()
        closed = 0
        for row in cls.live().filter("key", key).get():
            if value is not None and (row.value or "").lower().strip() != value.lower().strip():
                continue
            row.active = 0
            row.valid_to = now
            row.retrieval_weight *= cls._SUPERSEDE_RW_FACTOR
            row.save()
            closed += 1
        return closed
