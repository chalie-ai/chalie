"""The ``system`` vertical of ``data_graph`` — operational state read and
written by exact key: the user-summary prose the prompt builder reads every
turn, the subconscious worker's pattern/geo cursors, durable clocks, and
user-authored "system" memories saved via the memory tool. Subclasses the
shared :class:`~models.data_graph.DataGraphRow` shell (persistence, FTS/vec
write-sync, reinforce, kind-scoped live/search); adds only the SYSTEM kind,
its exact-key store/forget/supersede, and a narrowed FTS write-sync. Sole
home of this lane's SQL (I6)."""

from __future__ import annotations

from typing import ClassVar, Self

from models.data_graph import DataGraphRow
from services.time_utils import utc_now


class SystemRow(DataGraphRow):
    KIND: ClassVar[str] = "system"

    # Retrieval-weight haircut on the demoted (superseded) row (_SUPERSEDE_RW_FACTOR).
    _SUPERSEDE_RW_FACTOR: ClassVar[float] = 0.5

    @classmethod
    def active_by_key(cls, key: str) -> Self | None:
        """The single active row for this kind's exact ``key`` — the store lookup
        (mirrors ``_SELECT_ACTIVE_BY_KIND_KEY_SQL``: ``active = 1`` only, NOT
        ``deleted_at``-filtered)."""
        return (cls.filter("kind", cls.KIND).filter("key", key)
                   .filter("active", 1).first())

    @classmethod
    def store(cls, key: str, value: str, source: str | None = None) -> tuple[Self, str, str | None]:
        """Exact-key upsert. Returns ``(row, status, old_value)`` where status is
        ``created`` | ``reinforced`` | ``superseded`` (system always temporally
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
