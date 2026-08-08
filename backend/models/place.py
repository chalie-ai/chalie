"""The ``place`` vertical of ``data_graph`` — named locations (home, work, gym,
…) saved by the ``place`` ability, each row's ``value`` a JSON blob of
coordinates/label/radius. Subclasses the shared
:class:`~models.data_graph.DataGraphRow` shell (persistence, FTS/vec
write-sync, reinforce, kind-scoped live/search); adds only the PLACE kind, its
exact-key embedding-free store/supersede and a dedicated soft-delete. Sole home
of this lane's SQL (I6).

Places are permanent (no decay method, no ``Decayable`` registration) — the
decay engine GCs superseded rows cross-kind for free, same as CONTACT.
"""

from __future__ import annotations

from typing import ClassVar, Self

from models.data_graph import DataGraphRow
from services.time_utils import utc_now


class PlaceRow(DataGraphRow):
    KIND: ClassVar[str] = "place"

    # Retrieval-weight haircut on the demoted (superseded) row (_SUPERSEDE_RW_FACTOR).
    _SUPERSEDE_RW_FACTOR: ClassVar[float] = 0.5

    @classmethod
    def store(cls, key: str, value: str, source: str | None = None) -> tuple[Self, str, str | None]:
        """Exact-key supersede upsert. Returns ``(row, status, old_value)``;
        status is ``created`` | ``reinforced`` | ``superseded``. ``value`` is a
        JSON blob, so it is compared EXACTLY (no case-fold / strip — unlike
        ``ExactKeyRow``, which stores free text): any change to the stored
        coordinates/label supersedes the prior row (ruling #4 — embedding-free
        exact-key supersede restored in the vertical itself)."""
        now = utc_now().isoformat()
        existing = cls.active_by_key(key)
        if existing is None:
            row = cls(kind=cls.KIND, key=key, value=value, source=source,
                      first_seen_at=now, last_confirmed_at=now).save()
            return row, "created", None
        old_value = existing.value
        if value == existing.value:                     # EXACT — JSON, not prose
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
    def delete_by_id(cls, row_id: int) -> bool:
        """Soft-delete the live place row with this id (``active = 0``,
        ``deleted_at`` stamped). Returns ``False`` if no live row has that id
        (delete of a non-existent place)."""
        row = (cls.filter("kind", cls.KIND).filter("id", row_id)
                  .filter("active", 1).first())
        if row is None:
            return False
        row.active = 0
        row.deleted_at = utc_now().isoformat()
        row.save()
        return True
