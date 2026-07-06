"""The ``user_specific`` (FACTS) vertical of ``data_graph`` — the exact-key
upsert/reinforce/temporal-supersede lane behind the memory tool, the worker
fact pipeline and the prompt's traits section. Subclasses the shared
:class:`~models.data_graph.DataGraphRow` shell (persistence, FTS/vec write-sync,
reinforce, kind-scoped live/search); adds only the FACTS kind, its exact-key
store/forget/supersede and its power-law decay. Sole home of this lane's SQL (I6)."""

from __future__ import annotations

from typing import ClassVar, Self

from models.data_graph import DataGraphRow
from services.time_utils import utc_now


class FactRow(DataGraphRow):
    KIND: ClassVar[str] = "user_specific"

    # Retrieval-weight haircut on the demoted (superseded) row (_SUPERSEDE_RW_FACTOR).
    _SUPERSEDE_RW_FACTOR: ClassVar[float] = 0.5
    # Power-law live-fact decay policy for user_specific (legacy _KIND_POLICY):
    # ttl 30d gates that decay runs at all; d_base/salience_floor set the curve.
    _DECAY_D_BASE: ClassVar[float] = 0.5
    _DECAY_SALIENCE_FLOOR: ClassVar[float] = 0.2
    _DECAY_RW_EPSILON: ClassVar[float] = 0.0001

    @classmethod
    def active_by_key(cls, key: str) -> Self | None:
        """The single active row for this kind's exact ``key`` — the store lookup
        (mirrors ``_SELECT_ACTIVE_BY_KIND_KEY_SQL``: ``active = 1`` only, NOT
        ``deleted_at``-filtered)."""
        return (cls.filter("kind = ?", cls.KIND).filter("key = ?", key)
                   .filter("active = 1").first())

    @classmethod
    def store(cls, key: str, value: str, source: str | None = None) -> tuple[Self, str, str | None]:
        """Exact-key upsert. Returns ``(row, status, old_value)`` where status is
        ``created`` | ``reinforced`` | ``superseded`` (user_specific always
        temporally supersedes on a contradicting value)."""
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
        VERBATIM (no canonicalisation — the LLM chose the key from shown facts).
        Returns the number of rows closed. Ported from ``invalidate``."""
        now = utc_now().isoformat()
        closed = 0
        for row in cls.live().filter("key = ?", key).get():
            if value is not None and (row.value or "").lower().strip() != value.lower().strip():
                continue
            row.active = 0
            row.valid_to = now
            row.retrieval_weight *= cls._SUPERSEDE_RW_FACTOR
            row.save()
            closed += 1
        return closed

    @classmethod
    def decay(cls) -> int:
        """Absolute power-law decay of live FACTS' retrieval_weight (idempotent):
        ``rw = max(salience_floor, max(1, age_days) ** -d_base)`` from
        ``last_confirmed_at``. Only rows last confirmed > 1h ago, and only when the
        new weight actually moves (> epsilon). Returns rows updated. The shared
        superseded-decay and 90-day hard-GC are the DecayEngine's job, not here."""
        from datetime import timedelta  # noqa: PLC0415
        now = utc_now()
        cutoff = (now - timedelta(hours=1)).isoformat()
        now_ts = now.timestamp()
        updated = 0
        for row in cls.live().filter("last_confirmed_at < ?", cutoff).get():
            new_rw = cls._decayed_rw(row.last_confirmed_at, now_ts)
            if new_rw is not None and abs(new_rw - row.retrieval_weight) > cls._DECAY_RW_EPSILON:
                row.retrieval_weight = new_rw
                row.save()
                updated += 1
        return updated

    @classmethod
    def _decayed_rw(cls, confirmed_at: str | None, now_ts: float) -> float | None:
        from services.time_utils import parse_utc  # noqa: PLC0415
        if not confirmed_at:
            return None
        try:
            confirmed_ts = parse_utc(confirmed_at).timestamp()
        except Exception:
            return None
        age_days = (now_ts - confirmed_ts) / 86400.0
        if age_days <= 0:
            return None
        return max(cls._DECAY_SALIENCE_FLOOR, float(max(1.0, age_days) ** (-cls._DECAY_D_BASE)))
