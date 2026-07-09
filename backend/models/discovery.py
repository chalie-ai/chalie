"""The ``discovery`` vertical of ``data_graph`` — findings from the
proactive-research loop (:class:`~cron.jobs.discovery.DiscoveryJob`),
stored via the memory tool on the ``discovery`` channel
and recalled (recall-first dedup, restored in a later slice) via FTS.
Subclasses the shared :class:`~models.data_graph.DataGraphRow` shell
(persistence, FTS/vec write-sync, reinforce, kind-scoped live/search); adds
only the DISCOVERY kind, its non-superseding exact-key store and its
power-law decay. Sole home of this lane's SQL (I6).

Discovery never contradicts (legacy ``_KIND_POLICY`` ``contradiction: None``):
a repeated identical value reinforces, but a *different* value under the same
key coexists as a fresh row — never demotes the prior one. Findings are never
forgotten by any caller, so this vertical carries no ``forget``/``demote``/
``delete_by_id``. Every row is FTS-indexed on insert (the recall-first premise
depends on it): ``discovery`` inherits the base ``data_graph`` search config (the
``Searchable`` trait), unlike the exact-key-only kinds
(:class:`~models.machine_state.MachineStateRow`) that declare ``__search__ =
None`` and are excluded from the shared write-sync.

Decay is a straight power-law rw-decay toward a floor, identical in shape and
constants to :class:`~models.fact.FactRow` (the legacy ``KIND_DISCOVERY``
policy shared ``d_base=0.5``/``salience_floor=0.2`` with ``KIND_USER_SPECIFIC``).
The old ``ttl_days=14`` never entered that math — it only gated whether a kind
decayed at all, which the vertical now expresses by existing as a
:class:`~orchestrators.decayable.Decayable` rather than by a stored constant;
the 1h cutoff plus the 0.2 floor already encode roughly the same ~2-week
horizon (floor reached at ~25 days, same as FACTS)."""

from __future__ import annotations

from typing import ClassVar, Self

from models.data_graph import DataGraphRow
from services.time_utils import utc_now


class DiscoveryRow(DataGraphRow):
    KIND: ClassVar[str] = "discovery"

    # Power-law live-discovery decay policy (legacy _KIND_POLICY KIND_DISCOVERY):
    # identical d_base/salience_floor to FACTS (KIND_USER_SPECIFIC).
    _DECAY_D_BASE: ClassVar[float] = 0.5
    _DECAY_SALIENCE_FLOOR: ClassVar[float] = 0.2
    _DECAY_RW_EPSILON: ClassVar[float] = 0.0001

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
        ``created`` | ``reinforced`` (discovery's ``contradiction: None`` policy
        never supersedes: a different value under the same key inserts a fresh
        coexisting row rather than demoting the prior one, so ``superseded``
        never occurs here)."""
        now = utc_now().isoformat()
        existing = cls.active_by_key(key)
        if existing is None:
            row = cls(kind=cls.KIND, key=key, value=value, source=source,
                      first_seen_at=now, last_confirmed_at=now).save()
            return row, "created", None
        if (value or "").lower().strip() == (existing.value or "").lower().strip():
            return existing.reinforce(), "reinforced", existing.value
        row = cls(kind=cls.KIND, key=key, value=value, source=source,
                  first_seen_at=now, last_confirmed_at=now).save()
        return row, "created", None

    @classmethod
    def decay(cls) -> int:
        """Absolute power-law decay of live DISCOVERY rows' retrieval_weight
        (idempotent): ``rw = max(salience_floor, max(1, age_days) ** -d_base)``
        from ``last_confirmed_at``. Only rows last confirmed > 1h ago, and only
        when the new weight actually moves (> epsilon). Returns rows updated.
        The shared superseded-decay and 90-day hard-GC are the DecayEngine's
        job, not here."""
        from datetime import timedelta  # noqa: PLC0415
        now = utc_now()
        cutoff = (now - timedelta(hours=1)).isoformat()
        now_ts = now.timestamp()
        updated = 0
        for row in cls.live().filter("last_confirmed_at", cutoff, "<").get():
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
