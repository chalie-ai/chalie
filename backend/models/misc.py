"""The ``misc`` vertical of ``data_graph`` — short-lived scratchpad memory
stored via the memory tool on the ``misc`` kind: notes that are worth keeping
for a couple of days but carry no long-term salience. Subclasses the shared
:class:`~models.data_graph.DataGraphRow` shell (persistence, FTS/vec
write-sync, reinforce, kind-scoped live/search); adds only the MISC kind, its
non-superseding exact-key store and its two-part decay. Sole home of this
lane's SQL (I6).

Misc never contradicts (legacy ``_KIND_POLICY`` ``contradiction: None``,
identical shape to :class:`~models.discovery.DiscoveryRow`): a repeated
identical value reinforces, but a *different* value under the same key
coexists as a fresh row — never demotes the prior one. This is ruling #5: the
legacy service's ``reinforce=False`` no-op for MISC is replaced by the base's
unconditional :meth:`~models.data_graph.DataGraphRow.reinforce` — an accepted
divergence from the pre-rewrite behaviour.

Decay is two parts, run every cycle: (a) a power-law rw-decay toward a 0.0
floor, same shape as :class:`~models.discovery.DiscoveryRow` but with MISC's
own constants (``d_base=1.5``); and (b) a 2-day hard-purge of any row (plus
its FTS + key/value-vec shadows) not reconfirmed within the TTL — MISC is the
one lane in ``_KIND_POLICY`` with ``deletion: 'hard'``, so unlike DISCOVERY it
carries its own GC rather than relying on the engine's shared superseded-row
sweep. Both the 2-day TTL and ``d_base=1.5``/``salience_floor=0.0`` are
ported verbatim from the legacy ``_KIND_POLICY[KIND_MISC]``."""

from __future__ import annotations

from datetime import timedelta
from typing import ClassVar, Self

from models.data_graph import DataGraphRow
from services.time_utils import utc_now


class MiscRow(DataGraphRow):
    KIND: ClassVar[str] = "misc"

    # Power-law live-decay policy (legacy _KIND_POLICY KIND_MISC).
    _DECAY_D_BASE: ClassVar[float] = 1.5
    _DECAY_SALIENCE_FLOOR: ClassVar[float] = 0.0

    # Hard-purge TTL (legacy _KIND_POLICY KIND_MISC ttl_days).
    _PURGE_TTL_DAYS: ClassVar[int] = 2

    @classmethod
    def store(cls, key: str, value: str, source: str | None = None) -> tuple[Self, str, str | None]:
        """Exact-key upsert. Returns ``(row, status, old_value)`` where status is
        ``created`` | ``reinforced`` (misc's ``contradiction: None`` policy
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
        """Two-part MISC maintenance, returning the sum of rows maintained:
        (a) absolute power-law decay of live rows' retrieval_weight toward the
        0.0 floor (idempotent, only rows last confirmed > 1h ago and only when
        the new weight actually moves), then (b) a hard-purge of rows not
        reconfirmed within ``_PURGE_TTL_DAYS`` — the row and its FTS + key/value
        vec shadows are deleted outright, mirroring
        ``orchestrators.decay_engine.DecayEngine._gc_dead_data_graph``'s
        FTS-then-DELETE idiom. Order matches the legacy ``decay_cycle``
        (power-law then purge); the end state is order-independent."""
        return cls._decay_live_rw(cls._DECAY_D_BASE, cls._DECAY_SALIENCE_FLOOR) + cls._purge_expired()

    @classmethod
    def _purge_expired(cls) -> int:
        """Hard-delete MISC rows not reconfirmed within ``_PURGE_TTL_DAYS``,
        purging each row's FTS + key/value vector shadow rows before the base
        row goes (external-content FTS5 must be told the indexed values before
        the source row disappears). The ``DELETE FROM data_graph`` fires an
        AFTER-DELETE trigger that cascades ``expanded_semantic``/
        ``expanded_semantic_vec`` on its own."""
        conn = cls._bound_connection()
        cutoff = (utc_now() - timedelta(days=cls._PURGE_TTL_DAYS)).isoformat()
        dead = conn.execute(
            "SELECT rowid, key, value, kind, search_queries FROM data_graph "
            "WHERE kind = ? AND deleted_at IS NULL "
            "  AND julianday(last_confirmed_at) < julianday(?)",
            (cls.KIND, cutoff),
        ).fetchall()
        for rowid, key, value, kind, search_queries in dead:
            cls._purge_search_index(
                conn, rowid,
                {"key": key, "value": value, "kind": kind, "search_queries": search_queries},
            )
            conn.execute("DELETE FROM data_graph WHERE rowid = ?", (rowid,))
        return len(dead)
