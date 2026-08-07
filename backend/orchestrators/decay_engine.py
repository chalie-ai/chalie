"""The decay orchestrator — a pure sequencer over registered
:class:`~orchestrators.decayable.Decayable` subsystems, plus the cross-kind
``data_graph`` maintenance that belongs to no single vertical.

Two responsibilities:

1. **Sequence** every registered Decayable, each isolated in its own
   try/except — one blowing up is logged LOUDLY (with traceback) and skipped,
   never aborting the cycle. This is the anti-R1 guard: a dead subsystem
   surfaces as a loud failure, never a silent zero that hides dead decay for
   weeks.
2. **Own** the generic ``data_graph`` table maintenance that no single kind
   owns: the superseded-row fast-decay (``retrieval_weight`` → ``0.01`` for
   ``active = 0`` rows) and the 90-day hard-GC of long-dead superseded rows,
   including each row's FTS + key/value vector shadow-row purge. These run as
   raw SQL keyed by rowid/status — never through a per-kind vertical.

Per-kind ``data_graph`` decay (the live-fact power-law weights, the misc TTL
purge) belongs to each kind's vertical and registers here as a Decayable when
the vertical lands (E1+); the engine holds none of it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import timedelta

from orchestrators.decayable import Decayable
from services._fts_delete import FtsDelete
from services.database import Database
from services.time_utils import utc_now

logger = logging.getLogger(__name__)
LOG_PREFIX = "[DECAY ENGINE]"

# Superseded (active=0) rows are pinned to this near-zero retrieval weight so
# they stop competing with live facts in recall, then hard-GC'd after the window.
_SUPERSEDED_DECAY_WEIGHT = 0.01
_SUPERSEDED_DELETE_AFTER_DAYS = 90


class DecayEngine:
    """Sequences registered Decayables and runs the cross-kind ``data_graph``
    maintenance, isolating every unit so one failure never aborts the cycle."""

    def __init__(self) -> None:
        self._decayables: list[Decayable] = []

    def register(self, decayable: Decayable) -> DecayEngine:
        """Add a Decayable to the cycle; returns ``self`` for chaining — the
        single registration seam every future per-vertical decayable plugs
        into."""
        self._decayables.append(decayable)
        return self

    def run(self) -> int:
        """Run one full decay cycle — the engine-owned ``data_graph``
        maintenance first, then every registered Decayable, each isolated.
        Returns the total rows maintained across all units."""
        total = self._isolate("data_graph superseded fast-decay", self._decay_superseded_data_graph)
        total += self._isolate("data_graph 90-day GC", self._gc_dead_data_graph)
        for decayable in self._decayables:
            total += self._isolate(type(decayable).__name__, decayable.decay)
        return total

    @staticmethod
    def _isolate(name: str, unit: Callable[[], int]) -> int:
        """Run one maintenance unit, isolating any failure: a blow-up is logged
        LOUDLY (with traceback) and skipped so the cycle continues — never a
        silent zero (the R1 anti-pattern that hid dead decay for weeks)."""
        try:
            count = unit()
            logger.info("%s %s: %d maintained", LOG_PREFIX, name, count)
            return count
        except Exception:
            logger.exception("%s %s BLEW UP — isolated, cycle continues", LOG_PREFIX, name)
            return 0

    # ── cross-kind data_graph maintenance (raw SQL, no vertical) ──────────────

    @staticmethod
    def _decay_superseded_data_graph() -> int:
        """Pin every superseded (``active = 0``) live ``data_graph`` row to the
        near-zero retrieval weight so it stops resurfacing in recall. Returns
        rows updated (rows already at or below the floor are skipped, so a
        settled corpus produces zero writes)."""
        with Database.transaction() as conn:
            cursor = conn.execute(
                "UPDATE data_graph SET retrieval_weight = ? "
                "WHERE active = 0 AND deleted_at IS NULL AND retrieval_weight > ?",
                (_SUPERSEDED_DECAY_WEIGHT, _SUPERSEDED_DECAY_WEIGHT),
            )
            return cursor.rowcount or 0

    @staticmethod
    def _gc_dead_data_graph() -> int:
        """Hard-delete superseded (``active = 0``) rows invalidated past the
        90-day window, purging each row's FTS + key/value vector shadow rows.
        The invalidation clock is ``valid_to``, falling back to
        ``last_confirmed_at`` for rows superseded before the bi-temporal columns
        existed. Returns rows deleted."""
        cutoff = (utc_now() - timedelta(days=_SUPERSEDED_DELETE_AFTER_DAYS)).isoformat()
        with Database.transaction() as conn:
            dead = conn.execute(
                "SELECT rowid, key, value, kind FROM data_graph "
                "WHERE active = 0 AND deleted_at IS NULL "
                "  AND julianday(COALESCE(valid_to, last_confirmed_at)) < julianday(?)",
                (cutoff,),
            ).fetchall()
            for rowid, key, value, kind in dead:
                # FTS is external-content: purge its posting with the indexed
                # values BEFORE the base row goes, in the schema's column order
                # (key, value, kind).
                FtsDelete.fts5_external_delete(
                    conn, "data_graph_fts", rowid,
                    {"key": key, "value": value, "kind": kind},
                )
                conn.execute("DELETE FROM data_graph WHERE rowid = ?", (rowid,))
                conn.execute("DELETE FROM data_graph_key_vec WHERE rowid = ?", (rowid,))
                conn.execute("DELETE FROM data_graph_value_vec WHERE rowid = ?", (rowid,))
            return len(dead)
