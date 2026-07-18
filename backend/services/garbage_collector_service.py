"""GarbageCollectorService — the hourly retention sweep over ``transcript``
and ``tool_calls``.

Consolidates what used to be two separate mechanisms (a dedicated
tool-call-orphan Decayable, and a never-finished transcript sweep) into one
owner, one window: both tables only ever drop rows once they age past
:attr:`~models.transcript.Transcript.UNLINKED_GC_AFTER_DAYS` /
:attr:`~models.tool_call.ToolCall.ORPHAN_GC_AFTER_DAYS`.

``tool_calls.transcript_id`` is a foreign key with no ON DELETE cascade (a
schema constant, not an oversight — a cascade would need a SQLite table
rebuild and would change every other transcript-delete path's semantics), so
with ``PRAGMA foreign_keys=ON`` the doomed transcript ids must have their
anchored ``tool_calls`` rows cleared BEFORE the transcript rows themselves
can be deleted. :meth:`run` computes that id set once, then sweeps
``tool_calls`` then ``transcript`` off that SAME id set inside one
``Database.transaction()`` — children first, so the parent delete never
trips the FK. A separate, independent second step then reaps ``tool_calls``
orphans that pre-date this sweep (a transcript row removed by an older code
path, or any other legacy dangle): :meth:`~models.tool_call.ToolCall.decay`
runs outside the transaction since it has nothing to coordinate with — it
sees only rows this pass's transcript delete could not have produced (those
were already swept above).

Thin orchestrator (Rule-3 depth, mp-less): it owns no SQL of its own and
delegates every step to the owning model.
"""

from __future__ import annotations

from dataclasses import dataclass

from models.tool_call import ToolCall
from models.transcript import Transcript
from services.database import Database


@dataclass
class GarbageCollectionResult:
    """The three row-counts one garbage-collection pass reports."""

    transcripts_deleted: int
    tool_calls_swept: int
    tool_calls_orphans_reaped: int


class GarbageCollectorService:
    """Runs the hourly transcript + tool_calls retention sweep."""

    def run(self) -> GarbageCollectionResult:
        """Sweep unlinked transcript rows and their anchored tool_calls
        together, then separately reap tool_calls orphans that pre-date this
        sweep. Raises on any failure — no try/except here; the transaction
        rolls back atomically and the caller owns isolation and logging.

        The doomed transcript-id set (:meth:`~models.transcript.Transcript.
        unlinked_ids`) is computed once and both tables are cleared by that
        same set inside one transaction — ``tool_calls`` first (children),
        ``transcript`` second (parent) — so the FK from
        ``tool_calls.transcript_id`` (no cascade) never blocks the delete.
        An empty id set makes both calls clean no-ops.
        ``ToolCall.decay()`` then reaps any orphan the pre-clear above could
        not have produced (it already swept every child of this pass's
        doomed ids) — legacy dangles left by an older code path — outside
        the transaction since it has nothing to coordinate with."""
        with Database.transaction():
            ids = Transcript.unlinked_ids()
            tool_calls_swept = ToolCall.delete_by_transcripts(ids)
            transcripts_deleted = Transcript.delete_by_ids(ids)
        tool_calls_orphans_reaped = ToolCall.decay()
        return GarbageCollectionResult(
            transcripts_deleted=transcripts_deleted,
            tool_calls_swept=tool_calls_swept,
            tool_calls_orphans_reaped=tool_calls_orphans_reaped,
        )
