"""``MachineStateRow`` — the operational ``machine_state`` vertical of
``data_graph``: state read and written by exact key, never by semantic search.
The user-summary prose the prompt builder reads every turn, the subconscious
worker's pattern/geo cursors, and durable clocks all live here. Because these
rows are read only by exact key, ``should_fts_index`` excludes the
``machine_state`` kind from FTS/vec indexing entirely (dead weight otherwise).
Adds only the cursor read :meth:`newest_active_by_key`; the exact-key
store/forget/supersede lifecycle lives on :class:`~models.exact_key.ExactKeyRow`."""

from __future__ import annotations

from typing import ClassVar, Self

from models.exact_key import ExactKeyRow


class MachineStateRow(ExactKeyRow):
    KIND: ClassVar[str] = "machine_state"

    @classmethod
    def newest_active_by_key(cls, key: str) -> Self | None:
        """The newest *live* row for this kind's exact ``key`` — the shared live
        predicate (``active = 1 AND deleted_at IS NULL``) plus a deterministic
        ``ORDER BY id DESC`` so that if more than one active row exists for the
        key (a historical or concurrent UPSERT collision in the store path) the
        newest deterministically wins, rather than SQLite's
        implementation-defined order. Distinct from :meth:`active_by_key` — the
        store lookup, which is ``active``-only and unordered — this is the read
        for durable cursors that must survive duplicate-active rows."""
        return cls.live().filter("key", key).order_by("id DESC").first()
