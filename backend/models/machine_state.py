"""``MachineStateRow`` — the operational ``machine_state`` vertical of
``data_graph``: state read and written by exact key, never by semantic search.
The stable background-thread turn ids (discovery, memory hygiene) and durable
clocks all live here. Because these
rows are read only by exact key, the ``Searchable`` trait is declared off
(``__search__ = None``), excluding the ``machine_state`` kind from FTS/vec
indexing entirely (dead weight otherwise). Adds only the cursor read
:meth:`newest_active_by_key`; the exact-key store/forget/supersede lifecycle
lives on :class:`~models.exact_key.ExactKeyRow`."""

from __future__ import annotations

from typing import ClassVar, Self

from contracts.search_config import SearchConfig
from models.exact_key import ExactKeyRow


class MachineStateRow(ExactKeyRow):
    KIND: ClassVar[str] = "machine_state"

    #: Non-searchable: operational state read only by exact key. Overrides the
    #: base ``data_graph`` config to ``None`` — presence of a config IS
    #: enablement, so ``None`` keeps these rows out of the search-index pipeline.
    __search__: ClassVar[SearchConfig | None] = None

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
