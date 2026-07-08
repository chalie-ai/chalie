"""Migration 009 — split the conflated ``data_graph`` ``system`` kind.

``kind="system"`` historically meant two disjoint things at once: user/agent
knowledge memories saved through the memory tool (``source`` prefixed
``skill:memory:store:`` — recalled), and internal operational state — the
subconscious pattern/geo cursors, durable clocks, and user-summary prose — read
only by exact key and never recalled. The model layer now splits these into two
L3 verticals: ``SystemMemoryRow`` (KIND ``"system"``, searchable) and
``MachineStateRow`` (KIND ``"machine_state"``, operational).

This migration re-kinds the operational rows OUT of ``system`` into
``machine_state``. The discriminator is the source: machine rows carry internal
sources (``"user_summary"``, ``"fact_extraction"``, cursor sources, …) that
never start ``skill:memory:store:``, so the predicate moves ONLY machine rows and
leaves the memory-tool ``system`` memories in place. No sidecar cleanup is
needed — machine rows were excluded from FTS/vec indexing by ``should_fts_index``,
so they hold no ``data_graph_fts``/``*_vec`` postings and their
``search_queries`` is NULL.

Idempotent: re-running matches nothing (the moved rows are no longer ``system``).
The run-once sentinel gate lives in the boot caller (backend/run.py), not here.

Usage: `python backend/migrations/migration_009_system_kind_split.py`
"""

import os
import sqlite3
import sys

# Add backend/ to sys.path so services.* imports resolve when invoked standalone.
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from services.file_mapper_service import FileMapperService  # noqa: E402

_REKIND_SQL = (
    "UPDATE data_graph SET kind = 'machine_state' "
    "WHERE kind = 'system' "
    "AND (source IS NULL OR source NOT LIKE 'skill:memory:store:%')"
)


def apply(db_path: str) -> None:
    """Re-kind operational ``system`` rows to ``machine_state``.

    Idempotent: the moved rows stop matching ``kind = 'system'`` on any re-run.
    The memory-tool ``system`` memories (``source`` prefixed
    ``skill:memory:store:``) are left untouched.
    """
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        cursor = conn.execute(_REKIND_SQL)
        conn.commit()
        print(
            f"[migration_009] Re-kinded {cursor.rowcount} operational row(s) "
            f"from 'system' to 'machine_state' ({db_path})"
        )
    finally:
        conn.close()


if __name__ == "__main__":
    apply(str(FileMapperService.get_db_path()))
