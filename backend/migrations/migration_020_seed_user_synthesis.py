"""Migration 020 — seed ``user_synthesis`` v1 from the legacy machine-state summary.

The user synthesis used to live as one ``kind='machine_state'`` ``data_graph``
row keyed ``user_summary``; the versioned ``user_synthesis`` table replaces it
as the prompt spine's source. Databases upgrading from the machine-state era
carry a live summary the new table doesn't have yet — this migration copies it
in as version 1 so the first boot on the new code serves the same profile the
last boot of the old code did. The legacy row itself stays inert (nothing
reads it any more). A plain INSERT with no search-index teardown — the target
table is not indexed.

Idempotent — ``needed()`` is False once ``user_synthesis`` has any row, and
False on fresh installs (no legacy summary to copy), so re-runs are no-ops.

Usage: ``python backend/migrations/migration_020_seed_user_synthesis.py``
"""

import os
import sqlite3
import sys

# Add backend/ to sys.path so services.* imports resolve when invoked standalone.
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from migrations.connection import connect  # noqa: E402
from services.file_mapper_service import FileMapperService  # noqa: E402

# The newest live legacy row — mirrors the retired reader's ``active = 1``
# scope, resolved deterministically to the highest id should a historical
# UPSERT collision have left more than one active row.
_LEGACY_SUMMARY_SQL = (
    "SELECT value FROM data_graph "
    "WHERE kind = 'machine_state' AND key = 'user_summary' AND active = 1 "
    "ORDER BY id DESC LIMIT 1"
)


def needed(conn: sqlite3.Connection) -> bool:
    """True when ``user_synthesis`` is still empty AND a live legacy summary
    with non-blank prose exists to copy. Fresh installs (no legacy row) and
    already-seeded or already-writing databases answer False."""
    if conn.execute("SELECT 1 FROM user_synthesis LIMIT 1").fetchone() is not None:
        return False
    row = conn.execute(_LEGACY_SUMMARY_SQL).fetchone()
    return row is not None and bool((row[0] or "").strip())


def apply(db_path: str) -> None:
    """Copy the live legacy summary into ``user_synthesis`` as version 1.
    Idempotent: guarded by the same emptiness check as ``needed()``, so a
    standalone re-run never double-seeds."""
    conn = connect(db_path)
    try:
        if not needed(conn):
            print(f"[migration_020] Nothing to seed ({db_path})")
            return
        value = conn.execute(_LEGACY_SUMMARY_SQL).fetchone()[0]
        conn.execute(
            "INSERT INTO user_synthesis (version, content) VALUES (1, ?)", (value,)
        )
        conn.commit()
        print(f"[migration_020] Seeded user_synthesis v1 from the legacy summary ({db_path})")
    finally:
        conn.close()


if __name__ == "__main__":
    apply(str(FileMapperService.get_db_path()))
