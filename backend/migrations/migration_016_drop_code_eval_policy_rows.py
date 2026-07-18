"""Migration 016 — delete seeded ``code_eval`` policy rows.

The ``code_eval`` ability was replaced by ``code_agent`` (a delegate tool with
its own 12-tool sandboxed file workspace); ``policy_defaults.json`` no longer
seeds ``code_eval`` rows, but a database that booted before this change still
carries them (one row per channel: chat, external_agent, subconscious). A
stale row is otherwise harmless — the gate simply never looks up a permission
nothing calls any more — but it clutters the Brain policy surface with a
control for an ability that no longer exists, so this cleans it up once.

Idempotent — ``needed()`` is False once no ``code_eval`` row remains, so fresh
installs and re-runs are no-ops.

Usage: ``python backend/migrations/migration_016_drop_code_eval_policy_rows.py``
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

_STALE_PERMISSION = "code_eval"


def needed(conn: sqlite3.Connection) -> bool:
    """True when at least one ``policy`` row still targets the retired
    ``code_eval`` permission."""
    return (
        conn.execute(
            "SELECT 1 FROM policy WHERE permission = ? LIMIT 1", (_STALE_PERMISSION,)
        ).fetchone()
        is not None
    )


def apply(db_path: str) -> None:
    """Delete every ``policy`` row whose permission is the retired
    ``code_eval`` tool. Never touches ``code_agent`` rows — those are seeded
    fresh by ``policy_defaults.json`` and are a distinct permission."""
    conn = connect(db_path)
    try:
        if not needed(conn):
            print(f"[migration_016] no code_eval policy rows — no-op ({db_path})")
            return

        deleted = conn.execute(
            "DELETE FROM policy WHERE permission = ?", (_STALE_PERMISSION,)
        ).rowcount
        conn.commit()
        print(f"[migration_016] deleted {deleted} code_eval policy row(s) ({db_path})")
    finally:
        conn.close()


if __name__ == "__main__":
    apply(str(FileMapperService.get_db_path()))
