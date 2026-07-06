"""Migration 006 — add transcript.settled column and backfill.

``settled`` is a positive flag: 1 on assistant rows that carry no model-driven
tool (the turn's settle0), 0 on every other row. Written as 1 by the write path;
``ToolCallService.start()`` demotes to 0 when a settling tool is recorded. ``schema.sql``
declares the column; SchemaConvergenceService adds it automatically on the next
boot. This file is the standalone idempotent script for operators applying the
change manually.

Backfill: mark settled=1 on every assistant row that the old NOT-EXISTS predicate
would have considered settled — i.e. rows with no tool_calls entry carrying a
non-internal tool name.

Usage: `python backend/migrations/migration_006_transcript_settled.py`
"""

import os
import sqlite3
import sys

# Add backend/ to sys.path so services.* imports resolve when invoked standalone.
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from services.file_mapper_service import FileMapperService  # noqa: E402

_NON_SETTLING = ("chat_history_compactor", "thinking")


def apply(db_path: str) -> None:
    """Add and backfill transcript.settled. Idempotent — safe to re-run."""
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        live = {row[1] for row in conn.execute("PRAGMA table_info(transcript)")}
        if "settled" not in live:
            conn.execute(
                "ALTER TABLE transcript ADD COLUMN settled INTEGER NOT NULL DEFAULT 0"
            )
            print(f"[migration_006] ADD COLUMN transcript.settled — done ({db_path})")

        # Backfill: mark settled=1 on assistant rows that were settle0 under the
        # old predicate — those with no tool_calls row for a non-internal tool.
        ph = ",".join("?" * len(_NON_SETTLING))
        conn.execute(
            f"""
            UPDATE transcript SET settled = 1
            WHERE role = 'assistant'
              AND id NOT IN (
                  SELECT DISTINCT t.id
                  FROM transcript t
                  JOIN tool_calls tc ON tc.transcript_id = t.id
                  WHERE tc.tool_name NOT IN ({ph})
              )
            """,
            _NON_SETTLING,
        )
        print(f"[migration_006] backfilled transcript.settled=1 — done ({db_path})")
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    apply(str(FileMapperService.get_db_path()))
