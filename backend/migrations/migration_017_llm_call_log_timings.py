
"""Migration 017 — add ``llm_call_log.prefill_ms`` and ``llm_call_log.decode_ms`` columns.

The provider clients report the prefill/decode timing split of each LLM call,
but the ledger only stored the whole-call ``latency_ms``. These two nullable
REAL columns carry the per-phase timings end-to-end from the provider response
through to the telemetry table. A NULL value means "not measured" — never
coerced to 0, because 0ms would be a fake measurement.

Idempotent — the ADD COLUMN is guarded by a PRAGMA table_info presence check,
so fresh installs and re-runs are no-ops.

Usage: ``python backend/migrations/migration_017_llm_call_log_timings.py``
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


def needed(conn: sqlite3.Connection) -> bool:
    """Columns still absent? Only databases predating the timing split carry
    the gap."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(llm_call_log)")}
    return "prefill_ms" not in columns or "decode_ms" not in columns


def apply(db_path: str) -> None:
    """Add ``llm_call_log.prefill_ms`` and ``llm_call_log.decode_ms``.
    Idempotent — no-op when both columns already exist."""
    conn = connect(db_path)
    try:
        existing = {row[1] for row in conn.execute("PRAGMA table_info(llm_call_log)")}
        added = [c for c in ("prefill_ms", "decode_ms") if c not in existing]
        for column in added:
            conn.execute(f"ALTER TABLE llm_call_log ADD COLUMN {column} REAL")
        if added:
            conn.commit()
            print(f"[migration_017] added llm_call_log {', '.join(added)} (nullable REAL) ({db_path})")
        else:
            print(f"[migration_017] llm_call_log timing columns already present — no-op ({db_path})")
    finally:
        conn.close()


if __name__ == "__main__":
    apply(str(FileMapperService.get_db_path()))
