"""Migration 003 — drop the wrapper_tokens authz columns.

The `capabilities` and `permissions` columns existed solely to gate the
`/api/query` and `/api/signals` REST endpoints. Both endpoints are gone, so
nothing reads either column — wrapper tokens are now opaque bearer credentials
(authN only). `schema.sql` no longer declares the columns, so
SchemaConvergenceService drops them automatically on the next boot
(`CHALIE_SCHEMA_ALLOW_DESTRUCTIVE=1`, the default). This file is the
standalone idempotent script for operators who want to apply the drop manually.

No backfill is needed — the dropped data gated removed endpoints and has no
remaining consumer.

Usage: `python backend/migrations/migration_002_drop_wrapper_authz_columns.py [DB_PATH]`
"""

import os
import sqlite3
import sys

# Add backend/ to sys.path so services.* imports resolve when invoked standalone.
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from services.file_mapper_service import FileMapperService  # noqa: E402

_STALE_COLUMNS = ("capabilities", "permissions")


def apply(db_path: str) -> None:
    """Drop the stale authz columns. Idempotent — safe to run more than once."""
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        live = {row[1] for row in conn.execute("PRAGMA table_info(wrapper_tokens)")}
        for col in _STALE_COLUMNS:
            if col in live:
                conn.execute(f"ALTER TABLE wrapper_tokens DROP COLUMN {col}")
                print(f"[migration_002] DROP COLUMN wrapper_tokens.{col} — done ({db_path})")
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    _path = sys.argv[1] if len(sys.argv) > 1 else str(FileMapperService.get_db_path())
    apply(_path)
