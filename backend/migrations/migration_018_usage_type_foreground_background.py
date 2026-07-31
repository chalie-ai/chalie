"""Migration 018 — rename ``llm_call_log.type`` spend-bucket values.

``llm_call_log.type`` records which spend bucket an LLM call counts against:
calls made in direct response to something the user is waiting on, versus
calls the subconscious/background machinery fires on its own schedule. That
axis was originally named after the two channel configs that set it
(``UserConfig.USAGE_TYPE = "chat"``, ``DiscoveryConfig.USAGE_TYPE =
"system"``), but those names collide in spirit with unrelated axes elsewhere
in the codebase (``PolicyChannel.CHAT``, ``ProviderType.CHAT``, the
``data_graph.kind`` "system" memories, the LLM provider "system" role) and
say nothing about what the bucket actually measures. The vocabulary is
renamed to what the column means: ``'foreground'`` (was ``'chat'``) and
``'background'`` (was ``'system'``).

This migration rewrites the historical ledger in place so existing spend
reporting keeps working under the new vocabulary. Older vintages of this
column used a different vocabulary still; the predicate targets only the two
known legacy values and leaves anything else untouched rather than assuming
a closed set.

Idempotent: re-running matches nothing once every legacy value has been
rewritten. The run-once ledger gate lives in ``migrations/runner.py``, not
here.

Usage: `python backend/migrations/migration_018_usage_type_foreground_background.py`
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

# The two legacy spend-bucket values being renamed. Shared by needed() and
# apply() so the check and the rewrite can never drift apart. Any other value
# already carried by the column (older vintages used a different vocabulary
# still) is left untouched.
_LEGACY_FILTER = "type IN ('chat', 'system')"

_RENAME_SQL = f"""
    UPDATE llm_call_log
    SET type = CASE type
        WHEN 'chat' THEN 'foreground'
        WHEN 'system' THEN 'background'
    END
    WHERE {_LEGACY_FILTER}
"""


def needed(conn: sqlite3.Connection) -> bool:
    """Rows still carrying a legacy spend-bucket value? The post-rename
    channel configs write 'foreground'/'background', so any match is
    pre-rename residue."""
    return conn.execute(
        f"SELECT 1 FROM llm_call_log WHERE {_LEGACY_FILTER} LIMIT 1"
    ).fetchone() is not None


def apply(db_path: str) -> None:
    """Rename legacy ``llm_call_log.type`` values: 'chat' -> 'foreground',
    'system' -> 'background'.

    Idempotent: the rewritten rows stop matching the legacy filter on any
    re-run. Rows already carrying an unrelated value are left untouched.
    """
    conn = connect(db_path)
    try:
        cursor = conn.execute(_RENAME_SQL)
        conn.commit()
        print(
            f"[migration_018] Renamed {cursor.rowcount} llm_call_log row(s) "
            f"from 'chat'/'system' to 'foreground'/'background' ({db_path})"
        )
    finally:
        conn.close()


if __name__ == "__main__":
    apply(str(FileMapperService.get_db_path()))
