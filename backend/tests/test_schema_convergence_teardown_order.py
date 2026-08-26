"""Destructive convergence must drop stale triggers before the tables they feed.

Regression shape (observed live on a legacy-database upgrade): an upgrade
removes a virtual table that a trigger writes into.  Convergence used to drop
the virtual table first and the trigger last; once the virtual table was gone,
every later ``ALTER TABLE ... DROP COLUMN`` in the same pass failed with
``error in trigger <name>: no such table: main.<virtual>`` — SQLite re-parses
all trigger bodies when it rewrites a table, and a body naming a missing table
aborts the statement.  The stale column then survived the boot and only fell on
the NEXT boot, leaving the database half-converged with the failure visible
only as a WARNING.

These tests converge a real SQLite file seeded with exactly that legacy state —
a stale trigger feeding a stale virtual table, plus a stale column on a
surviving table — and require full convergence in ONE pass.  The only patch is
the DB path (mirrors test_policy_seed_converge.py).
"""
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

from services.database import Database
from services.file_mapper_service import FileMapperService
from services.schema_convergence_service import SchemaConvergenceService

pytestmark = pytest.mark.unit


@pytest.fixture()
def legacy_db(tmp_path: Path) -> Iterator[Path]:
    """A converged DB carrying pre-upgrade residue: a stale trigger →
    virtual-table cascade plus a stale column on a surviving table."""
    db_path = tmp_path / "converge.db"
    with patch.object(FileMapperService, "get_db_path", return_value=db_path):
        Database.close()
        SchemaConvergenceService().converge()  # boot: current schema in place
        conn = Database.conn()
        conn.executescript("""
            -- legacy sidecar lane, removed from schema.sql by an upgrade
            CREATE TABLE legacy_semantic (
                id INTEGER PRIMARY KEY,
                relates_to_table TEXT,
                related_to_id INTEGER
            );
            CREATE VIRTUAL TABLE legacy_semantic_vec USING fts5(payload);
            CREATE TRIGGER legacy_semantic_vec_sync
                AFTER DELETE ON legacy_semantic BEGIN
                DELETE FROM legacy_semantic_vec WHERE rowid = OLD.id;
            END;
            -- stale column on a table that survives the upgrade
            ALTER TABLE episodes ADD COLUMN legacy_queries TEXT;
        """)
        conn.commit()
        yield db_path
        Database.close()


def _residue_count(conn: sqlite3.Connection) -> int:
    """Rows left behind by the legacy lane: its table, the virtual table and
    that table's fts5 shadow children, and the cascade trigger."""
    return int(
        conn.execute(
            "SELECT count(*) FROM sqlite_master "
            "WHERE name = 'legacy_semantic' "
            "   OR name LIKE 'legacy_semantic_vec%' "
            "   OR name = 'legacy_semantic_vec_sync'"
        ).fetchone()[0]
    )


def test_stale_trigger_virtual_table_and_column_fall_in_one_boot(
    legacy_db: Path,
) -> None:
    with patch.object(FileMapperService, "get_db_path", return_value=legacy_db):
        SchemaConvergenceService().converge()  # the upgrade boot under test
        conn = Database.conn()

        assert _residue_count(conn) == 0, (
            "stale trigger/virtual-table lane survived the boot"
        )

        cols = {row[1] for row in conn.execute("PRAGMA table_info(episodes)")}
        assert "legacy_queries" not in cols, (
            "stale column survived the boot — the stale trigger was still "
            "present when the column drop ran"
        )
        # The pass pruned exactly the stale lane, nothing else.
        assert "gist" in cols, "convergence over-deleted live episodes columns"


def test_second_boot_is_a_no_op(legacy_db: Path) -> None:
    """Full convergence in one boot must not churn or double-drop on boot 2."""
    with patch.object(FileMapperService, "get_db_path", return_value=legacy_db):
        SchemaConvergenceService().converge()
        SchemaConvergenceService().converge()  # boot 2

        conn = Database.conn()
        assert _residue_count(conn) == 0
        cols = {row[1] for row in conn.execute("PRAGMA table_info(episodes)")}
        assert "legacy_queries" not in cols
