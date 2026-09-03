"""Feature tests — startup migration runner (migrations/runner.py).

Real provisioned database files (the ``db`` fixture), the real migration modules,
zero mocks. Covers the three upgrade postures the runner must resolve — fresh
install (every step no-ops), sentinel-era upgrade (ledger import without
execution), legacy-shape upgrade (apply fires) — plus run-once semantics and
the lost-ledger safety net that keeps the destructive scheduler steps from ever
re-running blind.
"""

import sqlite3
from pathlib import Path

import pytest

from migrations.runner import _STEPS, run_all
from services.file_mapper_service import FileMapperService

pytestmark = pytest.mark.unit

_ALL_KEYS = {key for key, _ in _STEPS}


def _run() -> None:
    run_all(str(FileMapperService.get_db_path()))


def _ledger(db: sqlite3.Connection) -> dict[str, str]:
    return {
        str(row[0]): str(row[1])
        for row in db.execute("SELECT key, outcome FROM schema_migrations")
    }


class TestFreshDatabase:
    def test_all_steps_record_noop_without_executing(
        self, db: sqlite3.Connection, tmp_path: Path
    ) -> None:
        _run()
        ledger = _ledger(db)
        assert set(ledger) == _ALL_KEYS
        assert set(ledger.values()) == {"noop"}
        # Every resolution re-arms the wrapper-era downgrade gate: the sentinel
        # file each pre-ledger boot wrapper checks must exist afterwards — the
        # destructive scheduler wrappers gate on nothing else. migration_008's
        # bumped key must also satisfy the old v1 stem.
        for key in _ALL_KEYS:
            assert (tmp_path / f".{key}.done").exists()
        assert (tmp_path / ".scheduled-items-prompt-only-v1.done").exists()

    def test_second_run_changes_nothing(self, db: sqlite3.Connection) -> None:
        _run()
        before = db.execute(
            "SELECT key, outcome, applied_at FROM schema_migrations ORDER BY key"
        ).fetchall()
        _run()
        after = db.execute(
            "SELECT key, outcome, applied_at FROM schema_migrations ORDER BY key"
        ).fetchall()
        assert after == before


class TestSentinelEraUpgrade:
    def test_sentinel_imports_without_executing(
        self, db: sqlite3.Connection, tmp_path: Path
    ) -> None:
        # A pre-ledger database: legacy compaction row present, but the old boot
        # wrapper's sentinel file says migration_002 already ran. The runner must
        # import that state — record it, keep the file, and NOT purge the row.
        db.execute(
            "INSERT INTO transcript (channel, role, content) "
            "VALUES ('general', 'compaction', 'legacy checkpoint')"
        )
        db.commit()
        sentinel = tmp_path / ".compactions-table-migration-v1.done"
        sentinel.write_text("done")

        _run()

        assert _ledger(db)["compactions-table-migration-v1"] == "sentinel"
        assert sentinel.exists()
        survivors = db.execute(
            "SELECT COUNT(*) FROM transcript WHERE role = 'compaction'"
        ).fetchone()[0]
        assert survivors == 1


class TestLegacyShapeUpgrade:
    def test_needed_steps_apply(self, db: sqlite3.Connection) -> None:
        # Recreate the pre-migration vintages each step exists to fix.
        db.execute("ALTER TABLE tool_calls ADD COLUMN invoked_by TEXT")
        db.execute(
            "INSERT INTO transcript (channel, role, content) "
            "VALUES ('general', 'compaction', 'legacy checkpoint')"
        )
        db.execute(
            "INSERT INTO data_graph (kind, key, value, source) "
            "VALUES ('system', 'user_summary', 'prose', 'user_summary')"
        )
        db.execute(
            "INSERT INTO episodes (id, gist, salience, channel) "
            "VALUES ('ep-legacy', 'old episode', 5, 'general')"
        )
        db.commit()

        _run()

        ledger = _ledger(db)
        assert ledger["compactions-table-migration-v1"] == "applied"
        assert ledger["tool-calls-drop-invoked-by-v1"] == "applied"
        assert ledger["system-kind-split-migration-v1"] == "applied"
        assert ledger["search-indexed-at-backfill-v1"] == "applied"

        cols = {row[1] for row in db.execute("PRAGMA table_info(tool_calls)")}
        assert "invoked_by" not in cols
        assert (
            db.execute(
                "SELECT COUNT(*) FROM transcript WHERE role = 'compaction'"
            ).fetchone()[0]
            == 0
        )
        assert (
            db.execute(
                "SELECT kind FROM data_graph WHERE key = 'user_summary'"
            ).fetchone()[0]
            == "machine_state"
        )
        assert (
            db.execute(
                "SELECT indexed_at FROM episodes WHERE id = 'ep-legacy'"
            ).fetchone()[0]
            is not None
        )
        assert (
            db.execute(
                "SELECT indexed_at FROM data_graph WHERE key = 'user_summary'"
            ).fetchone()[0]
            is not None
        )

    def test_applied_step_never_reruns(self, db: sqlite3.Connection) -> None:
        db.execute("ALTER TABLE tool_calls ADD COLUMN invoked_by TEXT")
        db.commit()
        _run()
        # The ledger row now guards the step: a re-created column must survive
        # the second run untouched.
        db.execute("ALTER TABLE tool_calls ADD COLUMN invoked_by TEXT")
        db.commit()
        _run()
        cols = {row[1] for row in db.execute("PRAGMA table_info(tool_calls)")}
        assert "invoked_by" in cols


class TestLostLedgerSafety:
    def test_current_shape_data_survives_ledger_loss(
        self, db: sqlite3.Connection, tmp_path: Path
    ) -> None:
        # A database with no run tracking at all (restored backup, wiped ledger,
        # no sentinel files) but rows in the CURRENT scheduler shape: the
        # destructive wipe (007) and rebuild (008) must answer needed()=False
        # instead of destroying live schedules.
        db.execute(
            "INSERT INTO scheduled_items (message, start_at) "
            "VALUES ('standup ping', '2026-01-01T09:00:00+00:00')"
        )
        db.commit()
        _run()
        db.execute("DELETE FROM schema_migrations")
        db.commit()
        for sentinel in tmp_path.glob(".*.done"):
            sentinel.unlink()

        _run()

        assert db.execute("SELECT COUNT(*) FROM scheduled_items").fetchone()[0] == 1
        ledger = _ledger(db)
        assert ledger["scheduled-items-cron-wipe-v1"] == "noop"
        assert ledger["scheduled-items-prompt-only-v2"] == "noop"
