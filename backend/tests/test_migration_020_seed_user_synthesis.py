"""Feature tests — migration 020 (seed ``user_synthesis`` v1 from the legacy
machine-state summary).

Recreates the upgrading database's shape: a live ``kind='machine_state'``
``data_graph`` row keyed ``user_summary`` and an empty ``user_synthesis``
table. The migration must copy the summary in as version 1, resolve a
duplicate-active collision to the newest row, refuse to touch a table that
already has rows, and no-op on fresh installs. Real migration module, real
SQLite, zero mocks.
"""

import sqlite3

import pytest

from migrations import migration_020_seed_user_synthesis as mig
from migrations.runner import run_all
from models.user_synthesis import UserSynthesisRow
from services.file_mapper_service import FileMapperService

pytestmark = pytest.mark.unit


def _db_path() -> str:
    return str(FileMapperService.get_db_path())


def _seed_legacy(db: sqlite3.Connection, value: str, *, active: int = 1) -> None:
    db.execute(
        "INSERT INTO data_graph (kind, key, value, active) "
        "VALUES ('machine_state', 'user_summary', ?, ?)",
        (value, active),
    )
    db.commit()


class TestNeeded:
    def test_true_with_legacy_summary_and_empty_table(self, db: sqlite3.Connection) -> None:
        _seed_legacy(db, "Short portrait.")
        assert mig.needed(db) is True

    def test_false_on_fresh_db(self, db: sqlite3.Connection) -> None:
        assert mig.needed(db) is False

    def test_false_once_user_synthesis_has_rows(self, db: sqlite3.Connection) -> None:
        """A database already writing syntheses is never re-seeded — the
        legacy row must not clobber newer versions."""
        _seed_legacy(db, "Old portrait.")
        UserSynthesisRow.write("Newer portrait.")
        assert mig.needed(db) is False

    def test_false_when_legacy_row_is_inactive(self, db: sqlite3.Connection) -> None:
        _seed_legacy(db, "Retired portrait.", active=0)
        assert mig.needed(db) is False

    def test_false_when_legacy_summary_is_blank(self, db: sqlite3.Connection) -> None:
        """A blank legacy value seeds nothing — the reader's fallback line is
        better than an empty version 1."""
        _seed_legacy(db, "   ")
        assert mig.needed(db) is False


class TestApply:
    def test_copies_the_summary_as_version_1(self, db: sqlite3.Connection) -> None:
        _seed_legacy(db, "Short portrait.")
        mig.apply(_db_path())

        row = UserSynthesisRow.latest()
        assert row is not None
        assert row.version == 1 and row.content == "Short portrait."

    def test_duplicate_active_rows_resolve_to_newest(self, db: sqlite3.Connection) -> None:
        """Mirrors the newest-active-by-key defense: an UPSERT collision that
        left two active rows seeds the newest one."""
        _seed_legacy(db, "Older duplicate.")
        _seed_legacy(db, "Newest duplicate.")
        mig.apply(_db_path())

        assert UserSynthesisRow.latest_content() == "Newest duplicate."

    def test_legacy_row_stays_inert(self, db: sqlite3.Connection) -> None:
        _seed_legacy(db, "Short portrait.")
        mig.apply(_db_path())

        kept = [
            tuple(row) for row in db.execute(
                "SELECT value FROM data_graph WHERE kind='machine_state' AND key='user_summary'"
            )
        ]
        assert kept == [("Short portrait.",)]


class TestIdempotence:
    def test_apply_twice_seeds_once(self, db: sqlite3.Connection) -> None:
        _seed_legacy(db, "Short portrait.")
        mig.apply(_db_path())
        assert mig.needed(db) is False  # second pass sees nothing to do
        mig.apply(_db_path())

        rows = [
            tuple(row) for row in db.execute("SELECT version, content FROM user_synthesis")
        ]
        assert rows == [(1, "Short portrait.")]


class TestRunnerIntegration:
    def test_runner_applies_and_ledgers_the_step(self, db: sqlite3.Connection) -> None:
        _seed_legacy(db, "Short portrait.")
        run_all(_db_path())

        outcome = db.execute(
            "SELECT outcome FROM schema_migrations WHERE key = 'user-synthesis-seed-v1'"
        ).fetchone()
        assert outcome is not None and outcome[0] == "applied"
        assert UserSynthesisRow.latest_content() == "Short portrait."

    def test_runner_noops_on_fresh_db(self, db: sqlite3.Connection) -> None:
        run_all(_db_path())
        outcome = db.execute(
            "SELECT outcome FROM schema_migrations WHERE key = 'user-synthesis-seed-v1'"
        ).fetchone()
        assert outcome is not None and outcome[0] == "noop"
