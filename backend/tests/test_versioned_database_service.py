"""Feature tests — versioned copy-forward provisioning of the main database.

Every test drives the real :meth:`VersionedDatabaseService.provision` against a
real temp data directory (``FileMapperService.get_db_path`` redirected, exactly
as ``conftest`` does), the real ``schema.sql``, and real SQLite with the real
sqlite-vec extension. Zero mocks: previous-release files are built by the same
``provision()`` call that built them in production, and the corruption case
overwrites a b-tree root page on disk so SQLite raises SQLITE_CORRUPT for real.

The contract under test: a release opens its own file, an existing file is never
touched again, a damaged source table costs only itself, and nothing older than
three files survives.
"""

import logging
import shutil
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
import sqlite_vec

from services.app_version import get_version
from services.file_mapper_service import FileMapperService
from services.versioned_database_service import VersionedDatabaseService

pytestmark = pytest.mark.unit

_GARBAGE_PAGE = b"\xde\xad\xbe\xef"
_STALE_WAL = b"frames belonging to the aside file"
_SQLITE_SIDECARS = ("-wal", "-shm")

# The running build's file name, read once at import — every test redirects
# ``get_db_path``, so asking it again mid-test would return whatever file the
# test last provisioned.
_TARGET_NAME = FileMapperService.get_db_path().name


def _connect(path: Path) -> sqlite3.Connection:
    """Open a database the way the app does — autocommit, sqlite-vec loaded."""
    conn = sqlite3.connect(str(path))
    conn.isolation_level = None
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    return conn


@pytest.fixture()
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """An empty data directory that ``get_db_path`` resolves into.

    The target is the running build's real file name, so version selection and
    retention see exactly the layout a live install has.
    """
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setattr(FileMapperService, "get_db_path", lambda *_: root / _TARGET_NAME)
    yield root


def _provision_at(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    """Provision a database file at *path* — how a previous release built its own."""
    monkeypatch.setattr(FileMapperService, "get_db_path", lambda *_: path)
    VersionedDatabaseService().provision()


def _target(data_dir: Path) -> Path:
    return data_dir / _TARGET_NAME


def _lineage(path: Path) -> list[tuple[str, str | None, str, str]]:
    conn = sqlite3.connect(str(path))
    try:
        return [
            (str(r[0]), r[1], str(r[2]), str(r[3]))
            for r in conn.execute(
                "SELECT version, source_file, completed_at, failed_tables "
                "FROM database_lineage ORDER BY version"
            )
        ]
    finally:
        conn.close()


class TestFreshInstall:

    def test_creates_the_schema_and_stamps_its_lineage(self, data_dir: Path) -> None:
        """No earlier database: the file is built from schema.sql and stamped."""
        VersionedDatabaseService().provision()

        target = _target(data_dir)
        assert target.exists(), "the running build's database file was not created"
        assert _lineage(target) == [(get_version(), None, _lineage(target)[0][2], "")], (
            "a fresh install must record one lineage row with no source and no failures"
        )
        conn = sqlite3.connect(str(target))
        try:
            assert conn.execute(
                "SELECT is_sensitive FROM settings WHERE key = 'api_key'"
            ).fetchone() == (1,), "schema.sql's seed pass did not run"
        finally:
            conn.close()

    def test_second_provision_leaves_the_file_untouched(self, data_dir: Path) -> None:
        """A file carrying the running build's lineage row is never rewritten."""
        VersionedDatabaseService().provision()
        target = _target(data_dir)
        before = target.read_bytes()

        VersionedDatabaseService().provision()

        assert target.read_bytes() == before, "a provisioned database was written to again"


class TestCopyForward:

    def test_legacy_database_rows_survive_a_column_set_that_moved(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """chalie.db is the source when no versioned file exists.

        Its ``settings`` table carries a column this build dropped and lacks one
        this build added: the row copies on the intersection, the extra column is
        left behind, and the new column takes its declared default.
        """
        legacy = data_dir / "chalie.db"
        _provision_at(monkeypatch, legacy)
        conn = _connect(legacy)
        try:
            # The pre-versioning shape: no lineage table, a column this build
            # dropped, and no `is_sensitive` (added by a later release).
            conn.execute("DROP TABLE database_lineage")
            conn.execute("ALTER TABLE settings ADD COLUMN retired_flag TEXT DEFAULT 'x'")
            conn.execute("ALTER TABLE settings DROP COLUMN is_sensitive")
            conn.execute(
                "INSERT INTO settings (key, value, retired_flag) VALUES ('user_row', 'kept', 'y')"
            )
        finally:
            conn.close()

        monkeypatch.setattr(FileMapperService, "get_db_path", lambda *_: _target(data_dir))
        VersionedDatabaseService().provision()

        conn = sqlite3.connect(str(_target(data_dir)))
        try:
            assert conn.execute(
                "SELECT value, is_sensitive FROM settings WHERE key = 'user_row'"
            ).fetchone() == ("kept", 0), (
                "the copied row lost its value or did not take the new column's default"
            )
            assert "retired_flag" not in {
                r[1] for r in conn.execute("PRAGMA table_info(settings)")
            }, "a column the source carried leaked into the new schema"
        finally:
            conn.close()
        assert _lineage(_target(data_dir))[0][1] == "chalie.db", (
            "the lineage row does not name chalie.db as the source"
        )

    def test_newest_older_file_wins_and_a_newer_one_is_ignored(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Source selection is by version, and never picks a newer file.

        A file at or above the running version belongs to a release this build
        knows nothing about — grafting its rows onto an older schema is how a
        downgrade eats data.
        """
        for version, marker in (("1.0.0", "oldest"), ("1.2.0", "newest-older"), ("9.9.9", "newer")):
            path = data_dir / f"chalie-{version}.sqlite"
            _provision_at(monkeypatch, path)
            conn = _connect(path)
            try:
                conn.execute("INSERT INTO emails_sent (key) VALUES (?)", (marker,))
            finally:
                conn.close()

        monkeypatch.setattr(FileMapperService, "get_db_path", lambda *_: _target(data_dir))
        VersionedDatabaseService().provision()

        conn = sqlite3.connect(str(_target(data_dir)))
        try:
            carried = {r[0] for r in conn.execute("SELECT key FROM emails_sent")}
        finally:
            conn.close()
        assert carried == {"oldest", "newest-older"}, (
            f"expected the 1.2.0 chain (which already carried 1.0.0's row), got {carried}"
        )
        assert _lineage(_target(data_dir))[-1][1] == "chalie-1.2.0.sqlite", (
            "the newest OLDER file was not chosen as the source"
        )

    def test_seeds_never_overwrite_a_copied_user_row(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """schema.sql's seeds run AFTER the copy, so a user's value wins."""
        source = data_dir / "chalie-1.2.0.sqlite"
        _provision_at(monkeypatch, source)
        conn = _connect(source)
        try:
            conn.execute("UPDATE settings SET value = 'the-users-key' WHERE key = 'api_key'")
        finally:
            conn.close()

        monkeypatch.setattr(FileMapperService, "get_db_path", lambda *_: _target(data_dir))
        VersionedDatabaseService().provision()

        conn = sqlite3.connect(str(_target(data_dir)))
        try:
            assert conn.execute(
                "SELECT value FROM settings WHERE key = 'api_key'"
            ).fetchone() == ("the-users-key",), "a seed default overwrote the carried-forward value"
        finally:
            conn.close()

    def test_deleted_providers_do_not_come_back(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A source still carrying ``is_active`` holds rows the user deleted."""
        source = data_dir / "chalie.db"
        _provision_at(monkeypatch, source)
        conn = _connect(source)
        try:
            conn.execute("ALTER TABLE providers ADD COLUMN is_active INTEGER DEFAULT 1")
            conn.execute(
                "INSERT INTO providers (name, platform, model, is_active) "
                "VALUES ('live', 'openai', 'gpt-4o', 1), ('deleted', 'openai', 'gpt-4o-mini', 0)"
            )
        finally:
            conn.close()

        monkeypatch.setattr(FileMapperService, "get_db_path", lambda *_: _target(data_dir))
        VersionedDatabaseService().provision()

        conn = sqlite3.connect(str(_target(data_dir)))
        try:
            assert [r[0] for r in conn.execute("SELECT name FROM providers")] == ["live"], (
                "a soft-deleted provider was resurrected as an active one"
            )
        finally:
            conn.close()

    def test_autoincrement_ids_continue_past_the_source_high_water_mark(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An id handed out by the previous release is never handed out again.

        Copying rows only lifts the counter to the highest id present, so a row
        allocated and then deleted would have its id reused — colliding with
        every reference the rest of the instance still holds to it.
        """
        source = data_dir / "chalie-1.2.0.sqlite"
        _provision_at(monkeypatch, source)
        conn = _connect(source)
        try:
            conn.execute("INSERT INTO emails_sent (key) VALUES ('a'), ('b'), ('c')")
            conn.execute("DELETE FROM emails_sent WHERE key = 'c'")
        finally:
            conn.close()

        monkeypatch.setattr(FileMapperService, "get_db_path", lambda *_: _target(data_dir))
        VersionedDatabaseService().provision()

        conn = sqlite3.connect(str(_target(data_dir)))
        try:
            conn.execute("INSERT INTO emails_sent (key) VALUES ('d')")
            conn.commit()
            assert conn.execute(
                "SELECT id FROM emails_sent WHERE key = 'd'"
            ).fetchone() == (4,), "the AUTOINCREMENT counter was not carried forward"
        finally:
            conn.close()

    def test_search_indexes_and_vectors_survive(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """FTS5 postings are rebuilt from the copied content table; vec0 rows copy.

        FTS5 shadow tables are never touched directly — writing behind the
        module's back corrupts the index — so the index has to come back from a
        real rebuild, and a vec0 row has to keep the rowid that joins it to its
        episode.
        """
        source = data_dir / "chalie-1.2.0.sqlite"
        _provision_at(monkeypatch, source)
        conn = _connect(source)
        try:
            conn.execute(
                "INSERT INTO episodes (id, gist, salience, channel) "
                "VALUES ('ep-1', 'kayaking down the estuary', 5, 'user')"
            )
            rowid = conn.execute("SELECT rowid FROM episodes WHERE id = 'ep-1'").fetchone()[0]
            conn.execute(
                "INSERT INTO episodes_vec (rowid, embedding) VALUES (?, ?)",
                (rowid, sqlite_vec.serialize_float32([0.25] * 768)),
            )
        finally:
            conn.close()

        monkeypatch.setattr(FileMapperService, "get_db_path", lambda *_: _target(data_dir))
        VersionedDatabaseService().provision()

        conn = _connect(_target(data_dir))
        try:
            hits = conn.execute(
                "SELECT e.id FROM episodes_fts f JOIN episodes e ON e.rowid = f.rowid "
                "WHERE episodes_fts MATCH 'kayaking'"
            ).fetchall()
            assert hits == [("ep-1",)], f"the FTS5 index was not rebuilt from the copy: {hits}"
            new_rowid = conn.execute("SELECT rowid FROM episodes WHERE id = 'ep-1'").fetchone()[0]
            assert conn.execute(
                "SELECT count(*) FROM episodes_vec WHERE rowid = ?", (new_rowid,)
            ).fetchone() == (1,), "the vec0 row lost the rowid that joins it to its episode"
        finally:
            conn.close()

    def test_rowids_survive_a_gap_so_vectors_still_point_at_their_rows(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A deleted episode leaves a rowid gap; every survivor keeps its rowid.

        episodes_vec joins on rowid and carries its own rowid across. Without an
        explicit rowid in the copy SQLite renumbers the survivors contiguously,
        so every episode after the gap lands on another episode's embedding and
        recall returns the wrong memory without a single error.
        """
        source = data_dir / "chalie-1.2.0.sqlite"
        _provision_at(monkeypatch, source)
        conn = _connect(source)
        try:
            conn.execute(
                "INSERT INTO episodes (id, gist, salience, channel) VALUES "
                "('ep-1', 'first', 5, 'user'), ('ep-2', 'second', 5, 'user'), "
                "('ep-3', 'third', 5, 'user')"
            )
            conn.execute("DELETE FROM episodes WHERE id = 'ep-2'")
            survivors = conn.execute("SELECT id, rowid FROM episodes ORDER BY rowid").fetchall()
            for _, rowid in survivors:
                conn.execute(
                    "INSERT INTO episodes_vec (rowid, embedding) VALUES (?, ?)",
                    (rowid, sqlite_vec.serialize_float32([float(rowid)] * 768)),
                )
        finally:
            conn.close()
        assert survivors == [("ep-1", 1), ("ep-3", 3)], "the fixture must leave a gap"

        monkeypatch.setattr(FileMapperService, "get_db_path", lambda *_: _target(data_dir))
        VersionedDatabaseService().provision()

        conn = _connect(_target(data_dir))
        try:
            assert conn.execute("SELECT id, rowid FROM episodes ORDER BY rowid").fetchall() == survivors
            joined = conn.execute(
                "SELECT v.rowid, e.id FROM episodes_vec v "
                "LEFT JOIN episodes e ON e.rowid = v.rowid ORDER BY v.rowid"
            ).fetchall()
            assert joined == [(1, "ep-1"), (3, "ep-3")], f"a vector lost its episode: {joined}"
        finally:
            conn.close()


class TestDamageContainment:

    def test_an_unreadable_table_costs_only_itself(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The whole reason this service exists.

        One table's b-tree root page is overwritten with garbage on disk, so
        SQLite raises SQLITE_CORRUPT the moment the copy walks it. The boot must
        still finish: the table is logged by name, recorded in the lineage row,
        and every other table copies intact.
        """
        source = data_dir / "chalie-1.2.0.sqlite"
        _provision_at(monkeypatch, source)
        conn = _connect(source)
        try:
            # DELETE journalling keeps every page in the main file, so the root
            # page overwritten below is the page SQLite actually reads back.
            conn.execute("PRAGMA journal_mode=DELETE")
            conn.execute(
                "INSERT INTO episodes (id, gist, salience, channel) "
                "VALUES ('ep-1', 'a memory that cannot be read', 5, 'user')"
            )
            conn.execute("INSERT INTO emails_sent (key) VALUES ('intact')")
            page_size = conn.execute("PRAGMA page_size").fetchone()[0]
            root_page = conn.execute(
                "SELECT rootpage FROM sqlite_master WHERE type='table' AND name='episodes'"
            ).fetchone()[0]
        finally:
            conn.close()
        with open(source, "r+b") as fh:
            fh.seek((root_page - 1) * page_size)
            fh.write(_GARBAGE_PAGE * (page_size // len(_GARBAGE_PAGE)))

        monkeypatch.setattr(FileMapperService, "get_db_path", lambda *_: _target(data_dir))
        with caplog.at_level(logging.ERROR, logger="services.versioned_database_service"):
            VersionedDatabaseService().provision()

        target = _target(data_dir)
        assert target.exists(), "a damaged source table aborted the whole provisioning"
        assert any("episodes" in record.message for record in caplog.records), (
            f"the unreadable table was not named in an ERROR log: {[r.message for r in caplog.records]}"
        )
        assert "episodes" in _lineage(target)[-1][3].split(","), (
            f"the lineage row does not record the failure: {_lineage(target)}"
        )
        conn = sqlite3.connect(str(target))
        try:
            assert conn.execute("SELECT count(*) FROM episodes").fetchone() == (0,), (
                "the unreadable table was left half-copied"
            )
            assert [r[0] for r in conn.execute("SELECT key FROM emails_sent")] == ["intact"], (
                "a table after the damaged one did not copy"
            )
            assert conn.execute(
                "SELECT count(*) FROM settings WHERE key = 'api_key'"
            ).fetchone() == (1,), "the seed pass did not run after the failure"
        finally:
            conn.close()

    def test_a_target_without_its_lineage_row_is_moved_aside_not_deleted(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A file this build did not provision is never unlinked.

        No lineage row usually means an earlier boot died mid-copy — but a
        snapshot restore also lands a database at this exact path, and that one
        can be the only copy of its data. It moves aside, byte for byte, with
        its sidecars, and the boot continues on a file it built itself.
        """
        source = data_dir / "chalie-1.2.0.sqlite"
        _provision_at(monkeypatch, source)
        conn = _connect(source)
        try:
            conn.execute("INSERT INTO emails_sent (key) VALUES ('from-the-source')")
        finally:
            conn.close()

        target = _target(data_dir)
        shutil.copy2(source, target)
        conn = _connect(target)
        try:
            conn.execute("DELETE FROM database_lineage")
            conn.execute("DELETE FROM emails_sent")
            conn.execute("INSERT INTO emails_sent (key) VALUES ('only-copy-of-this')")
        finally:
            conn.close()
        # Not a real WAL — a file that must travel with its database, whatever
        # it holds, because it can hold committed frames the database lacks.
        target.with_name(target.name + "-wal").write_bytes(_STALE_WAL)
        original = target.read_bytes()

        monkeypatch.setattr(FileMapperService, "get_db_path", lambda *_: target)
        VersionedDatabaseService().provision()

        asides = {p.name: p for p in data_dir.glob(f"{target.name}.incomplete-*")}
        db_aside = next(
            (p for name, p in asides.items() if not name.endswith(_SQLITE_SIDECARS)), None
        )
        assert db_aside is not None, f"the database was not moved aside, found {sorted(asides)}"
        assert db_aside.read_bytes() == original, "the aside file is not the original, byte for byte"
        assert asides[f"{db_aside.name}-wal"].read_bytes() == _STALE_WAL, (
            "the WAL did not travel with the database whose frames it holds"
        )

        assert _lineage(target)[-1][0] == get_version(), "the rebuilt target carries no lineage row"
        conn = sqlite3.connect(str(target))
        try:
            assert [r[0] for r in conn.execute("SELECT key FROM emails_sent")] == [
                "from-the-source"
            ], "the target was not rebuilt from the newest earlier database"
        finally:
            conn.close()


class TestRetention:

    def test_only_the_three_newest_files_survive(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Three files is the rollback window; the legacy name is the oldest.

        Pruning takes the sidecars with it — a stale ``-wal`` left beside a
        deleted database would replay onto whatever takes its name next.
        """
        legacy = data_dir / "chalie.db"
        _provision_at(monkeypatch, legacy)
        for suffix in ("-wal", "-shm"):
            legacy.with_name(legacy.name + suffix).write_bytes(b"stale frames")
        for version in ("1.0.0", "1.1.0", "1.2.0"):
            _provision_at(monkeypatch, data_dir / f"chalie-{version}.sqlite")

        monkeypatch.setattr(FileMapperService, "get_db_path", lambda *_: _target(data_dir))
        VersionedDatabaseService().provision()

        assert sorted(p.name for p in data_dir.iterdir()) == sorted(
            ["chalie-1.1.0.sqlite", "chalie-1.2.0.sqlite", _target(data_dir).name]
        ), f"retention left {sorted(p.name for p in data_dir.iterdir())}"
