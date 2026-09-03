# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Feature tests: a snapshot restores into the versioned-database layout.

``services/snapshot_service.py::SnapshotService`` and
``services/versioned_database_service.py::VersionedDatabaseService`` split the
main database's identity in two: the running build only ever OPENS
``data/chalie-<VERSION>.sqlite`` (or the pre-versioning ``data/chalie.db``),
but a restore does not land the snapshot's database there directly — it lands
under the NAME OF THE RELEASE THAT TOOK IT
(``SnapshotService._restored_database_path``) and leaves the rest to the
ordinary boot. This is deliberate: ``VersionedDatabaseService.provision()``
treats a file already standing at the running path with no lineage row for
the running version as an unfinished build and silently moves it aside — so a
restore that always landed at ``get_db_path()`` would have an older release's
data quietly discarded on the very next boot.

These tests drive the REAL ``SnapshotService`` and the REAL
``VersionedDatabaseService`` against REAL SQLite databases and a REAL zip on
disk — export, mutate or corrupt the live instance, ``stage_import``,
``apply_pending``, then assert observable, on-disk state. No mock, patch, or
stub touches either service's own logic.

Every ``FileMapperService`` path either service can reach is redirected under
``tmp_path`` before a line of production code runs, extending the
conftest-blessed pattern from ``test_snapshot_docs_roundtrip.py`` to every
constant these tests exercise:

* ``_DATA_DIR`` — so every database, the ``.pending-restore`` marker and the
  ``.pre-restore-*`` / ``.restore-failed-*`` asides land under ``tmp_path``.
* ``_SECURE_DIR`` / ``_DOCUMENTS_DIR`` / ``_USER_SKILLS_DIR`` — none of the
  three is derived from ``_DATA_DIR`` at call time (each is a separate
  class-load-time constant), so each needs its own redirect. Left alone,
  ``_SECURE_DIR`` would keep pointing at this checkout's real vault
  directory — harmless here only because that directory happens not to
  exist yet; redirecting it removes that dependency on ambient state.
* ``get_version_path`` / ``get_skills_db_path`` / ``get_mcp_tools_db_path`` —
  VERSION lives at the repo root, ``skills.sqlite`` is a real checked-in
  asset, and ``mcp_tools.sqlite`` is this checkout's real gitignored runtime
  index; all three are pointed at paths that do not exist so that export's
  own ``if not src.exists(): continue`` skips them cleanly, and restore never
  reaches for a real, checked-in, or otherwise-owned file.

``get_db_path`` is deliberately left un-redirected: its real body resolves
through ``get_version()`` (module-level, not a ``FileMapperService`` method)
and the redirected ``get_version_path`` / ``_DATA_DIR``, so it keeps routing
through the exact production version-to-filename logic this suite exists to
prove.
"""

import shutil
import sqlite3
import zipfile
from pathlib import Path

import pytest

from exceptions import SnapshotError
from services.file_mapper_service import FileMapperService
from services.snapshot_service import SnapshotService
from services.versioned_database_service import VersionedDatabaseService

pytestmark = pytest.mark.unit

_RUNNING_VERSION = "1.3.0-beta"
_OLDER_VERSION = "1.2.0"
_PROBE_KEY = "snapshot_probe"


@pytest.fixture()
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An isolated instance layout under tmp_path — see the module docstring
    for why each redirected path is needed."""
    data = tmp_path / "data"
    data.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(FileMapperService, "_DATA_DIR", data)
    monkeypatch.setattr(FileMapperService, "_SECURE_DIR", data / "secure")
    monkeypatch.setattr(FileMapperService, "_DOCUMENTS_DIR", data / "docs")
    monkeypatch.setattr(FileMapperService, "_USER_SKILLS_DIR", data / "skills" / "user")
    monkeypatch.setattr(FileMapperService, "get_version_path", lambda *_: data / "VERSION")
    monkeypatch.setattr(FileMapperService, "get_skills_db_path", lambda *_: data / "unused-skills.sqlite")
    monkeypatch.setattr(
        FileMapperService, "get_mcp_tools_db_path", lambda *_: data / "unused-mcp-tools.sqlite"
    )
    return data


def _set_version(version: str) -> None:
    """Write *version* to this test's isolated VERSION file."""
    FileMapperService.get_version_path().write_text(version + "\n", encoding="utf-8")


def _tiny_sqlite(path: Path) -> None:
    """A minimal, real, valid SQLite file — enough to exist as a database
    artifact without the full application schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE placeholder (id INTEGER PRIMARY KEY)")
        conn.commit()
    finally:
        conn.close()


def _write_probe(db_path: Path, value: str) -> None:
    """Upsert a distinctive ``settings`` row so a restore's before/after
    content is unambiguous — proves a restore actually rewrote the file's
    data, not merely that a file with the right name exists."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (_PROBE_KEY, value),
        )
        conn.commit()
    finally:
        conn.close()


def _probe_value(db_path: Path) -> str | None:
    """Read back the probe row written by :func:`_write_probe`."""
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT value FROM settings WHERE key = ?", (_PROBE_KEY,)
        ).fetchone()
        return None if row is None else str(row[0])
    finally:
        conn.close()


def _main_db_names(data_dir: Path) -> list[str]:
    """Every main-database file name standing directly in *data_dir* — the
    legacy name plus every versioned name — mirroring what
    ``SnapshotService._live_main_databases()`` considers a main database."""
    legacy_name = FileMapperService.get_legacy_db_path().name
    return sorted(
        path.name
        for path in data_dir.glob("*")
        if path.name == legacy_name or FileMapperService.version_from_db_path(path) is not None
    )


class TestRestoredDatabaseLanding:
    """Where SnapshotService lands a restored main database — the core of the
    versioned-restore fix: never unconditionally ``get_db_path()``."""

    def test_same_version_snapshot_restores_at_the_running_path(self, data_dir: Path) -> None:
        """A snapshot taken by THIS release restores at get_db_path() itself,
        carrying the snapshot's own data, not whatever was written after it
        was taken.

        Would catch: apply_pending() no-op'ing a same-version restore, or
        landing it under the legacy/versioned name instead of the running
        path the very same release already owns.
        """
        _set_version(_RUNNING_VERSION)
        VersionedDatabaseService().provision()
        live = FileMapperService.get_db_path()
        _write_probe(live, "in-the-snapshot")

        svc = SnapshotService()
        zip_path = svc.export(password=None)
        try:
            _write_probe(live, "written-after-the-snapshot")

            svc.stage_import(zip_path, password=None)
            SnapshotService.apply_pending()

            with zipfile.ZipFile(zip_path) as zf:
                snapshot_bytes = zf.read(f"chalie_db/{live.name}")
            assert live.read_bytes() == snapshot_bytes
            assert _probe_value(live) == "in-the-snapshot"
        finally:
            zip_path.unlink(missing_ok=True)

    def test_older_version_snapshot_lands_under_its_own_name_then_copies_forward(
        self, data_dir: Path
    ) -> None:
        """An older release's snapshot must never land at get_db_path() (the
        RUNNING release's own file): VersionedDatabaseService treats a file
        standing there with no lineage row for the running version as an
        unfinished build and rebuilds over it, discarding whatever a
        pre-fix restore had silently dropped there. Landed under its own
        versioned name instead, the very next provision() copies it forward.

        Would catch: a regression to always landing a restore at
        get_db_path() regardless of the snapshot's own version — which makes
        an older snapshot's data vanish on the next boot.
        """
        _set_version(_OLDER_VERSION)
        VersionedDatabaseService().provision()
        older = FileMapperService.get_db_path()
        _write_probe(older, "from-the-older-snapshot")

        svc = SnapshotService()
        zip_path = svc.export(password=None)
        try:
            _write_probe(older, "changed-after-the-snapshot")
            _set_version(_RUNNING_VERSION)
            VersionedDatabaseService().provision()
            running = FileMapperService.get_db_path()

            svc.stage_import(zip_path, password=None)
            SnapshotService.apply_pending()

            assert older.exists()
            assert _probe_value(older) == "from-the-older-snapshot"
            assert not running.exists()

            VersionedDatabaseService().provision()
            assert _probe_value(running) == "from-the-older-snapshot"
        finally:
            zip_path.unlink(missing_ok=True)

    def test_version_less_snapshot_lands_as_the_legacy_database(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A pre-versioning snapshot (no VERSION artifact at all) restores at
        get_legacy_db_path() (chalie.db).

        provision() calls get_version() unconditionally, so the legacy-shaped
        source database has to be built WITH a VERSION file in place; only
        afterward, right before export, is the VERSION file removed — the
        only way to reach a genuinely VERSION-less snapshot.

        Would catch: routing a VERSION-less snapshot through
        get_versioned_db_path(None) (a crash), or defaulting it to
        get_db_path() (the running path) instead of the legacy name.
        """
        original_get_db_path = FileMapperService.get_db_path
        monkeypatch.setattr(
            FileMapperService, "get_db_path", lambda *_: FileMapperService.get_legacy_db_path()
        )
        _set_version(_OLDER_VERSION)
        VersionedDatabaseService().provision()
        legacy = FileMapperService.get_legacy_db_path()
        _write_probe(legacy, "pre-versioning-data")
        FileMapperService.get_version_path().unlink()

        svc = SnapshotService()
        zip_path = svc.export(password=None)
        try:
            with zipfile.ZipFile(zip_path) as zf:
                assert not any(name.startswith("version/") for name in zf.namelist())

            monkeypatch.setattr(FileMapperService, "get_db_path", original_get_db_path)
            _set_version(_RUNNING_VERSION)
            legacy.unlink()
            VersionedDatabaseService().provision()
            running = FileMapperService.get_db_path()

            svc.stage_import(zip_path, password=None)
            SnapshotService.apply_pending()

            assert legacy.exists()
            assert _probe_value(legacy) == "pre-versioning-data"
            assert not running.exists()

            VersionedDatabaseService().provision()
            assert _probe_value(running) == "pre-versioning-data"
        finally:
            zip_path.unlink(missing_ok=True)


class TestVersionMarkerImmutability:
    """The VERSION artifact is read for the guard and the landing name, and
    never written back — see SnapshotService._is_restorable."""

    def test_restore_never_rewrites_the_running_version_file(self, data_dir: Path) -> None:
        """Restoring a snapshot exported under a DIFFERENT version must not
        change one byte of the running build's own VERSION file, even though
        the zip carries its own version/VERSION member.

        Would catch: _is_restorable losing its VERSION exclusion (or a future
        refactor routing the VERSION artifact through the same swap as any
        other single-file KIND), which would overwrite the running process's
        own version identity with the snapshot's — corrupting the target
        every subsequent provision() call resolves to.
        """
        _set_version(_OLDER_VERSION)
        VersionedDatabaseService().provision()

        svc = SnapshotService()
        zip_path = svc.export(password=None)
        try:
            with zipfile.ZipFile(zip_path) as zf:
                assert "version/VERSION" in zf.namelist()

            _set_version(_RUNNING_VERSION)
            VersionedDatabaseService().provision()
            version_path = FileMapperService.get_version_path()
            before = version_path.read_text(encoding="utf-8")
            assert before.strip() == _RUNNING_VERSION

            svc.stage_import(zip_path, password=None)
            SnapshotService.apply_pending()

            assert version_path.read_text(encoding="utf-8") == before
        finally:
            zip_path.unlink(missing_ok=True)


class TestMainDatabaseSwap:
    """The success and failure paths of clearing every live main database
    before landing a restored one."""

    def test_every_pre_existing_main_database_moves_into_the_aside_directory(
        self, data_dir: Path
    ) -> None:
        """Every main-database file in data/ — the one about to be
        overwritten included, plus unrelated sibling releases and their WAL
        sidecar — moves into .pre-restore-<ts>/chalie_db/, byte-identical,
        with none left standing in data/ afterward.

        VersionedDatabaseService.provision() always builds the running
        release's file from the NEWEST database strictly older than it, so
        any main database left behind that is newer than the restored one
        would be copied forward in its place and the restore would be
        invisible.

        Would catch: clearing only the file the snapshot's database is about
        to land on, leaving sibling chalie-*.sqlite files standing that a
        later provision() could pick up instead of the restored data.
        """
        _set_version(_RUNNING_VERSION)
        VersionedDatabaseService().provision()
        live = FileMapperService.get_db_path()
        _write_probe(live, "in-the-snapshot")

        svc = SnapshotService()
        zip_path = svc.export(password=None)
        try:
            _tiny_sqlite(data_dir / "chalie.db")
            _tiny_sqlite(data_dir / "chalie-1.1.0.sqlite")
            _tiny_sqlite(data_dir / "chalie-1.2.0.sqlite")
            (data_dir / "chalie-1.2.0.sqlite-wal").write_bytes(b"stale frames of the 1.2.0 database")
            before = {p.name: p.read_bytes() for p in data_dir.glob("chalie*")}

            svc.stage_import(zip_path, password=None)
            SnapshotService.apply_pending()

            aside = next(data_dir.glob(".pre-restore-*")) / "chalie_db"
            assert sorted(p.name for p in aside.glob("*")) == sorted(before)
            for name, content in before.items():
                assert (aside / name).read_bytes() == content
            assert _main_db_names(data_dir) == [live.name]
        finally:
            zip_path.unlink(missing_ok=True)

    def test_a_mid_swap_failure_rolls_every_moved_file_back_byte_identical(
        self, data_dir: Path
    ) -> None:
        """A failure partway through the swap — here, the staged documents
        tree cannot be written because a FILE now sits where a directory
        needs to be created — undoes every artifact already moved, including
        ones swapped in before the failing one (the main database and a
        sibling release with its WAL sidecar), and quarantines the staged set
        instead of leaving it for the next boot to re-attempt.

        Would catch: a rollback that only reverts the artifact that failed
        and leaves an earlier, already-applied move (the main database swap)
        in a half-restored state, or a failed staged set left at
        .pending-restore so the next boot re-attempts the same broken
        restore forever.
        """
        _set_version(_RUNNING_VERSION)
        VersionedDatabaseService().provision()
        live = FileMapperService.get_db_path()
        _write_probe(live, "in-the-snapshot")
        docs_dir = FileMapperService.get_documents_path()
        reports_dir = docs_dir / "reports"
        reports_dir.mkdir(parents=True)
        (reports_dir / "quarterly.txt").write_text("nested document\n", encoding="utf-8")

        svc = SnapshotService()
        zip_path = svc.export(password=None)
        try:
            _tiny_sqlite(data_dir / "chalie-1.1.0.sqlite")
            (data_dir / "chalie-1.1.0.sqlite-wal").write_bytes(b"stale frames")
            _write_probe(live, "written-after-the-snapshot")
            before = {p.name: p.read_bytes() for p in data_dir.glob("chalie*")}

            # A genuine filesystem failure: dest.parent.mkdir() must find a
            # directory (or nothing) at documents/reports/ — put a FILE there
            # instead, so it raises FileExistsError mid-swap.
            shutil.rmtree(reports_dir)
            reports_dir.write_text("not a directory\n", encoding="utf-8")

            svc.stage_import(zip_path, password=None)
            SnapshotService.apply_pending()

            after = {p.name: p.read_bytes() for p in data_dir.glob("chalie*")}
            assert after == before
            assert not list(data_dir.glob(".pre-restore-*"))
            assert list(data_dir.glob(".restore-failed-*"))
            assert not FileMapperService.get_pending_restore_path().exists()
        finally:
            zip_path.unlink(missing_ok=True)


class TestRestoreGuard:
    """SnapshotService._guard_restore refuses a snapshot this build cannot
    safely restore, before anything is staged."""

    def test_unparsable_version_is_refused_with_nothing_staged_and_live_file_untouched(
        self, data_dir: Path
    ) -> None:
        """A snapshot whose VERSION cannot be parsed is refused loudly at
        stage_import — before anything is staged and before the live
        instance is touched.

        Would catch: the ValueError from version_sort_key leaking out
        unconverted, the guard running after staging is committed (leaving a
        .pending-restore directory for the next boot to trip over on a
        refusal), or the guard mutating the live database before it finishes
        validating the snapshot.
        """
        _set_version("not-a-release")
        _tiny_sqlite(FileMapperService.get_db_path())
        svc = SnapshotService()
        zip_path = svc.export(password=None)
        try:
            _set_version(_RUNNING_VERSION)
            VersionedDatabaseService().provision()
            live = FileMapperService.get_db_path()
            _write_probe(live, "must-survive-the-refused-restore")
            before = live.read_bytes()

            with pytest.raises(SnapshotError, match="not-a-release"):
                svc.stage_import(zip_path, password=None)

            assert live.read_bytes() == before
            assert not FileMapperService.get_pending_restore_path().exists()
            assert not list(data_dir.glob(".pre-restore-*"))
        finally:
            zip_path.unlink(missing_ok=True)
