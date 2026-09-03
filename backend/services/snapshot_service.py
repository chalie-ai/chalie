"""Snapshot Service — whole-instance Time-Machine export / restore.

Produces a single standard ``.zip`` that is a complete clone of the running
instance (the WAL-folded SQLite databases, the vault key-material backups, the
document store, the user skills, and the VERSION marker), and restores such a
clone with a true wipe-and-replace at the next boot.

Restore is two-phase so a half-finished swap can never corrupt a live instance:
  * Phase A — ``stage_import``: extract + verify checksums + run the restore
    guard into a private temp dir, then *atomically* rename it
    to ``data/.pending-restore``. Any failure leaves nothing staged.
  * Phase B — ``apply_pending`` (called from ``run.py`` before the DB is opened):
    re-verify each staged artifact, move the live artifacts aside into a single
    ``data/.pre-restore-<ts>`` directory, move the staged artifacts into place,
    then clear the marker. On any failure it rolls the live artifacts back, logs
    loudly, and quarantines the staged set so boot does not loop.

The main database is versioned per release, so a restore never writes the
running build's own file. The snapshot's database lands under the name of the
release that TOOK it — ``data/chalie-<snapshot VERSION>.sqlite``, or the
pre-versioning ``data/chalie.db`` when the snapshot carries no VERSION — and
every other main database in ``data/`` moves aside with it, so the restored
file is the newest one left. The normal boot does the rest:
``VersionedDatabaseService.provision`` opens that file directly when it is this
release's, and copies it forward when it is older. The VERSION file itself is
never restored — it is the running build's identity; the staged copy is read
only for the downgrade guard and for that landing name.

Consumed by ``api/actions/snapshot/`` (HTTP surface) and ``run.py`` (boot apply).
Depends on ``FileMapperService`` (all paths), ``services.app_version`` (the
version comparison behind the restore guard) and ``services.time_utils``
(UTC stamps).
"""

import hashlib
import json
import logging
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import cast

import pyzipper

from exceptions import SnapshotError
from services.app_version import get_version, version_sort_key
from services.file_mapper_service import FileMapperService
from services.time_utils import utc_now

logger = logging.getLogger(__name__)

# Manifest + zip layout constants (no magic literals at call sites).
_MANIFEST_NAME = "manifest.json"
_MANIFEST_VERSION = 1
_HASH_CHUNK_BYTES = 65536
_AES_BITS = 256
_TS_FORMAT = "%Y%m%dT%H%M%S%fZ"

# Filesystem perms re-enforced after a restore (zip extraction loses them).
# Mirrors VaultService: secure DIR owner-rwx, each backup file owner-read-only.
_SECURE_DIR_MODE = 0o700
_SECURE_FILE_MODE = 0o400
_RESTORE_FAILED_PREFIX = ".restore-failed-"
_PRE_RESTORE_PREFIX = ".pre-restore-"

# Logical artifact kinds. Single-file DB/file kinds map to one destination
# path; tree kinds carry a relative sub-path under a destination directory.
_KIND_CHALIE_DB = "chalie_db"
_KIND_MCP_TOOLS = "mcp_tools"
_KIND_SKILLS = "skills"
_KIND_SECURE = "secure"
_KIND_DOCUMENTS = "documents"
_KIND_SKILLS_USER = "skills_user"
_KIND_VERSION = "version"

# The three WAL-folded SQLite databases, in (kind, destination-helper) form.
_SINGLE_FILE_DB_KINDS = (_KIND_CHALIE_DB, _KIND_MCP_TOOLS, _KIND_SKILLS)

# SQLite WAL-mode sidecars. The staged DBs are WAL-folded (no sidecars), so on
# swap the live sidecars must be cleared away from the destination or a fresh
# connection would replay stale frames onto the restored DB.
_SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm")

# KINDs whose members are tree-relative (an arcname under a destination dir),
# as opposed to single-file kinds that map straight to one destination path.
_TREE_KINDS = (_KIND_SECURE, _KIND_DOCUMENTS, _KIND_SKILLS_USER)

# Every artifact KIND this build understands. VERSION is one of them — it is
# exported, checksum-verified and read — but it is the only known KIND that is
# never written back into the live instance; see _is_restorable, which owns
# both that rule and the skip for a KIND this build does not know at all.
_KNOWN_KINDS = frozenset((*_SINGLE_FILE_DB_KINDS, _KIND_VERSION, *_TREE_KINDS))


class SnapshotService:
    """Engine for the whole-instance snapshot Time-Machine.

    No-arg constructor — every path resolves through ``FileMapperService`` at
    call time, so a redirected layout (tests, alternate data roots) is honoured
    without injection here.
    """

    def __init__(self) -> None:
        self._fm = FileMapperService

    # ── KIND → live destination ──────────────────────────────────────────────

    def _single_file_destination(self, kind: str, snapshot_version: str | None) -> Path:
        """Map a single-file artifact KIND to its live filesystem destination.

        ``_KIND_VERSION`` has no destination on purpose — it is never restored
        (see :meth:`_is_restorable`), so routing one here is a bug and raises.
        """
        routes = {
            _KIND_CHALIE_DB: self._restored_database_path(snapshot_version),
            _KIND_MCP_TOOLS: self._fm.get_mcp_tools_db_path(),
            _KIND_SKILLS: self._fm.get_skills_db_path(),
        }
        return routes[kind]

    def _restored_database_path(self, snapshot_version: str | None) -> Path:
        """Where the snapshot's main database lands: the file name of the
        release that took it, or the pre-versioning ``chalie.db`` when the
        snapshot carries no VERSION marker.

        Deliberately not ``get_db_path()``. ``VersionedDatabaseService`` opens
        the running release's own file and treats one standing there without a
        lineage row for that release as an unfinished build: it moves it aside
        and rebuilds from whatever older database the data dir still holds — so
        an older snapshot landed at the running name is silently discarded.
        Landed under its own name it is just the newest older database, and the
        ordinary boot copies it forward. A snapshot from this same release
        lands at the running name anyway, carrying its own lineage row, and is
        opened as-is.
        """
        if snapshot_version is None:
            return self._fm.get_legacy_db_path()
        return self._fm.get_versioned_db_path(snapshot_version)

    def _live_main_databases(self) -> list[Path]:
        """Every main-database file in the data dir: the pre-versioning
        ``chalie.db`` and each ``data/chalie-<version>.sqlite``.

        Both names are ``FileMapperService``'s to own —
        :meth:`~services.file_mapper_service.FileMapperService.get_legacy_db_path`
        and
        :meth:`~services.file_mapper_service.FileMapperService.version_from_db_path`
        (the inverse of the versioned name) — so the layout is declared in one
        place and this module never re-spells the pattern.
        """
        legacy = self._fm.get_legacy_db_path()
        return ([legacy] if legacy.exists() else []) + sorted(
            path
            for path in self._fm.get_data_path().glob("*")
            if self._fm.version_from_db_path(path) is not None
        )

    def _tree_root(self, kind: str) -> Path:
        """Map a tree artifact KIND to its live destination directory root."""
        roots = {
            _KIND_SECURE: self._fm.get_secure_dir(),
            _KIND_DOCUMENTS: self._fm.get_documents_path(),
            _KIND_SKILLS_USER: self._fm.get_user_skills_path(),
        }
        return roots[kind]

    def _destination_for(self, entry: dict[str, object], snapshot_version: str | None) -> Path:
        """Resolve the live destination path for a manifest entry (any KIND)."""
        kind = cast(str, entry["kind"])
        if kind in _TREE_KINDS:
            return self._tree_root(kind) / cast(str, entry["rel"])
        return self._single_file_destination(kind, snapshot_version)

    # ── Hashing (shared by export verify-write and import verify-read) ─────────

    @staticmethod
    def _sha256(path: Path) -> str:
        """Return the hex SHA-256 of a file, streamed in fixed chunks."""
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(_HASH_CHUNK_BYTES), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _wal_fold(src: Path, dst: Path) -> None:
        """Fold a live SQLite DB (incl. any -wal/-shm) into a single consistent
        file via the backup API — yields a clone with no sidecar files."""
        src_conn = sqlite3.connect(str(src))
        dst_conn = sqlite3.connect(str(dst))
        try:
            src_conn.backup(dst_conn)
        finally:
            dst_conn.close()
            src_conn.close()

    # ── Export ─────────────────────────────────────────────────────────────────

    def export(self, password: str | None) -> Path:
        """Produce a snapshot zip of the whole instance and return its path.

        A password yields a real AES-256 zip; otherwise a plain deflate zip.
        """
        workspace = Path(tempfile.mkdtemp(prefix="chalie-snapshot-"))
        staged = workspace / "snapshot"
        staged.mkdir(parents=True, exist_ok=True)

        manifest = self._assemble_export(staged)
        (staged / _MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        stamp = utc_now().strftime(_TS_FORMAT)
        zip_path = workspace / f"chalie-snapshot-{stamp}.zip"
        self._write_zip(staged, zip_path, manifest, password)
        return zip_path

    def _assemble_export(self, staged: Path) -> dict[str, object]:
        """Lay every artifact into *staged* and return the manifest dict.

        WAL-folds the three DBs, copies the secure/documents/skills_user trees,
        and copies the VERSION marker, recording KIND + arcname + sha256 for
        each member.
        """
        entries: list[dict[str, object]] = []
        self._stage_single_file_dbs(staged, entries)
        self._stage_trees(staged, entries)
        self._stage_version(staged, entries)
        return {"version": _MANIFEST_VERSION, "artifacts": entries}

    def _stage_single_file_dbs(self, staged: Path, entries: list[dict[str, object]]) -> None:
        """WAL-fold each of the three databases into the staging dir."""
        sources = {
            _KIND_CHALIE_DB: self._fm.get_db_path(),
            _KIND_MCP_TOOLS: self._fm.get_mcp_tools_db_path(),
            _KIND_SKILLS: self._fm.get_skills_db_path(),
        }
        for kind in _SINGLE_FILE_DB_KINDS:
            src = sources[kind]
            if not src.exists():
                continue
            arcname = f"{kind}/{src.name}"
            target = staged / arcname
            target.parent.mkdir(parents=True, exist_ok=True)
            self._wal_fold(src, target)
            entries.append(self._entry(kind, arcname, target))

    def _stage_trees(self, staged: Path, entries: list[dict[str, object]]) -> None:
        """Copy the secure / documents / skills_user trees into staging."""
        roots = {
            _KIND_SECURE: self._fm.get_secure_dir(),
            _KIND_DOCUMENTS: self._fm.get_documents_path(),
            _KIND_SKILLS_USER: self._fm.get_user_skills_path(),
        }
        for kind, root in roots.items():
            if not root.is_dir():
                continue
            for item in root.rglob("*"):
                if not item.is_file():
                    continue
                rel = item.relative_to(root).as_posix()
                arcname = f"{kind}/{rel}"
                target = staged / arcname
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)
                entries.append(self._entry(kind, arcname, target, rel=rel))

    def _stage_version(self, staged: Path, entries: list[dict[str, object]]) -> None:
        """Copy the VERSION marker into staging."""
        src = self._fm.get_version_path()
        if not src.exists():
            return
        arcname = f"{_KIND_VERSION}/{src.name}"
        target = staged / arcname
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        entries.append(self._entry(_KIND_VERSION, arcname, target))

    def _entry(self, kind: str, arcname: str, staged_file: Path, rel: str | None = None) -> dict[str, object]:
        """Build one manifest entry recording KIND + arcname + sha256 (+ rel)."""
        entry: dict[str, object] = {"kind": kind, "arcname": arcname, "sha256": self._sha256(staged_file)}
        if rel is not None:
            entry["rel"] = rel
        return entry

    def _write_zip(self, staged: Path, zip_path: Path, manifest: dict[str, object], password: str | None) -> None:
        """Write every staged member (and the manifest) into the zip, AES-256 if
        a password is supplied, plain deflate otherwise."""
        with pyzipper.AESZipFile(
            str(zip_path), "w", compression=pyzipper.ZIP_DEFLATED
        ) as zf:
            if password:
                zf.setpassword(password.encode("utf-8"))
                zf.setencryption(pyzipper.WZ_AES, nbits=_AES_BITS)
            zf.writestr(_MANIFEST_NAME, json.dumps(manifest, indent=2))
            for entry in cast(list[dict[str, object]], manifest["artifacts"]):
                zf.write(str(staged / cast(str, entry["arcname"])), cast(str, entry["arcname"]))

    # ── Import (Phase A) ─────────────────────────────────────────────────────

    def stage_import(self, zip_path: Path, password: str | None) -> None:
        """Extract + verify a snapshot into a temp dir, run the downgrade guard,
        then atomically rename it to ``data/.pending-restore``.

        Raises ``SnapshotError`` (or the underlying zip error) on any failure,
        leaving NO ``.pending-restore`` and NO ``.pre-restore-*`` behind.
        """
        scratch = Path(tempfile.mkdtemp(
            prefix="stage-", dir=str(self._staging_parent())
        ))
        try:
            self._extract_all(zip_path, scratch, password)
            manifest = self._read_manifest(scratch)
            self._verify_members(scratch, manifest)
            self._guard_restore(scratch, manifest)
            self._commit_staging(scratch)
        except Exception:
            shutil.rmtree(scratch, ignore_errors=True)
            raise

    def _staging_parent(self) -> Path:
        """Return (creating if needed) the snapshot-staging dir that hosts the
        scratch extract. It lives under data/ so it shares a filesystem with the
        final ``.pending-restore`` and the commit rename is atomic."""
        parent = self._fm.get_snapshot_staging_path()
        parent.mkdir(parents=True, exist_ok=True)
        return parent

    def _extract_all(self, zip_path: Path, dest: Path, password: str | None) -> None:
        """Extract every member through the single AES-capable read path. A
        wrong password raises ``RuntimeError`` from pyzipper — surfaced loudly."""
        with pyzipper.AESZipFile(str(zip_path), "r") as zf:
            if password:
                zf.setpassword(password.encode("utf-8"))
            zf.extractall(str(dest))

    def _read_manifest(self, root: Path) -> dict[str, object]:
        """Load and shallow-validate the manifest from an extracted snapshot."""
        manifest_path = root / _MANIFEST_NAME
        if not manifest_path.exists():
            raise SnapshotError("Snapshot is missing manifest.json")
        manifest: dict[str, object] = cast(dict[str, object], json.loads(manifest_path.read_text(encoding="utf-8")))
        if not isinstance(manifest.get("artifacts"), list) or not cast(list[object], manifest["artifacts"]):
            raise SnapshotError("Snapshot manifest declares no artifacts")
        return manifest

    def _verify_members(self, root: Path, manifest: dict[str, object]) -> None:
        """Re-hash every extracted member and compare to the manifest sha256."""
        for entry in cast(list[dict[str, object]], manifest["artifacts"]):
            member = root / cast(str, entry["arcname"])
            if not member.exists():
                raise SnapshotError(f"Snapshot is missing artifact {cast(str, entry['arcname'])}")
            actual = self._sha256(member)
            if actual != cast(str, entry["sha256"]):
                raise SnapshotError(
                    f"Checksum mismatch for {cast(str, entry['arcname'])}"
                )

    def _guard_restore(self, root: Path, manifest: dict[str, object]) -> None:
        """Block a snapshot this build cannot restore.

        Two refusals: a snapshot with no database is nothing to restore, and a
        snapshot taken by a NEWER release cannot be carried backwards — the
        older build would provision its own database from an older schema.sql
        and copy forward only the columns it still declares, silently dropping
        whatever the newer release added. A snapshot carrying no VERSION marker
        predates versioning and is by definition older, so it is allowed.

        A VERSION this build cannot read is a third refusal, and it belongs
        here rather than at apply time: the same string names the file the
        restored database lands under, ``apply_pending`` swallows its own
        errors so boot survives them, and a database landed under an
        unreadable version is one ``VersionedDatabaseService`` skips — the
        restore would evaporate with nothing but a warning. Refused at stage
        time it is an error the operator sees, with nothing staged.
        """
        artifacts = cast(list[dict[str, object]], manifest["artifacts"])
        if not any(e["kind"] == _KIND_CHALIE_DB for e in artifacts):
            raise SnapshotError("Snapshot has no main database to restore")

        snapshot_version = self._snapshot_version(root, manifest)
        if snapshot_version is None:
            return

        try:
            snapshot_key = version_sort_key(snapshot_version)
        except ValueError as exc:
            raise SnapshotError(
                f"Snapshot carries an unreadable VERSION ({snapshot_version!r}) "
                "and cannot be restored"
            ) from exc

        running_version = get_version()
        if snapshot_key > version_sort_key(running_version):
            raise SnapshotError(
                f"Snapshot was taken by a newer build ({snapshot_version} > "
                f"{running_version}) and cannot be restored into this one"
            )

    def _snapshot_version(self, root: Path, manifest: dict[str, object]) -> str | None:
        """Return the version of the release that took the snapshot, or None
        when it carries no VERSION marker (a snapshot from before versioning).

        The staged VERSION is read for exactly two decisions — the downgrade
        guard above and the name the restored database lands under. It is never
        written over the running build's own VERSION file.
        """
        entry = next(
            (
                e
                for e in cast(list[dict[str, object]], manifest["artifacts"])
                if e["kind"] == _KIND_VERSION
            ),
            None,
        )
        if entry is None:
            return None
        return (root / cast(str, entry["arcname"])).read_text(encoding="utf-8").strip()

    def _commit_staging(self, scratch: Path) -> None:
        """Atomically promote the verified scratch dir to ``.pending-restore``
        as the final step — nothing partial survives an earlier failure."""
        pending = self._fm.get_pending_restore_path()
        if pending.exists():
            shutil.rmtree(pending)
        os.replace(str(scratch), str(pending))

    # ── Boot apply (Phase B) ───────────────────────────────────────────────────

    @staticmethod
    def apply_pending() -> None:
        """Apply a staged restore at boot, before the DB is opened (run.py:402).

        No-op when nothing is staged. Otherwise: re-verify each staged artifact
        against the staged manifest, move the live artifacts into one
        ``.pre-restore-<ts>`` aside, move the staged artifacts into place, and
        clear the marker. Owns its own try/except — on failure it rolls the live
        artifacts back and logs loudly; it never propagates (boot safety).

        A failure DURING the swap quarantines the staged set so the next boot
        does not re-attempt a half-applied restore. A failure during the
        pre-swap re-verification (e.g. an unreadable staged artifact) leaves the
        live instance wholly untouched — there is nothing to roll back and the
        staged set is left in place to be inspected."""
        fm = FileMapperService
        pending = fm.get_pending_restore_path()
        if not pending.exists():
            return

        svc = SnapshotService()
        try:
            svc._apply_staged(pending)
        except Exception:
            logger.exception("[snapshot] apply_pending failed — live instance left intact")

    def _apply_staged(self, pending: Path) -> None:
        """Re-verify the staged set, then swap it into place with a rollback
        aside. Raises on any failure (the caller logs; boot continues)."""
        manifest = self._read_manifest(pending)
        # Re-verify BEFORE touching any live artifact: a failure here (e.g. an
        # unreadable staged file) leaves the live instance entirely intact and
        # the staged set in place — no aside, no rollback needed.
        self._verify_members(pending, manifest)
        self._swap_in(pending, manifest)

    def _swap_in(self, pending: Path, manifest: dict[str, object]) -> None:
        """Move live artifacts into one aside, staged artifacts into place, then
        clear the marker. On a mid-swap failure, roll the live artifacts back and
        quarantine the staged set so the next boot does not re-apply it."""
        aside = self._fm.get_data_path(f"{_PRE_RESTORE_PREFIX}{utc_now().strftime(_TS_FORMAT)}")
        aside.mkdir(parents=True, exist_ok=True)
        snapshot_version = self._snapshot_version(pending, manifest)

        moved: list[tuple[Path, Path]] = []  # (live_path, aside_copy)
        try:
            for entry in cast(list[dict[str, object]], manifest["artifacts"]):
                if not self._is_restorable(entry):
                    continue
                self._swap_artifact(pending, aside, entry, moved, snapshot_version)
        except Exception:
            self._rollback(moved)
            shutil.rmtree(aside, ignore_errors=True)
            self._quarantine_pending(pending)
            raise

        shutil.rmtree(pending, ignore_errors=True)
        self._reenforce_secure_perms()

    @staticmethod
    def _is_restorable(entry: dict[str, object]) -> bool:
        """Return True when this artifact is swapped into the live instance.

        Two are read but never written back. The VERSION marker is the running
        build's identity: restoring an older release's copy would leave the
        process reporting a version it is not and opening a database named
        after it, so it is skipped here and used only for the guard and the
        restored database's name. An artifact whose KIND this build does not
        know is skipped too — a snapshot is a portable, cross-version backup,
        so it may legitimately carry an artifact a newer build has since
        dropped (e.g. the removed session_secret), and that is no reason to
        abort the whole restore.
        """
        kind = entry["kind"]
        if kind == _KIND_VERSION:
            logger.info(
                "[snapshot] not restoring the VERSION marker — this build keeps its own "
                "version; the snapshot's names the file its database is restored into"
            )
            return False
        if kind not in _KNOWN_KINDS:
            logger.warning(
                "[snapshot] skipping unknown artifact kind %r (arcname=%s) — "
                "snapshot was exported by a build that declared an artifact "
                "this build no longer restores",
                kind, entry.get("arcname"),
            )
            return False
        return True

    def _reenforce_secure_perms(self) -> None:
        """Re-lock the restored key material — zip extraction does not preserve
        the restrictive perms VaultService writes (secure dir 0o700, each backup
        0o400). Best-effort: a restore that already succeeded must not be undone
        by a chmod failure, so log loudly instead."""
        secure_dir = self._fm.get_secure_dir()
        try:
            if secure_dir.is_dir():
                os.chmod(secure_dir, _SECURE_DIR_MODE)
                for backup in self._fm.list_vault_backups():
                    os.chmod(backup, _SECURE_FILE_MODE)
        except OSError:
            logger.exception("[snapshot] could not re-enforce secure-dir perms after restore")

    def _swap_artifact(
        self,
        pending: Path,
        aside: Path,
        entry: dict[str, object],
        moved: list[tuple[Path, Path]],
        snapshot_version: str | None,
    ) -> None:
        """Clear whatever this artifact would overwrite into the aside, then
        move the staged copy into its live destination.

        The main database clears every main-database file in the data dir, not
        just the one it lands on; every other KIND clears its single
        destination.
        """
        staged_file = pending / cast(str, entry["arcname"])
        dest = self._destination_for(entry, snapshot_version)

        if entry["kind"] == _KIND_CHALIE_DB:
            self._clear_main_databases(dest, aside, moved)
        else:
            self._clear_destination(entry, dest, aside, moved)

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staged_file), str(dest))

    def _clear_main_databases(self, dest: Path, aside: Path, moved: list[tuple[Path, Path]]) -> None:
        """Move every live main-database file — the legacy name, every versioned
        release file, and each one's WAL sidecars — into the aside.

        Not only the file the restore lands on: ``VersionedDatabaseService``
        builds the running release's database from the NEWEST database older
        than it, so any main database left behind that is newer than the
        restored one would be copied forward in its place and the restore would
        be invisible. The snapshot replaces the whole main database, whichever
        release's name it wears.

        *dest* is recorded even when no file stands there, so a failure while
        the staged database is being moved in still rolls that file away.
        """
        live_files = self._live_main_databases()
        if dest not in live_files:
            live_files.append(dest)
        for live in live_files:
            self._move_aside(live, aside / _KIND_CHALIE_DB / live.name, moved, sidecars=True)

    def _clear_destination(
        self, entry: dict[str, object], dest: Path, aside: Path, moved: list[tuple[Path, Path]]
    ) -> None:
        """Move the one live file this artifact overwrites into the aside.

        For a database the live WAL-mode sidecars go with it, so the restored
        (WAL-folded) file starts clean instead of replaying stale frames.
        """
        self._move_aside(
            dest,
            aside / cast(str, entry["arcname"]),
            moved,
            sidecars=entry["kind"] in _SINGLE_FILE_DB_KINDS,
        )

    def _move_aside(self, live: Path, aside_copy: Path, moved: list[tuple[Path, Path]], sidecars: bool) -> None:
        """Move one live file (optionally with its WAL sidecars) into the aside
        and record its return ticket.

        The pair is recorded whether or not anything was there to move, and
        before the staged file lands: the record is what ``_rollback`` needs to
        both put the live file back AND remove whatever the swap wrote at that
        path, and the aside directory is deleted on failure — a file moved
        aside without its ticket recorded would go with it.
        """
        aside_copy.parent.mkdir(parents=True, exist_ok=True)
        if live.exists():
            shutil.move(str(live), str(aside_copy))
        if sidecars:
            self._move_sidecars(live, aside_copy)
        moved.append((live, aside_copy))

    @staticmethod
    def _move_sidecars(live: Path, aside_copy: Path) -> None:
        """Move any ``-wal``/``-shm`` sidecars of *live* next to its aside copy
        (so rollback can restore them and the new DB starts clean)."""
        for suffix in _SQLITE_SIDECAR_SUFFIXES:
            sidecar = live.parent / f"{live.name}{suffix}"
            if sidecar.exists():
                shutil.move(str(sidecar), str(aside_copy.parent / f"{aside_copy.name}{suffix}"))

    def _rollback(self, moved: list[tuple[Path, Path]]) -> None:
        """Put every file the swap moved aside back where it was, newest first,
        removing whatever the swap wrote at that path — so the live instance is
        byte-identical to before the failed apply, sidecars included."""
        for live, aside_copy in reversed(moved):
            if live.exists():
                live.unlink()
            if aside_copy.exists():
                live.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(aside_copy), str(live))
            for suffix in _SQLITE_SIDECAR_SUFFIXES:
                aside_sidecar = aside_copy.parent / f"{aside_copy.name}{suffix}"
                if aside_sidecar.exists():
                    shutil.move(str(aside_sidecar), str(live.parent / f"{live.name}{suffix}"))

    def _quarantine_pending(self, pending: Path) -> None:
        """Rename a failed staged set out of the way so the next boot does not
        re-attempt the same broken restore. Never raises."""
        try:
            failed = self._fm.get_data_path(
                f"{_RESTORE_FAILED_PREFIX}{utc_now().strftime(_TS_FORMAT)}"
            )
            os.replace(str(pending), str(failed))
        except Exception:
            logger.exception("[snapshot] could not quarantine the failed staged restore")
