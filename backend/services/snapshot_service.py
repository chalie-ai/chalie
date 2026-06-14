"""Snapshot Service — whole-instance Time-Machine export / restore (TKT-949).

Produces a single standard ``.zip`` that is a complete clone of the running
instance (the WAL-folded SQLite databases, the vault key-material backups, the
document store, the user skills, and the VERSION marker), and restores such a
clone with a true wipe-and-replace at the next boot.

Crypto: a password yields a real WZ-AES-256 zip via ``pyzipper``; no password
yields a plain ``ZIP_DEFLATED`` zip. One read path (``pyzipper.AESZipFile``)
handles both on import.

Restore is two-phase so a half-finished swap can never corrupt a live instance:
  * Phase A — ``stage_import``: extract + verify checksums + run the
    schema-downgrade guard into a private temp dir, then *atomically* rename it
    to ``data/.pending-restore``. Any failure leaves nothing staged.
  * Phase B — ``apply_pending`` (called from ``run.py`` before the DB is opened):
    re-verify each staged artifact, move the live artifacts aside into a single
    ``data/.pre-restore-<ts>`` directory, move the staged artifacts into place,
    then clear the marker. On any failure it rolls the live artifacts back, logs
    loudly, and quarantines the staged set so boot does not loop.

Consumed by ``api/snapshot.py`` (HTTP surface) and ``run.py`` (boot apply).
Depends on ``FileMapperService`` (all paths), ``SchemaConvergenceService``
(shared ``column_set`` for the downgrade guard) and ``services.time_utils``
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

import pyzipper

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


class SnapshotError(Exception):
    """Raised when an import is rejected loudly (bad password handled by the
    zip layer, corrupt zip, checksum mismatch, or a schema-downgrade block)."""


class SnapshotService:
    """Engine for the whole-instance snapshot Time-Machine.

    No-arg constructor — every path resolves through ``FileMapperService`` at
    call time, so a redirected layout (tests, alternate data roots) is honoured
    without injection here.
    """

    def __init__(self) -> None:
        self._fm = FileMapperService

    # ── KIND → live destination ──────────────────────────────────────────────

    def _single_file_destination(self, kind: str) -> Path:
        """Map a single-file artifact KIND to its live filesystem destination."""
        routes = {
            _KIND_CHALIE_DB: self._fm.get_db_path(),
            _KIND_MCP_TOOLS: self._fm.get_mcp_tools_db_path(),
            _KIND_SKILLS: self._fm.get_skills_db_path(),
            _KIND_VERSION: self._fm.get_version_path(),
        }
        return routes[kind]

    def _tree_root(self, kind: str) -> Path:
        """Map a tree artifact KIND to its live destination directory root."""
        roots = {
            _KIND_SECURE: self._fm.get_secure_dir(),
            _KIND_DOCUMENTS: self._fm.get_documents_path(),
            _KIND_SKILLS_USER: self._fm.get_user_skills_path(),
        }
        return roots[kind]

    def _destination_for(self, entry: dict) -> Path:
        """Resolve the live destination path for a manifest entry (any KIND)."""
        kind = entry["kind"]
        if kind in _TREE_KINDS:
            return self._tree_root(kind) / entry["rel"]
        return self._single_file_destination(kind)

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

    def _assemble_export(self, staged: Path) -> dict:
        """Lay every artifact into *staged* and return the manifest dict.

        WAL-folds the three DBs, copies the secure/documents/skills_user trees,
        and copies the VERSION marker, recording KIND + arcname + sha256 for
        each member.
        """
        entries: list[dict] = []
        self._stage_single_file_dbs(staged, entries)
        self._stage_trees(staged, entries)
        self._stage_version(staged, entries)
        return {"version": _MANIFEST_VERSION, "artifacts": entries}

    def _stage_single_file_dbs(self, staged: Path, entries: list[dict]) -> None:
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

    def _stage_trees(self, staged: Path, entries: list[dict]) -> None:
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

    def _stage_version(self, staged: Path, entries: list[dict]) -> None:
        """Copy the VERSION marker into staging."""
        src = self._fm.get_version_path()
        if not src.exists():
            return
        arcname = f"{_KIND_VERSION}/{src.name}"
        target = staged / arcname
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        entries.append(self._entry(_KIND_VERSION, arcname, target))

    def _entry(self, kind: str, arcname: str, staged_file: Path, rel: str | None = None) -> dict:
        """Build one manifest entry recording KIND + arcname + sha256 (+ rel)."""
        entry = {"kind": kind, "arcname": arcname, "sha256": self._sha256(staged_file)}
        if rel is not None:
            entry["rel"] = rel
        return entry

    def _write_zip(self, staged: Path, zip_path: Path, manifest: dict, password: str | None) -> None:
        """Write every staged member (and the manifest) into the zip, AES-256 if
        a password is supplied, plain deflate otherwise."""
        with pyzipper.AESZipFile(
            str(zip_path), "w", compression=pyzipper.ZIP_DEFLATED
        ) as zf:
            if password:
                zf.setpassword(password.encode("utf-8"))
                zf.setencryption(pyzipper.WZ_AES, nbits=_AES_BITS)
            zf.writestr(_MANIFEST_NAME, json.dumps(manifest, indent=2))
            for entry in manifest["artifacts"]:
                zf.write(str(staged / entry["arcname"]), entry["arcname"])

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
            self._guard_schema_downgrade(scratch, manifest)
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

    def _read_manifest(self, root: Path) -> dict:
        """Load and shallow-validate the manifest from an extracted snapshot."""
        manifest_path = root / _MANIFEST_NAME
        if not manifest_path.exists():
            raise SnapshotError("Snapshot is missing manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest.get("artifacts"), list) or not manifest["artifacts"]:
            raise SnapshotError("Snapshot manifest declares no artifacts")
        return manifest

    def _verify_members(self, root: Path, manifest: dict) -> None:
        """Re-hash every extracted member and compare to the manifest sha256."""
        for entry in manifest["artifacts"]:
            member = root / entry["arcname"]
            if not member.exists():
                raise SnapshotError(f"Snapshot is missing artifact {entry['arcname']}")
            actual = self._sha256(member)
            if actual != entry["sha256"]:
                raise SnapshotError(
                    f"Checksum mismatch for {entry['arcname']}"
                )

    def _guard_schema_downgrade(self, root: Path, manifest: dict) -> None:
        """Block a snapshot whose chalie.db carries any column the running
        build's schema.sql does not declare (snapshot ⊋ build = a newer DB whose
        restore would make convergence DROP user data)."""
        chalie_entry = next(
            (e for e in manifest["artifacts"] if e["kind"] == _KIND_CHALIE_DB), None
        )
        if chalie_entry is None:
            raise SnapshotError("Snapshot has no chalie.db to restore")

        build_cols = self._build_column_set()
        snapshot_cols = self._snapshot_column_set(root / chalie_entry["arcname"])

        extra = {
            f"{table}.{col}"
            for table, cols in snapshot_cols.items()
            for col in cols - build_cols.get(table, set())
        }
        if extra:
            raise SnapshotError(
                "Snapshot schema is newer than this build (would drop data): "
                + ", ".join(sorted(extra))
            )

    def _build_column_set(self) -> dict:
        """Column set of the running build, derived from schema.sql via the
        shared convergence introspection (so it matches convergence exactly)."""
        from services.database_service import get_shared_db_service
        from services.schema_convergence_service import SchemaConvergenceService

        convergence = SchemaConvergenceService(get_shared_db_service())
        schema_sql = self._fm.get_schema_path().read_text(encoding="utf-8")
        desired_conn = convergence._load_desired_state(schema_sql)
        try:
            return convergence.column_set(desired_conn)
        finally:
            desired_conn.close()

    def _snapshot_column_set(self, chalie_db: Path) -> dict:
        """Column set of the snapshot's chalie.db, read-only, via the shared
        convergence introspection."""
        from services.schema_convergence_service import SchemaConvergenceService

        convergence = SchemaConvergenceService(None)
        conn = sqlite3.connect(f"file:{chalie_db}?mode=ro", uri=True)
        try:
            return convergence.column_set(conn)
        finally:
            conn.close()

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

    def _swap_in(self, pending: Path, manifest: dict) -> None:
        """Move live artifacts into one aside, staged artifacts into place, then
        clear the marker. On a mid-swap failure, roll the live artifacts back and
        quarantine the staged set so the next boot does not re-apply it."""
        aside = self._fm.get_data_path(f"{_PRE_RESTORE_PREFIX}{utc_now().strftime(_TS_FORMAT)}")
        aside.mkdir(parents=True, exist_ok=True)

        moved: list[tuple[Path, Path]] = []  # (live_destination, aside_copy)
        try:
            for entry in manifest["artifacts"]:
                self._swap_artifact(pending, aside, entry, moved)
        except Exception:
            self._rollback(moved)
            shutil.rmtree(aside, ignore_errors=True)
            self._quarantine_pending(pending)
            raise

        shutil.rmtree(pending, ignore_errors=True)
        self._reenforce_secure_perms()

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

    def _swap_artifact(self, pending: Path, aside: Path, entry: dict, moved: list) -> None:
        """Move one live artifact into the aside, then move the staged copy into
        the live destination. Records the (dest, aside_copy) pair for rollback.

        For single-file DBs the live WAL-mode sidecars are moved aside alongside
        the main file so the restored (WAL-folded) DB starts clean."""
        staged_file = pending / entry["arcname"]
        dest = self._destination_for(entry)
        aside_copy = aside / entry["arcname"]
        aside_copy.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists():
            shutil.move(str(dest), str(aside_copy))
        if entry["kind"] in _SINGLE_FILE_DB_KINDS:
            self._move_sidecars(dest, aside_copy)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staged_file), str(dest))
        moved.append((dest, aside_copy))

    @staticmethod
    def _move_sidecars(dest: Path, aside_copy: Path) -> None:
        """Move any live ``-wal``/``-shm`` sidecars of *dest* next to its aside
        copy (so rollback can restore them and the new DB starts clean)."""
        for suffix in _SQLITE_SIDECAR_SUFFIXES:
            sidecar = dest.parent / f"{dest.name}{suffix}"
            if sidecar.exists():
                shutil.move(str(sidecar), str(aside_copy.parent / f"{aside_copy.name}{suffix}"))

    def _rollback(self, moved: list) -> None:
        """Restore every artifact that was already swapped, newest first, so the
        live instance is byte-identical to before the failed apply (incl. any
        DB sidecars that were moved aside)."""
        for dest, aside_copy in reversed(moved):
            if dest.exists():
                dest.unlink()
            if aside_copy.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(aside_copy), str(dest))
            for suffix in _SQLITE_SIDECAR_SUFFIXES:
                aside_sidecar = aside_copy.parent / f"{aside_copy.name}{suffix}"
                if aside_sidecar.exists():
                    shutil.move(str(aside_sidecar), str(dest.parent / f"{dest.name}{suffix}"))

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
