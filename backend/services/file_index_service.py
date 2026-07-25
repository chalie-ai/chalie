"""FileIndexService — Spotlight-style lexical file index over the local filesystem.

Owns a single FTS5 table in ``data/file_index.sqlite``.  Provides full-text
search over file paths, filenames, and extracted text content.  The index is
kept in sync with the filesystem via :meth:`reconcile`, which walks the scan
root and updates/deletes rows as files appear, change, or vanish.

All indexing constants (scan root, skip lists, extension whitelist, size
limit) are internal — no config surface, no UI, no settings.  The DB path
and scan root are constructor-overridable for tests only.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from services.database import Database
from services.file_mapper_service import FileMapperService
from services.text_reader import TextReader

logger = logging.getLogger(__name__)

# ── Internal constants (no config surface) ────────────────────────────────────

_SCAN_ROOT: str = "/"
_MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50 MB

# Subtrees to skip entirely during the walk.
_SKIP_SUBTREES: tuple[str, ...] = (
    "/proc",
    "/sys",
    "/dev",
    "/run",
    "/tmp",
    "/var",
    "/private",
    "/System",
    "/Volumes",
)

# Directory names to prune (never descend into them).
_SKIP_DIR_NAMES: tuple[str, ...] = (
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".cache",
    ".Trash",
)

# Chalie's own databases and sensitive directories to skip.
_SKIP_CHALIE_PATHS: tuple[str, ...] = (
    str(FileMapperService.get_db_path()),
    str(FileMapperService.get_file_index_db_path()),
    str(FileMapperService.get_abilities_db_path()),
    str(FileMapperService.get_skills_db_path()),
    str(FileMapperService.get_mcp_tools_db_path()),
    str(FileMapperService.get_secure_dir()),
    str(FileMapperService.get_snapshot_staging_path()),
    str(FileMapperService.get_pending_restore_path()),
)
# File extensions the index will accept.
_INDEXABLE_EXTENSIONS: tuple[str, ...] = (
    ".diff",
    ".patch",
    ".txt",
    ".md",
    ".rst",
    ".csv",
    ".tsv",
    ".log",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".py",
    ".js",
    ".ts",
    ".sh",
    ".pdf",
    ".docx",
    ".pptx",
    ".html",
    ".htm",
)


def _is_supported_extension(path: str) -> bool:
    """Return True if *path* has an extension the index accepts."""
    ext = Path(path).suffix.lower()
    return ext in _INDEXABLE_EXTENSIONS

def _is_dot_directory(name: str) -> bool:
    """Return True if *name* is a dot-directory (e.g. '.git', '.venv')."""
    return name.startswith(".")


def _is_in_skip_subtree(path: str) -> bool:
    """Return True if *path* is under one of the skipped subtrees."""
    for subtree in _SKIP_SUBTREES:
        if path == subtree or path.startswith(subtree + "/"):
            return True
    return False


def _is_in_skip_dirs(path: str) -> bool:
    """Return True if any component of *path* matches a pruned directory name."""
    parts = Path(path).parts
    for part in parts:
        if part in _SKIP_DIR_NAMES:
            return True
    return False


def _is_chalie_path(path: str) -> bool:
    """Return True if *path* is one of Chalie's own databases or sensitive dirs."""
    for entry in _SKIP_CHALIE_PATHS:
        if path == entry or path.startswith(entry + "/"):
            return True
    return False


def _is_parser_owned(path: str) -> bool:
    """Return True if *path* is a row only FileParserService creates.

    A file under the documents root whose extension the walk itself would
    never index (e.g. an image): its content column holds the vision/OCR
    description upserted at ingest. Reconcile keeps the row while the file
    exists and never re-extracts it — only FileParserService creates them.
    """
    root = str(FileMapperService.get_documents_path())
    return path.startswith(root + os.sep) and not _is_supported_extension(path)


def _is_library_caches(path: str) -> bool:
    """Return True if *path* is a directory named 'Caches' whose parent is 'Library'.

    This is a parent/child rule — the literal string ``'Library/Caches'`` can
    never be a single path component, so it is handled here instead of in
    ``_SKIP_DIR_NAMES``.
    """
    parent = os.path.dirname(path.rstrip(os.sep))
    return os.path.basename(path) == "Caches" and os.path.basename(parent) == "Library"


def _should_index(path: str) -> bool:
    """Return True if *path* should be indexed.

    Applies all the should_index rules:
    - Not under a skipped subtree.
    - No component matches a pruned directory name.
    - No path component starts with a dot.
    - Not inside a Library/Caches tree.
    - Not a Chalie internal path.
    - Has a supported extension.
    - File size <= 50 MB.
    """
    if _is_in_skip_subtree(path):
        return False
    if _is_in_skip_dirs(path):
        return False
    parts = Path(path).parts
    if any(part.startswith(".") for part in parts):
        return False
    if any(parts[i] == "Caches" and parts[i - 1] == "Library" for i in range(1, len(parts))):
        return False
    if _is_chalie_path(path):
        return False
    if not _is_supported_extension(path):
        return False
    try:
        size = os.path.getsize(path)
        if size > _MAX_FILE_SIZE:
            return False
    except OSError:
        return False
    return True


# ── FTS5 schema ───────────────────────────────────────────────────────────────

_FILE_INDEX_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS file_index_fts USING fts5(
    path,
    filename,
    content,
    mtime UNINDEXED,
    size UNINDEXED,
    tokenize='porter unicode61'
);
"""


class FileIndexService:
    """Spotlight-style lexical file index over the local filesystem.

    Manages a single FTS5 table in ``data/file_index.sqlite``.  Public API:
    - :meth:`index_file`: delete-then-insert a row for *path*.
    - :meth:`upsert_content`: delete-then-insert a row for *path* with given content.
    - :meth:`remove_file`: delete the row for *path*.
    - :meth:`clear`: delete every row (privacy delete-all).
    - :meth:`move_file`: delete src row, insert dst row.
    - :meth:`search`: return matching file paths.
    - :meth:`should_index`: return True if *path* should be indexed.
    - :meth:`reconcile`: walk the scan root, update index to match filesystem.
    """

    def __init__(self, scan_root: str = _SCAN_ROOT, db_path: str | None = None):
        """Initialize the file index service.

        Args:
            scan_root: Root directory to walk during reconcile.  Defaults to
                ``'/'``.  Constructor-overridable for tests.
            db_path: Path to the SQLite database.  Defaults to
                ``data/file_index.sqlite``.  Constructor-overridable for tests.
        """
        self.scan_root = scan_root
        self.db_path = db_path or str(FileMapperService.get_file_index_db_path())
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create the FTS5 table if it doesn't exist."""
        conn = Database.conn(self.db_path)
        conn.executescript(_FILE_INDEX_FTS_SCHEMA)

    def index_file(self, path: str) -> None:
        """Index a single file.

        Deletes any existing row for *path*, then inserts a new row with the
        file's content, filename, mtime, and size.  Content is extracted via
        :class:`TextReader`.  Per-file failures are logged and skipped.
        """
        try:
            content = TextReader(path).get_value()
        except Exception as exc:
            logger.warning("[FileIndexService] Failed to read %s: %s", path, exc)
            return

        filename = os.path.basename(path)
        try:
            stat = os.stat(path)
            mtime = stat.st_mtime
            size = stat.st_size
        except OSError as exc:
            logger.warning("[FileIndexService] Failed to stat %s: %s", path, exc)
            return

        try:
            with Database.transaction(self.db_path) as conn:
                conn.execute(
                    "DELETE FROM file_index_fts WHERE path = ?",
                    (path,),
                )
                conn.execute(
                    "INSERT INTO file_index_fts(path, filename, content, mtime, size) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (path, filename, content, mtime, size),
                )
            logger.debug("[FileIndexService] Indexed %s", path)
        except Exception as exc:
            logger.error("[FileIndexService] Failed to index %s: %s", path, exc)

    def upsert_content(self, path: str, content: str) -> None:
        """Upsert a row for *path* with the given *content*.

        Deletes any existing row for *path*, then inserts a new row with the
        file's content, filename, mtime, and size.  Per-file failures are
        logged and skipped.
        """
        filename = os.path.basename(path)
        try:
            stat = os.stat(path)
            mtime = stat.st_mtime
            size = stat.st_size
        except OSError as exc:
            logger.warning("[FileIndexService] Failed to stat %s: %s", path, exc)
            return

        try:
            with Database.transaction(self.db_path) as conn:
                conn.execute(
                    "DELETE FROM file_index_fts WHERE path = ?",
                    (path,),
                )
                conn.execute(
                    "INSERT INTO file_index_fts(path, filename, content, mtime, size) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (path, filename, content, mtime, size),
                )
            logger.debug("[FileIndexService] Upserted %s", path)
        except Exception as exc:
            logger.error("[FileIndexService] Failed to upsert %s: %s", path, exc)

    def remove_file(self, path: str) -> None:
        """Remove a file from the index.

        Deletes the row for *path*.  No-op if the row doesn't exist.
        """
        try:
            with Database.transaction(self.db_path) as conn:
                conn.execute(
                    "DELETE FROM file_index_fts WHERE path = ?",
                    (path,),
                )
            logger.debug("[FileIndexService] Removed %s from index", path)
        except Exception as exc:
            logger.error("[FileIndexService] Failed to remove %s: %s", path, exc)

    def clear(self) -> None:
        """Delete every row from the index (privacy delete-all).

        Raises on failure — unlike the watcher-thread methods above, this runs
        on a request thread whose caller reports the outcome. The reconcile
        scan repopulates the index from disk afterwards.
        """
        with Database.transaction(self.db_path) as conn:
            conn.execute("DELETE FROM file_index_fts")
        logger.info("[FileIndexService] Index cleared")

    def move_file(self, src: str, dst: str) -> None:
        """Move a file in the index.

        Deletes the row for *src*, then inserts a new row for *dst* with the
        same content, mtime, and size.  If *dst* cannot be read, the src row
        is left intact.
        """
        try:
            content = TextReader(dst).get_value()
        except Exception as exc:
            logger.warning("[FileIndexService] Failed to read %s after move: %s", dst, exc)
            return

        filename = os.path.basename(dst)
        try:
            stat = os.stat(dst)
            mtime = stat.st_mtime
            size = stat.st_size
        except OSError as exc:
            logger.warning("[FileIndexService] Failed to stat %s after move: %s", dst, exc)
            return

        try:
            with Database.transaction(self.db_path) as conn:
                conn.execute(
                    "DELETE FROM file_index_fts WHERE path = ?",
                    (src,),
                )
                conn.execute(
                    "INSERT INTO file_index_fts(path, filename, content, mtime, size) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (dst, filename, content, mtime, size),
                )
            logger.debug("[FileIndexService] Moved %s -> %s in index", src, dst)
        except Exception as exc:
            logger.error("[FileIndexService] Failed to move %s -> %s: %s", src, dst, exc)

    def search(self, query: str, limit: int = 10) -> list[str]:
        """Search the index for *query*.

        Returns a list of file paths matching the query, limited to *limit*
        results.  Uses FTS5's built-in ranking to order results by relevance.
        """
        try:
            rows = Database.conn(self.db_path).execute(
                "SELECT path FROM file_index_fts WHERE file_index_fts MATCH ? "
                "ORDER BY rank LIMIT ?",
                (query, limit),
            ).fetchall()
            return [row["path"] for row in rows]
        except Exception as exc:
            logger.error("[FileIndexService] Search failed: %s", exc)
            return []

    def should_index(self, path: str) -> bool:
        """Return True if *path* should be indexed.

        Applies all the should_index rules (see module-level constants).
        """
        return _should_index(path)

    def reconcile(self) -> None:
        """Reconcile the index with the filesystem.

        Walks the scan root, compares mtime/size of each file to the stored
        rows, and:
        - Indexes new or changed files.
        - Deletes rows whose files no longer exist.

        Parser-owned rows (files under the documents root with non-indexable
        extensions) are handled specially: they are kept as long as the file
        exists on disk (never re-extracted), and deleted when the file is gone.
        reconcile never CREATES parser-owned rows — only FileParserService does.

        Per-file failures are logged and skipped.  The walk never aborts.
        """
        try:
            # Get all indexed paths and their mtime/size.
            indexed_rows = Database.conn(self.db_path).execute(
                "SELECT path, mtime, size FROM file_index_fts"
            ).fetchall()
            indexed = {row["path"]: (row["mtime"], row["size"]) for row in indexed_rows}

            # Walk the scan root and collect all files.
            current: dict[str, tuple[float, int]] = {}
            for dirpath, dirnames, filenames in os.walk(self.scan_root):
                # Prune directories in-place to avoid descending into them.
                dirnames[:] = [
                    d
                    for d in dirnames
                    if not _is_dot_directory(d)
                    and d not in _SKIP_DIR_NAMES
                    and not _is_in_skip_subtree(os.path.join(dirpath, d))
                    and not _is_chalie_path(os.path.join(dirpath, d))
                    and not _is_library_caches(os.path.join(dirpath, d))
                ]

                for filename in filenames:
                    path = os.path.join(dirpath, filename)
                    if not _should_index(path):
                        continue
                    try:
                        stat = os.stat(path)
                        current[path] = (stat.st_mtime, stat.st_size)
                    except OSError:
                        continue
            for path, (mtime, size) in current.items():
                if path not in indexed or indexed[path] != (mtime, size):
                    self.index_file(path)

            # Delete rows for files that no longer exist. Parser-owned rows
            # are outside the walk (unsupported extensions), so presence on
            # disk — not membership in `current` — decides their fate.
            for path in indexed:
                if path not in current:
                    if _is_parser_owned(path) and os.path.isfile(path):
                        continue
                    self.remove_file(path)

            logger.info(
                "[FileIndexService] Reconciled: %d indexed, %d current, %d removed",
                len(indexed),
                len(current),
                len(set(indexed.keys()) - set(current.keys())),
            )
        except Exception as exc:
            logger.error("[FileIndexService] Reconcile failed: %s", exc)
