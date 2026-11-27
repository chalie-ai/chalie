"""Versioned main-database provisioning — one database file per release.

Every release opens its own ``data/chalie-<version>.sqlite``
(:meth:`services.file_mapper_service.FileMapperService.get_db_path`). On the
first boot of a new build the file does not exist yet, so this service builds
it from ``schema.sql`` and copies the previous release's data forward table by
table. An existing file is never touched again: no ALTER, no DROP, no in-place
DDL of any kind.

The copy is deliberately *not* one transaction. Each table is a single
autocommit INSERT-SELECT, so a table SQLite cannot read — a damaged b-tree
raises SQLITE_CORRUPT the moment the copy walks it — costs only itself: the
error is logged, the table is recorded in the lineage row, and the next table
copies normally. The alternative shipped in 1.3.0 (converge the live file
in-place inside one ``BEGIN IMMEDIATE``) turned one damaged table into a
poisoned transaction and a boot that could not complete.

``compactions`` is the exception when the source still finds its rows by
turn number. Its rows are rebuilt onto transcript row ids from the source's own
transcript, and if that fails provisioning stops instead of passing over the
table: those rows are the conversation's checkpoints, and losing them silently
would change what the person's conversation shows. The old database is only
read, so it stays intact, and the next boot rebuilds from it.

A source from before threads had rows of their own gets them the same way:
one thread per channel and turn number its transcript holds, and every
transcript row pointed at its thread. Its thread labels, kept then in a table
of their own, move onto the threads they name. That too stops provisioning on
failure — a row that lost its thread would drop out of the conversation it
belongs to, and a lost label would leave its thread unnamed.

Boot order (``run.py``): the snapshot restore swaps artifacts in, then
:meth:`VersionedDatabaseService.provision` runs BEFORE the first
``Database.conn()`` — nothing may open the main database ahead of it.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path

from services.app_version import get_version, version_sort_key
from services.file_mapper_service import FileMapperService
from services.time_utils import utc_now

logger = logging.getLogger(__name__)

_LOG = "[VersionedDB]"

# SQLite WAL-mode sidecars. They belong to their database file and are removed
# with it — a stale ``-wal`` left beside a recreated file would replay frames
# of the dead database onto the new one.
_SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm")

# How many database files an install keeps. Three is the rollback window: the
# running release, the one it upgraded from, and one before that.
_RETAINED_DATABASE_FILES = 3

# Lineage bookkeeping. The row is written last, so its presence is the proof
# that a file was fully provisioned; ``failed_tables`` names every table the
# copy could not read, separated by this character.
_LINEAGE_TABLE = "database_lineage"
_FAILED_TABLES_SEPARATOR = ","

# A target file without a lineage row is moved to
# ``<name>.incomplete-<timestamp>`` rather than deleted. The name must not end
# in ``.sqlite``, or the source scan and retention would read it as a release
# database; microseconds keep two asides made in the same second from
# overwriting each other.
_ASIDE_INFIX = ".incomplete-"
_ASIDE_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%S.%fZ"

# ``CREATE VIRTUAL TABLE <name> USING <module>(...)`` — the module name decides
# how a virtual table is carried forward: vec0 rows copy, fts5 indexes rebuild.
_VIRTUAL_MODULE_RE = re.compile(r"\bUSING\s+(\w+)", re.IGNORECASE)
_FTS5_MODULE = "fts5"
_VEC0_MODULE = "vec0"

# The soft-delete flag providers used to carry. A source still holding it has
# rows parked at 0 that were deleted by the user; carrying them forward would
# resurrect them as active providers and collide on the UNIQUE name index.
_PROVIDERS_TABLE = "providers"
_PROVIDERS_ACTIVE_COLUMN = "is_active"

# Schema name the source database is ATTACHed under for the copy.
_SOURCE_SCHEMA = "src"

# Checkpoints used to find their rows by turn number; they now find them by
# transcript row id. A source still in the old shape holds the turn number in a
# column the target does not have, so the shared-column copy would leave it
# behind and no reader could find the row again. Each such table is carried by
# its own INSERT-SELECT, keyed here by the column only the old shape has, that
# computes the new columns from the source's transcript. Transcript ids carry
# forward unchanged, so an id read there names the same row here.
#
# compactions, FORK row (for_turn_id set): compacted_up_to is already a row id
# and stays as it is; opening_id is the thread's first row, the rule
# ``Transcript.first_id`` uses. A thread with no rows matches nothing and BOTH
# columns stay NULL, so the row is never read: an empty opening_id with a cut
# would read as the spine's checkpoint.
#
# compactions, MAIN row (for_turn_id empty): compacted_up_to is a turn number.
# The cut becomes the row just below the first non-memory row of a later turn,
# so every row the old read showed still shows; with no later turn it is the
# channel's last row, and on a channel with no rows 0, so the checkpoint's
# summary is always read. Where turns overlapped, a row of a folded turn can
# sit above the cut and shows again: that errs on the side of showing a row
# twice, never of hiding one.
_TURN_AXIS_BACKFILLS: dict[str, tuple[str, str]] = {
    "compactions": (
        "for_turn_id",
        "INSERT INTO main.compactions "
        "(rowid, id, channel, opening_id, compacted_up_to_id, content, created_at) "
        "SELECT c.rowid, c.id, c.channel, "
        "CASE WHEN c.for_turn_id IS NOT NULL THEN "
        f"(SELECT MIN(t.id) FROM {_SOURCE_SCHEMA}.transcript t "
        "WHERE t.channel = c.channel AND t.turn_id = c.for_turn_id) END, "
        "CASE WHEN c.for_turn_id IS NULL THEN COALESCE("
        f"(SELECT MIN(t.id) FROM {_SOURCE_SCHEMA}.transcript t "
        "WHERE t.channel = c.channel AND t.turn_id > c.compacted_up_to "
        "AND t.role != 'memory') - 1, "
        f"(SELECT MAX(t.id) FROM {_SOURCE_SCHEMA}.transcript t "
        "WHERE t.channel = c.channel), "
        "0) "
        f"WHEN EXISTS (SELECT 1 FROM {_SOURCE_SCHEMA}.transcript t "
        "WHERE t.channel = c.channel AND t.turn_id = c.for_turn_id) "
        "THEN c.compacted_up_to END, "
        "c.content, c.created_at "
        f"FROM {_SOURCE_SCHEMA}.compactions c",
    ),
}

# A thread used to be only the (channel, turn_id) pair its transcript rows
# share; it is now a row of its own that every transcript row points at. A source
# whose transcript has no thread column gets one thread per pair the copied rows
# hold, numbered in the order the threads began, and each row is pointed at its
# thread. A row with no turn number was written before turns existed and belongs
# to no thread. Run in order, each as one autocommit statement, after every table
# has been copied.
_THREAD_LINK_COLUMN = "thread_id"
_THREAD_BACKFILLS: tuple[tuple[str, str], ...] = (
    (
        "threads",
        "INSERT INTO main.threads (channel, turn_id) "
        "SELECT channel, turn_id FROM main.transcript WHERE turn_id IS NOT NULL "
        "GROUP BY channel, turn_id ORDER BY MIN(id)",
    ),
    (
        "transcript",
        "UPDATE main.transcript SET thread_id = (SELECT th.id FROM main.threads th "
        "WHERE th.channel = transcript.channel AND th.turn_id = transcript.turn_id) "
        "WHERE turn_id IS NOT NULL",
    ),
)

# A thread's label used to sit in a table of its own; it is now the thread's
# ``summary``. A source that still has that table gets each label written onto
# the thread it names, after the threads exist. Two shapes shipped: the older
# one names the thread by its channel and turn number, the newer one by the
# thread's first transcript row. A label whose thread has no rows names no
# thread and is not carried, as it was never read before either.
_LABEL_TABLE = "thread_gist"
_LABEL_BY_OPENING_ROW = "opening_id"
_LABEL_BACKFILLS: dict[str, str] = {
    _LABEL_BY_OPENING_ROW: (
        "UPDATE main.threads SET summary = (SELECT g.gist "
        f"FROM {_SOURCE_SCHEMA}.thread_gist g JOIN main.transcript t ON t.id = g.opening_id "
        "WHERE t.thread_id = threads.id)"
    ),
    "turn_id": (
        "UPDATE main.threads SET summary = (SELECT g.gist "
        f"FROM {_SOURCE_SCHEMA}.thread_gist g "
        "WHERE g.channel = threads.channel AND g.turn_id = threads.turn_id)"
    ),
}

# A thread's thinking level used to be stored on each of its transcript rows; it
# is now the thread's own. A source whose transcript still holds it gets each
# thread's newest valid level written onto the thread, after the threads exist;
# a thread with none keeps the default.
_LEVEL_COLUMN = "thinking_level"
_LEVEL_BACKFILL = (
    "UPDATE main.threads SET thinking_level = COALESCE((SELECT s.thinking_level "
    f"FROM {_SOURCE_SCHEMA}.transcript s "
    "WHERE s.channel = threads.channel AND s.turn_id = threads.turn_id "
    "AND s.thinking_level IN ('auto', 'medium', 'high') ORDER BY s.id DESC LIMIT 1), "
    "thinking_level)"
)

# The implicit rowid, named explicitly in every copy: it is the join key between a
# content table and its vec0/FTS5 sidecars, and no declared column carries it on
# a TEXT-keyed table. vec0 tables report it through table_info; ordinary tables do
# not, so the copy adds it itself.
_ROWID = "rowid"


class VersionedDatabaseService:
    """Provisions the running release's main database file.

    No-arg constructor — every path resolves through ``FileMapperService`` at
    call time, so a redirected layout (tests, alternate data roots) is honoured
    without injection here.
    """

    def __init__(self) -> None:
        self._fm = FileMapperService

    # ── Entry point ──────────────────────────────────────────────────────────

    def provision(self) -> None:
        """Ensure the running release's database file exists and holds the
        previous release's data.

        A no-op once the file carries its lineage row. Otherwise: move any file
        standing at the target path aside, build the schema, copy the newest
        older database forward, apply the declarative seeds, stamp the lineage
        row, and prune database files past the retention window.
        """
        target = self._fm.get_db_path()
        version = get_version()

        if target.exists():
            if self._is_provisioned(target, version):
                logger.info(f"{_LOG} {target.name} already provisioned for {version}")
                return
            # No lineage row means the file was not fully provisioned by this
            # build — usually an earlier boot that died mid-copy, but a snapshot
            # restore lands a database here too, and that one may be the only
            # copy of its data. It is moved aside, never unlinked: an aside file
            # costs a name and stays openable, deleting one is silent data loss.
            aside = self._set_aside(target)
            logger.warning(
                f"{_LOG} {target} has no lineage row for {version} — "
                f"it was not provisioned by this build; moved to {aside} "
                "and rebuilding from the newest earlier database"
            )

        source = self._select_source(target.parent, version)
        logger.info(
            f"{_LOG} provisioning {target.name} from "
            f"{source.name if source else 'a fresh schema (no earlier database)'}"
        )

        target.parent.mkdir(parents=True, exist_ok=True)
        conn = self._create_schema(target)
        try:
            failed = self._copy_forward(conn, source) if source else []
            self._apply_seeds(conn)
            self._write_lineage(conn, version, source, failed)
        finally:
            conn.close()

        if failed:
            logger.error(
                f"{_LOG} {target.name} provisioned with "
                f"{len(failed)} unreadable table(s): {', '.join(failed)}"
            )
        else:
            logger.info(f"{_LOG} {target.name} provisioned")

        self._prune_old_databases(target)

    # ── Existing target ──────────────────────────────────────────────────────

    def _is_provisioned(self, target: Path, version: str) -> bool:
        """Return True when *target* carries a lineage row for *version*.

        Opened read-only: an existing file is never written by this service.
        A file too damaged to answer the question is treated as unprovisioned
        rather than crashing the boot — the caller rebuilds it.
        """
        try:
            conn = sqlite3.connect(f"file:{target}?mode=ro", uri=True)
        except sqlite3.Error as exc:
            logger.warning(f"{_LOG} cannot open {target.name}: {exc}")
            return False
        try:
            row = conn.execute(
                f"SELECT 1 FROM {_LINEAGE_TABLE} WHERE version = ?", (version,)
            ).fetchone()
            return row is not None
        except sqlite3.Error as exc:
            logger.warning(f"{_LOG} cannot read the lineage of {target.name}: {exc}")
            return False
        finally:
            conn.close()

    def _set_aside(self, path: Path) -> Path:
        """Move a database file and its WAL sidecars out of the way, intact.

        The aside name deliberately does not end in ``.sqlite``, so neither the
        source scan nor retention can mistake it for a release database: it sits
        there until an operator reads it or removes it.

        The sidecars move first and keep their suffix relative to the new name,
        so the WAL stays attached to the file whose committed frames it holds
        and can never replay onto the fresh database taking the old name.
        """
        aside = path.with_name(
            f"{path.name}{_ASIDE_INFIX}{utc_now().strftime(_ASIDE_TIMESTAMP_FORMAT)}"
        )
        for suffix in _SQLITE_SIDECAR_SUFFIXES:
            sidecar = path.with_name(path.name + suffix)
            if sidecar.exists():
                sidecar.rename(aside.with_name(aside.name + suffix))
        path.rename(aside)
        return aside

    def _remove_database_file(self, path: Path) -> None:
        """Delete a database file together with its WAL sidecars."""
        path.unlink()
        for suffix in _SQLITE_SIDECAR_SUFFIXES:
            path.with_name(path.name + suffix).unlink(missing_ok=True)

    # ── Source selection ─────────────────────────────────────────────────────

    def _select_source(self, data_dir: Path, version: str) -> Path | None:
        """Return the newest database file older than *version*, or None.

        A file whose version is equal to or newer than the running one is never
        a source: downgrading must re-open the older release's own file, never
        graft a newer schema's rows onto it. The pre-versioning ``chalie.db``
        is the fallback — by definition it predates every versioned file.
        """
        running = version_sort_key(version)
        older = [
            path
            for path, found in self._versioned_databases(data_dir)
            if version_sort_key(found) < running
        ]
        if older:
            return older[-1]
        legacy = self._legacy_database(data_dir)
        return legacy if legacy.exists() else None

    def _legacy_database(self, data_dir: Path) -> Path:
        """The pre-versioning ``chalie.db`` inside *data_dir*.

        The name comes from ``FileMapperService``, the directory from the
        target: every file this service reads or writes lives beside the
        database it is provisioning, whatever data root that resolves to.
        """
        return data_dir / self._fm.get_legacy_db_path().name

    def _versioned_databases(self, data_dir: Path) -> list[tuple[Path, str]]:
        """Every ``chalie-<version>.sqlite`` in *data_dir*, oldest version first.

        A file whose name carries something this build cannot read as a version
        is not a Chalie release database (an operator's own copy, say): it is
        reported and skipped rather than crashing the boot or being mistaken
        for one.
        """
        found: list[tuple[Path, str]] = []
        for path in sorted(data_dir.glob("*")) if data_dir.is_dir() else []:
            version = self._fm.version_from_db_path(path)
            if version is None:
                continue
            try:
                version_sort_key(version)
            except ValueError:
                logger.warning(f"{_LOG} ignoring {path.name}: {version!r} is not a version")
                continue
            found.append((path, version))
        return sorted(found, key=lambda pair: version_sort_key(pair[1]))

    # ── Schema creation ──────────────────────────────────────────────────────

    def _create_schema(self, target: Path) -> sqlite3.Connection:
        """Create *target* from ``schema.sql`` and return its open connection.

        Everything but the seed INSERTs runs here; the seeds wait until after
        the copy so a carried-forward row wins over its default. Any failure
        propagates: a boot that cannot build the schema must stop, never serve
        a half-built database.
        """
        conn = sqlite3.connect(str(target))
        conn.isolation_level = None  # autocommit: one statement per table, never a transaction spanning the copy
        conn.enable_load_extension(True)
        import sqlite_vec
        sqlite_vec.load(conn)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(self._schema_statements(seeds=False))
        return conn

    def _apply_seeds(self, conn: sqlite3.Connection) -> None:
        """Run ``schema.sql``'s declarative seeds after the copy.

        They are all ``INSERT OR IGNORE``, so a row copied from the previous
        release keeps the user's value and only genuinely missing defaults land.
        """
        conn.executescript(self._schema_statements(seeds=True))

    def _schema_statements(self, seeds: bool) -> str:
        """Return the seed INSERTs of ``schema.sql`` (*seeds*) or everything else.

        Statement boundaries come from :func:`sqlite3.complete_statement`, which
        is SQLite's own scanner: it accounts for comments, quoted strings and
        ``CREATE TRIGGER ... BEGIN ... END`` bodies. Comment-only text rides
        along with the surrounding chunk and is harmless to execute.
        """
        schema_sql = self._fm.get_schema_path().read_text(encoding="utf-8")
        chunks: list[str] = []
        buffer = ""
        for line in schema_sql.splitlines(keepends=True):
            buffer += line
            if sqlite3.complete_statement(buffer):
                chunks.append(buffer)
                buffer = ""
        if buffer.strip():
            chunks.append(buffer)
        return "".join(c for c in chunks if self._is_seed(c) is seeds)

    @staticmethod
    def _is_seed(chunk: str) -> bool:
        """Return True when *chunk*'s statement (past any leading comments) inserts."""
        body = "\n".join(
            line for line in chunk.splitlines() if not line.strip().startswith("--")
        ).strip()
        return body.upper().startswith("INSERT")

    # ── Copy forward ─────────────────────────────────────────────────────────

    def _copy_forward(self, conn: sqlite3.Connection, source: Path) -> list[str]:
        """Copy every table *source* shares with the target; return the failures.

        The source is ATTACHed and only read — opened normally rather than
        immutable, because a live install's newest rows may still sit in its
        WAL. Foreign keys are held off for the duration, stated rather than
        inherited from SQLite's default: tables copy in whatever order they
        appear, so a child would otherwise be rejected ahead of its parent.

        A table the source still holds in its turn-axis shape, the threads of
        a source from before they had rows (:data:`_THREAD_BACKFILLS`), their
        labels (:data:`_LABEL_BACKFILLS`) and thinking levels
        (:data:`_LEVEL_BACKFILL`) are carried by
        :meth:`_backfill_table`, which raises rather than returning a failure.
        """
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute(f"ATTACH DATABASE ? AS {_SOURCE_SCHEMA}", (str(source),))
        try:
            failed: list[str] = []
            source_tables = self._table_names(conn, _SOURCE_SCHEMA)
            for table in self._copyable_tables(conn):
                if table not in source_tables:
                    continue
                backfill = self._backfill_statement(conn, table)
                if backfill is not None:
                    self._backfill_table(conn, table, backfill, source)
                elif not self._copy_table(conn, table):
                    failed.append(table)
            if "transcript" in source_tables and _THREAD_LINK_COLUMN not in self._columns(
                conn, _SOURCE_SCHEMA, "transcript"
            ):
                for table, statement in _THREAD_BACKFILLS:
                    self._backfill_table(conn, table, statement, source)
            if _LABEL_TABLE in source_tables:
                self._backfill_table(conn, _LABEL_TABLE, self._label_backfill(conn), source)
            if "transcript" in source_tables and _LEVEL_COLUMN in self._columns(
                conn, _SOURCE_SCHEMA, "transcript"
            ):
                self._backfill_table(conn, "threads", _LEVEL_BACKFILL, source)
            for table, module in self._virtual_tables(conn, "main").items():
                if module == _FTS5_MODULE and not self._rebuild_index(conn, table):
                    failed.append(table)
            self._carry_sequences(conn)
            return failed
        finally:
            conn.execute(f"DETACH DATABASE {_SOURCE_SCHEMA}")

    def _copy_table(self, conn: sqlite3.Connection, table: str) -> bool:
        """Copy one table, rowid included, on the columns both files share; return
        False on failure.

        One autocommit statement, so a table SQLite cannot read rolls back only
        itself. The error is logged with the table name because that name is
        what tells the owner which data was lost.
        """
        columns = self._shared_columns(conn, table)
        if not columns:
            logger.warning(f"{_LOG} {table}: no column in common with the source, skipped")
            return True
        if _ROWID not in columns:
            # SQLite hands every inserted row a fresh rowid unless the statement
            # names it. A deleted episode leaves a gap in the source, so the copy
            # would renumber the survivors and every episode after the gap would
            # sit on another episode's embedding: recall keeps answering, with
            # the wrong memory.
            columns = [_ROWID, *columns]
        projection = ", ".join(f'"{column}"' for column in columns)
        statement = (
            f'INSERT INTO main."{table}" ({projection}) '
            f'SELECT {projection} FROM {_SOURCE_SCHEMA}."{table}"{self._copy_filter(conn, table)}'
        )
        try:
            conn.execute(statement)
        except sqlite3.Error as exc:
            logger.error(f"{_LOG} {table}: not copied — {exc}")
            return False
        return True

    def _copy_filter(self, conn: sqlite3.Connection, table: str) -> str:
        """Return the WHERE clause restricting which source rows carry forward.

        Only ``providers`` has one: provider deletion is permanent now, so a
        source still carrying the old ``is_active`` flag holds rows the user
        already deleted. They must not come back.
        """
        if table != _PROVIDERS_TABLE:
            return ""
        if _PROVIDERS_ACTIVE_COLUMN not in self._columns(conn, _SOURCE_SCHEMA, table):
            return ""
        return f" WHERE {_PROVIDERS_ACTIVE_COLUMN} = 1"

    def _backfill_statement(self, conn: sqlite3.Connection, table: str) -> str | None:
        """Return the INSERT-SELECT that carries *table* out of a turn-axis
        source, or None when the shared-column copy applies.

        Only ``compactions`` has one, and only while the source still has the
        column that held the turn number
        (:data:`_TURN_AXIS_BACKFILLS`). A source already in the row-id shape
        copies like any other table.
        """
        backfill = _TURN_AXIS_BACKFILLS.get(table)
        if backfill is None:
            return None
        legacy_column, statement = backfill
        if legacy_column not in self._columns(conn, _SOURCE_SCHEMA, table):
            return None
        return statement

    def _label_backfill(self, conn: sqlite3.Connection) -> str:
        """Return the UPDATE that writes the source's thread labels onto the
        threads they name, picked by the shape the source's label table has."""
        columns = self._columns(conn, _SOURCE_SCHEMA, _LABEL_TABLE)
        shape = _LABEL_BY_OPENING_ROW if _LABEL_BY_OPENING_ROW in columns else "turn_id"
        return _LABEL_BACKFILLS[shape]

    def _backfill_table(
        self, conn: sqlite3.Connection, table: str, statement: str, source: Path
    ) -> None:
        """Run one table's backfill; a failure stops provisioning.

        One autocommit statement, so a failure writes none of the table's rows.
        Unlike :meth:`_copy_table` it is not recorded and passed over: these
        rows are the conversation's checkpoints, its threads, their labels and
        thinking levels, and losing them silently would change what the person's conversation
        shows. The error is logged before it is raised, so it reaches the log
        file and not only the console. No lineage row is written, so the next
        boot sets this file aside and rebuilds it from the same source.
        """
        try:
            conn.execute(statement)
        except sqlite3.Error as exc:
            message = (
                f"{_LOG} {table}: could not carry its rows forward from {source} — {exc}. "
                f"Nothing was lost: {source.name} is intact, and the next boot "
                "rebuilds this database from it."
            )
            logger.error(message)
            raise RuntimeError(message) from exc

    def _rebuild_index(self, conn: sqlite3.Connection, table: str) -> bool:
        """Rebuild one FTS5 index from its content table; return False on failure.

        Every FTS5 table in the schema is external-content, so its rows are not
        copied — they are re-derived from the content table that just was. Run
        for all of them unconditionally: rebuilding an index whose content
        table is empty is a no-op.
        """
        try:
            conn.execute(f'INSERT INTO main."{table}"("{table}") VALUES(\'rebuild\')')
        except sqlite3.Error as exc:
            logger.error(f"{_LOG} {table}: index not rebuilt — {exc}")
            return False
        return True

    def _carry_sequences(self, conn: sqlite3.Connection) -> None:
        """Carry each AUTOINCREMENT counter forward so no id is ever reused.

        Copying rows only lifts a counter to the highest id present; a source
        that allocated ids and then deleted those rows sits higher than that.
        Taking the larger of the two keeps a deleted row's id from being handed
        to a new one.
        """
        if not self._has_sequence_table(conn, _SOURCE_SCHEMA):
            return
        target_tables = self._table_names(conn, "main")
        rows = conn.execute(
            f"SELECT name, seq FROM {_SOURCE_SCHEMA}.sqlite_sequence"
        ).fetchall()
        for name, seq in rows:
            if name not in target_tables:
                continue
            current = conn.execute(
                "SELECT seq FROM main.sqlite_sequence WHERE name = ?", (name,)
            ).fetchone()
            if current is None:
                conn.execute(
                    "INSERT INTO main.sqlite_sequence (name, seq) VALUES (?, ?)", (name, seq)
                )
            elif current[0] < seq:
                conn.execute(
                    "UPDATE main.sqlite_sequence SET seq = ? WHERE name = ?", (seq, name)
                )

    # ── Lineage ──────────────────────────────────────────────────────────────

    def _write_lineage(
        self, conn: sqlite3.Connection, version: str, source: Path | None, failed: list[str]
    ) -> None:
        """Stamp the lineage row — the last write of a provisioning.

        Written only once everything else has landed, so a file carrying the
        row is by definition complete. The rows travel with the database, so a
        file states its own upgrade history: which release built it, from what,
        and what did not survive. OR REPLACE because the source may itself carry
        a row for the running version (this build provisioned it too, before a
        downgrade): the copy describes the source's history, this row describes
        THIS file's, and for this file's own version the latter is the truth.
        """
        conn.execute(
            f"INSERT OR REPLACE INTO {_LINEAGE_TABLE} "
            "(version, source_file, completed_at, failed_tables) "
            "VALUES (?, ?, ?, ?)",
            (
                version,
                source.name if source else None,
                utc_now().isoformat(),
                _FAILED_TABLES_SEPARATOR.join(failed),
            ),
        )

    # ── Retention ────────────────────────────────────────────────────────────

    def _prune_old_databases(self, target: Path) -> None:
        """Keep the newest :data:`_RETAINED_DATABASE_FILES` database files.

        The legacy ``chalie.db`` counts as the oldest — it predates versioning.
        The file this boot just provisioned is never pruned, however the
        versions around it sort: a downgrade would otherwise delete the
        database it is about to open.
        """
        legacy = self._legacy_database(target.parent)
        oldest_first = [legacy] if legacy.exists() else []
        oldest_first += [path for path, _ in self._versioned_databases(target.parent)]

        for path in oldest_first[:-_RETAINED_DATABASE_FILES]:
            if path == target:
                continue
            logger.info(f"{_LOG} pruning {path.name} (past the {_RETAINED_DATABASE_FILES}-file retention window)")
            self._remove_database_file(path)

    # ── SQLite introspection ─────────────────────────────────────────────────

    def _copyable_tables(self, conn: sqlite3.Connection) -> list[str]:
        """Target tables whose rows copy directly: ordinary tables plus vec0.

        FTS5 tables are excluded — they are rebuilt from their content table.
        So are the shadow tables FTS5 and vec0 create for their own bookkeeping
        (``<virtual>_data``, ``<virtual>_chunks``, …): writing to one behind its
        module's back corrupts the index it belongs to.
        """
        virtual = self._virtual_tables(conn, "main")
        shadow_prefixes = tuple(f"{name}_" for name in virtual)
        return [
            table
            for table in self._table_names(conn, "main")
            if table not in virtual and not table.startswith(shadow_prefixes)
        ] + [table for table, module in virtual.items() if module == _VEC0_MODULE]

    @staticmethod
    def _table_names(conn: sqlite3.Connection, schema: str) -> set[str]:
        """Every table name in *schema*, SQLite's own internals excluded."""
        rows = conn.execute(
            f"SELECT name FROM {schema}.sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite\\_%' ESCAPE '\\'"
        ).fetchall()
        return {name for (name,) in rows}

    @staticmethod
    def _has_sequence_table(conn: sqlite3.Connection, schema: str) -> bool:
        """Return True when *schema* has a ``sqlite_sequence`` table.

        Absent until the schema declares its first AUTOINCREMENT table, so a
        source that has none carries no counters to read.
        """
        row = conn.execute(
            f"SELECT 1 FROM {schema}.sqlite_master WHERE name = 'sqlite_sequence'"
        ).fetchone()
        return row is not None

    @staticmethod
    def _virtual_tables(conn: sqlite3.Connection, schema: str) -> dict[str, str]:
        """Map each virtual table in *schema* to its module name (fts5, vec0)."""
        rows = conn.execute(
            f"SELECT name, sql FROM {schema}.sqlite_master "
            "WHERE type = 'table' AND sql LIKE 'CREATE VIRTUAL TABLE%'"
        ).fetchall()
        modules: dict[str, str] = {}
        for name, sql in rows:
            match = _VIRTUAL_MODULE_RE.search(sql)
            if match is None:
                logger.warning(f"{_LOG} {name}: cannot read its module from the DDL, skipped")
                continue
            modules[name] = match.group(1).lower()
        return modules

    @staticmethod
    def _columns(conn: sqlite3.Connection, schema: str, table: str) -> list[str]:
        """Column names of *table* in *schema*, in declaration order.

        vec0 tables report their ``rowid`` here alongside the vector columns,
        which is exactly what a copy of one needs to preserve.
        """
        rows = conn.execute(f'PRAGMA {schema}.table_info("{table}")').fetchall()
        return [str(row[1]) for row in rows]

    def _shared_columns(self, conn: sqlite3.Connection, table: str) -> list[str]:
        """Columns *table* has in both files, in the target's declaration order.

        The intersection is what makes an upgrade and a downgrade the same
        operation: a column the source no longer has keeps the target's
        default, and a column only the source has is left behind.
        """
        source_columns = set(self._columns(conn, _SOURCE_SCHEMA, table))
        return [c for c in self._columns(conn, "main", table) if c in source_columns]
