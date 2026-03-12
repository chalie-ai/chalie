"""
Database Service — SQLite connection management with thread-local connections.

Replaces the PostgreSQL/SQLAlchemy implementation with sqlite3 + sqlite-vec + FTS5.
Each thread gets its own connection via threading.local(). WAL mode enables
concurrent reads during writes.
"""

import logging
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


class _TextClause:
    """Lightweight replacement for sqlalchemy.text().
    Just wraps a SQL string so SessionProxy.execute(str(obj)) works."""
    __slots__ = ('_sql',)

    def __init__(self, sql: str):
        """Initialize with a raw SQL string.

        Args:
            sql: The SQL statement to wrap.
        """
        self._sql = sql

    def __str__(self):
        """Return the underlying SQL string."""
        return self._sql

    def __repr__(self):
        """Return a developer-friendly representation."""
        return f"text({self._sql!r})"


def text(sql: str) -> _TextClause:
    """Wrap a raw SQL string as a text clause (drop-in for sqlalchemy.text()).

    Args:
        sql: The SQL statement string to wrap.

    Returns:
        A :class:`_TextClause` instance whose ``str()`` yields the SQL.
    """
    return _TextClause(sql)

# Default database path
_DEFAULT_DB_PATH = str(Path(__file__).resolve().parent.parent / "data" / "chalie.db")

# ── Thread-local storage for connections ────────────────────────
_local = threading.local()

# ── Singleton DatabaseService ───────────────────────────────────
_shared_db_service = None
_shared_lock = threading.Lock()


def get_db_path() -> str:
    """Return the database file path from the environment or the built-in default.

    Returns:
        Absolute path to the SQLite file, sourced from the ``CHALIE_DB_PATH``
        environment variable when set, otherwise the default path inside the
        Docker data volume.
    """
    return os.environ.get("CHALIE_DB_PATH", _DEFAULT_DB_PATH)


def get_shared_db_service() -> 'DatabaseService':
    """Return the process-wide shared DatabaseService singleton, creating it if needed.

    Returns:
        The singleton :class:`DatabaseService` instance.  Thread-safe via a
        double-checked lock pattern.
    """
    global _shared_db_service
    if _shared_db_service is None:
        with _shared_lock:
            if _shared_db_service is None:
                _shared_db_service = DatabaseService(get_db_path())
                logger.info("[DB] Created shared DatabaseService singleton")
    return _shared_db_service


# Kept for compatibility — SQLite needs no lightweight variant
get_lightweight_db_service = get_shared_db_service


class SessionProxy:
    """
    Lightweight shim that mimics SQLAlchemy session.execute(text("SQL"), params).
    Allows code that used get_session() to work with raw SQLite connections.

    Usage:
        with db.get_session() as session:
            result = session.execute(text("SELECT * FROM t WHERE id = :id"), {"id": 1})
            rows = result.fetchall()
    """

    def __init__(self, conn: sqlite3.Connection):
        """Wrap an existing sqlite3.Connection for SQL execution.

        Args:
            conn: An open sqlite3.Connection to delegate all SQL operations to.
        """
        self._conn = conn

    def execute(self, sql_or_text, params=None):
        """Execute a SQL statement and return a :class:`ResultProxy`.

        Accepts either a raw SQL string or a :class:`_TextClause` produced by
        :func:`text`.  Dict-style named parameters (``":name"`` placeholders)
        are transparently converted to SQLite positional ``"?"`` parameters.

        Args:
            sql_or_text: SQL string or :class:`_TextClause` instance to execute.
            params: Optional dict of named bind values (``{"name": value}``) or
                a sequence for positional binding.  Pass ``None`` for statements
                with no parameters.

        Returns:
            A :class:`ResultProxy` wrapping the executed ``sqlite3.Cursor``,
            supporting ``fetchone()``, ``fetchall()``, and ``scalar()``.
        """
        # Extract the string from sqlalchemy text() objects
        sql = str(sql_or_text)

        if params and isinstance(params, dict):
            # Convert :name params to ? positional params
            import re
            ordered_params = []
            def _replace(match):
                """Capture a named placeholder, append its value, return ``?``."""
                key = match.group(1)
                ordered_params.append(params[key])
                return '?'
            sql = re.sub(r':(\w+)', _replace, sql)
            cursor = self._conn.cursor()
            cursor.execute(sql, ordered_params)
        else:
            cursor = self._conn.cursor()
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)

        return ResultProxy(cursor)

    def commit(self):
        """Explicit commit (also happens automatically on context-manager exit)."""
        self._conn.commit()

    def rollback(self):
        """Explicit rollback."""
        self._conn.rollback()


class ResultProxy:
    """Wraps sqlite3.Cursor to mimic SQLAlchemy result set."""

    def __init__(self, cursor: sqlite3.Cursor):
        """Wrap an existing sqlite3.Cursor to mimic SQLAlchemy result sets.

        Args:
            cursor: An open sqlite3.Cursor used for all fetch operations.
        """
        self._cursor = cursor

    def fetchone(self):
        """Fetch the next row from the result set.

        Returns:
            A ``sqlite3.Row`` that supports both integer index (``row[0]``) and
            column-name access (``row["col"]``), or ``None`` if no further rows
            are available.
        """
        row = self._cursor.fetchone()
        return row  # sqlite3.Row already supports int and key indexing

    def fetchall(self):
        """Fetch all remaining rows from the result set.

        Returns:
            A list of ``sqlite3.Row`` objects, each supporting both integer
            index and column-name access.  Returns an empty list when no rows
            remain.
        """
        return self._cursor.fetchall()

    @property
    def rowcount(self):
        """Number of rows affected by the last statement."""
        return self._cursor.rowcount

    @property
    def lastrowid(self):
        """Row ID of the last inserted row."""
        return self._cursor.lastrowid

    def scalar(self):
        """Fetch the first column of the first row, or None if no rows.

        Returns:
            The scalar value or None.
        """
        row = self.fetchone()
        if row is None:
            return None
        return row[0]

    def close(self):
        """Close the underlying cursor, releasing its resources."""
        self._cursor.close()


class DictCursor:
    """Wraps sqlite3.Cursor to return list[dict] from fetchall()."""

    def __init__(self, cursor: sqlite3.Cursor):
        """Wrap an existing sqlite3.Cursor.

        Args:
            cursor: An open sqlite3.Cursor to delegate all operations to.
        """
        self._cursor = cursor

    def execute(self, sql, params=None):
        """Execute a single SQL statement.

        Args:
            sql: SQL statement string.
            params: Optional sequence or mapping of bind parameters.

        Returns:
            self, for chaining.
        """
        if params is None:
            self._cursor.execute(sql)
        else:
            self._cursor.execute(sql, params)
        return self

    def executemany(self, sql, params_list):
        """Execute a SQL statement against a sequence of parameter sets.

        Args:
            sql: SQL statement string.
            params_list: Iterable of parameter sequences or mappings.

        Returns:
            self, for chaining.
        """
        self._cursor.executemany(sql, params_list)
        return self

    def fetchone(self):
        """Fetch the next row as a dict, or None if no more rows.

        Returns:
            dict of column→value for the next row, or None.
        """
        row = self._cursor.fetchone()
        if row is None:
            return None
        return dict(row)

    def fetchall(self):
        """Fetch all remaining rows as a list of dicts.

        Returns:
            List of column→value dicts.
        """
        return [dict(row) for row in self._cursor.fetchall()]

    @property
    def lastrowid(self):
        """Row ID of the last inserted row."""
        return self._cursor.lastrowid

    @property
    def rowcount(self):
        """Number of rows affected by the last statement."""
        return self._cursor.rowcount

    @property
    def description(self):
        """Sequence of 7-item sequences describing each result column."""
        return self._cursor.description

    def close(self):
        """Close the cursor, releasing database resources."""
        self._cursor.close()

    def __iter__(self):
        """Iterate over result rows as dicts."""
        return (dict(row) for row in self._cursor)


class DatabaseService:
    """Manages SQLite connections with thread-local isolation and WAL mode."""

    def __init__(self, db_path: str = None):
        """Initialize the service and ensure the database directory exists.

        Args:
            db_path: Absolute path to the SQLite file. Defaults to the value
                returned by :func:`get_db_path` (env var or built-in default).
        """
        self.db_path = db_path or get_db_path()
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """Return the thread-local SQLite connection, creating it if necessary.

        On first call from a given thread (or when ``db_path`` has changed)
        opens a new connection, enables WAL journalling, foreign-key enforcement,
        a 5-second busy timeout, ``NORMAL`` synchronisation, and in-memory temp
        storage, then attempts to load the ``sqlite-vec`` extension.
        Subsequent calls from the same thread return the cached connection.

        Returns:
            An open ``sqlite3.Connection`` with ``row_factory`` set to
            ``sqlite3.Row`` so columns are accessible by name.
        """
        conn = getattr(_local, 'conn', None)
        db_path = getattr(_local, 'db_path', None)

        if conn is None or db_path != self.db_path:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

            conn = sqlite3.connect(self.db_path, timeout=10)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA temp_store=MEMORY")

            # Load sqlite-vec extension
            try:
                conn.enable_load_extension(True)
                import sqlite_vec
                sqlite_vec.load(conn)
            except Exception as e:
                logger.warning(f"[DB] sqlite-vec not available: {e}")

            _local.conn = conn
            _local.db_path = self.db_path
            logger.debug(f"[DB] New connection for thread {threading.current_thread().name}")

        return conn

    def get_connection(self):
        """Get a connection (thread-local). Compatible with old API."""
        return self._get_connection()

    def release_connection(self, conn):
        """No-op compatibility shim — SQLite connections are thread-local and reused.

        Args:
            conn: Ignored.  Present for API compatibility with the old PostgreSQL
                connection-pool pattern.
        """
        pass

    @contextmanager
    def connection(self):
        """Yield a thread-local connection, committing or rolling back on exit.

        Intended for direct cursor operations.  Use :meth:`get_session` when
        the calling code expects an SQLAlchemy-style ``session.execute()`` API.

        Yields:
            The thread-local ``sqlite3.Connection`` (see :meth:`_get_connection`).

        Raises:
            Exception: Re-raises any exception thrown inside the ``with`` block
                after rolling back the current transaction.
        """
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def execute(self, sql, params=None):
        """Execute a write statement (INSERT/UPDATE/DELETE) with auto-commit.

        Args:
            sql: SQL statement string to execute.
            params: Optional sequence or mapping of bind parameters.
        """
        with self.connection() as conn:
            cursor = conn.cursor()
            try:
                if params is None:
                    cursor.execute(sql)
                else:
                    cursor.execute(sql, params)
            finally:
                cursor.close()

    def fetch_all(self, sql, params=None):
        """Execute a SELECT statement and return all rows as a list of dicts.

        Args:
            sql: SQL SELECT string to execute.
            params: Optional sequence or mapping of bind parameters.

        Returns:
            List of column→value dicts, one per result row.
        """
        with self.connection() as conn:
            cursor = DictCursor(conn.cursor())
            try:
                cursor.execute(sql, params)
                return cursor.fetchall()
            finally:
                cursor.close()

    @contextmanager
    def get_session(self):
        """Yield a :class:`SessionProxy` compatible with SQLAlchemy session usage.

        Provides a drop-in shim for callers that previously used
        ``with db.get_session() as session: session.execute(text(...), params)``.
        The underlying connection is committed on clean exit and rolled back if
        an exception propagates out of the ``with`` block.

        Yields:
            A :class:`SessionProxy` wrapping the current thread's
            ``sqlite3.Connection``.

        Raises:
            Exception: Re-raises any exception thrown inside the ``with`` block
                after rolling back the active transaction.
        """
        conn = self._get_connection()
        proxy = SessionProxy(conn)
        try:
            yield proxy
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def close_pool(self):
        """Close the calling thread's SQLite connection and clear thread-local state.

        Silently ignores errors from ``sqlite3.Connection.close()`` so that
        worker teardown paths remain exception-safe.  After this call
        ``_local.conn`` and ``_local.db_path`` are both reset to ``None``,
        ensuring the next call to :meth:`_get_connection` opens a new
        connection.
        """
        conn = getattr(_local, 'conn', None)
        if conn:
            try:
                conn.close()
            except Exception:
                pass
            _local.conn = None
            _local.db_path = None

    def run_pending_migrations(self):
        """Apply unapplied SQL migration files and add any missing schema columns.

        Discovers ``*.sql`` files under ``backend/migrations/``, sorted
        lexicographically (so filenames should be prefixed with a zero-padded
        sequence number, e.g. ``001_init.sql``).  Already-applied filenames are
        recorded in the ``schema_migrations`` table; only new files are executed.

        After running file-based migrations the method also performs a set of
        idempotent ``ALTER TABLE … ADD COLUMN`` statements guarded by
        ``PRAGMA table_info`` checks, providing a forward-compatible way to
        introduce columns that ``sqlite`` cannot add via ``IF NOT EXISTS``.

        Raises:
            sqlite3.OperationalError: If a migration file contains invalid SQL
                or references a table/column that does not exist.
            OSError: If a migration file cannot be read from disk.
        """
        migrations_dir = Path(__file__).resolve().parent.parent / "migrations"

        if not migrations_dir.exists():
            logger.info("No migrations directory found, skipping migrations")
            return

        with self.connection() as conn:
            cursor = conn.cursor()

            # Create migrations tracking table if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT UNIQUE NOT NULL,
                    applied_at TEXT DEFAULT (datetime('now'))
                )
            """)

            # Get list of applied migrations
            cursor.execute("SELECT filename FROM schema_migrations")
            applied = {row[0] for row in cursor.fetchall()}

            # Find all .sql migration files
            migration_files = sorted(migrations_dir.glob("*.sql"))

            pending_count = 0
            for migration_file in migration_files:
                filename = migration_file.name

                if filename in applied:
                    continue

                logger.info(f"Applying migration: {filename}")
                with open(migration_file, 'r') as f:
                    sql = f.read()

                cursor.executescript(sql)

                cursor.execute(
                    "INSERT INTO schema_migrations (filename) VALUES (?)",
                    (filename,)
                )
                pending_count += 1
                logger.info(f"Migration applied: {filename}")

            # Idempotent column additions (SQLite lacks ADD COLUMN IF NOT EXISTS).
            # Each tuple: (table, column, column_def, optional_index_sql).
            _optional_columns = [
                ("documents", "watched_folder_id", "TEXT REFERENCES watched_folders(id)",
                 "CREATE INDEX IF NOT EXISTS idx_documents_watched_folder ON documents(watched_folder_id) WHERE watched_folder_id IS NOT NULL"),
                ("documents", "doc_category", "TEXT", None),
                ("documents", "doc_project", "TEXT", None),
                ("documents", "doc_date", "TEXT", None),
                ("documents", "meta_locked", "INTEGER DEFAULT 0", None),
                ("tool_capability_profiles", "effort", "TEXT DEFAULT 'moderate'", None),
                ("tool_capability_profiles", "skill_category", "TEXT", None),
                # Migration 005 — reasoning annotation from triage effort tagging
                ("routing_decisions", "reasoning", "TEXT", None),
                # Uncertainty Engine Phase 1 — reliability columns on durable memory stores
                ("user_traits",       "reliability", "TEXT DEFAULT 'reliable'", None),
                ("episodes",          "reliability", "TEXT DEFAULT 'reliable'", None),
                ("semantic_concepts", "reliability", "TEXT DEFAULT 'reliable'", None),
            ]
            for table, col, col_def, *extra in _optional_columns:
                cursor.execute(f"PRAGMA table_info({table})")
                existing_cols = {row[1] for row in cursor.fetchall()}
                if col not in existing_cols:
                    cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_def}")
                    logger.info(f"Added column {table}.{col}")
                # Always try to create the index (idempotent)
                if extra and extra[0]:
                    cursor.execute(extra[0])

            if pending_count == 0:
                logger.info("No pending migrations")
            else:
                logger.info(f"Applied {pending_count} migrations")

            cursor.close()
