"""SQLite connection management with thread-local connections.

Replaces the PostgreSQL/SQLAlchemy implementation with sqlite3 +
sqlite-vec + FTS5. Each thread gets its own connection via
threading.local(). WAL mode enables concurrent reads during writes.
"""

import logging
import os
import sqlite3
import threading
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional, Union

from services.file_mapper_service import FileMapperService

if TYPE_CHECKING:
    _Params = Union[Sequence[object], Mapping[str, object], None]

logger = logging.getLogger(__name__)


class _TextClause:
    """Wraps a SQL string so ``SessionProxy.execute(str(obj))`` works."""
    __slots__ = ('_sql',)

    def __init__(self, sql: str):
        self._sql = sql

    def __str__(self) -> str:
        return self._sql

    def __repr__(self) -> str:
        return f"text({self._sql!r})"


def text(sql: str) -> _TextClause:
    return _TextClause(sql)

# ── Thread-local storage for connections ────────────────────────
_local = threading.local()

# ── Singleton DatabaseService ───────────────────────────────────
_shared_db_service = None
_shared_lock = threading.Lock()


def get_shared_db_service() -> 'DatabaseService':
    """Thread-safe via double-checked lock."""
    global _shared_db_service
    if _shared_db_service is None:
        with _shared_lock:
            if _shared_db_service is None:
                _shared_db_service = DatabaseService(str(FileMapperService.get_db_path()))
                logger.info("[DB] Created shared DatabaseService singleton")
    return _shared_db_service


class SessionProxy:
    """Lightweight shim that mimics SQLAlchemy session.execute(text("SQL"), params).

    Usage:
        with db.get_session() as session:
            result = session.execute(text("SELECT * FROM t WHERE id = :id"), {"id": 1})
            rows = result.fetchall()
    """

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def execute(self, sql_or_text: object, params: "_Params" = None) -> "ResultProxy":
        """Accepts a raw SQL string or a :class:`_TextClause` produced by
        :func:`text`. Dict-style named parameters (``:name``) are
        transparently converted to SQLite positional ``?`` parameters."""
        # Extract the string from sqlalchemy text() objects
        sql = str(sql_or_text)

        if params and isinstance(params, dict):
            # Convert :name params to ? positional params
            import re
            ordered_params: list[object] = []
            def _replace(match: "re.Match[str]") -> str:
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

    def commit(self) -> None:
        """Explicit commit (also happens automatically on context-manager exit)."""
        self._conn.commit()

    def rollback(self) -> None:
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

    def fetchone(self) -> sqlite3.Row | None:
        row: sqlite3.Row | None = self._cursor.fetchone()
        return row  # sqlite3.Row already supports int and key indexing

    def fetchall(self) -> list[sqlite3.Row]:
        return self._cursor.fetchall()

    @property
    def rowcount(self) -> int:
        """Number of rows affected by the last statement."""
        return self._cursor.rowcount

    @property
    def lastrowid(self) -> int | None:
        """Row ID of the last inserted row."""
        return self._cursor.lastrowid

    def scalar(self) -> object:
        row = self.fetchone()
        if row is None:
            return None
        return row[0]

    def close(self) -> None:
        self._cursor.close()


class DictCursor:
    """Wraps sqlite3.Cursor to return list[dict] from fetchall()."""

    def __init__(self, cursor: sqlite3.Cursor):
        self._cursor = cursor

    def execute(self, sql: str, params: "_Params" = None) -> "DictCursor":
        if params is None:
            self._cursor.execute(sql)
        else:
            self._cursor.execute(sql, params)
        return self

    def executemany(self, sql: str, params_list: "Sequence[Sequence[object]]") -> "DictCursor":
        self._cursor.executemany(sql, params_list)
        return self

    def fetchone(self) -> dict[str, object] | None:
        row = self._cursor.fetchone()
        if row is None:
            return None
        return dict(row)

    def fetchall(self) -> list[dict[str, object]]:
        return [dict(row) for row in self._cursor.fetchall()]

    @property
    def lastrowid(self) -> int | None:
        """Row ID of the last inserted row."""
        return self._cursor.lastrowid

    @property
    def rowcount(self) -> int:
        """Number of rows affected by the last statement."""
        return self._cursor.rowcount

    @property
    def description(self) -> object:
        """Sequence of 7-item sequences describing each result column."""
        return self._cursor.description

    def close(self) -> None:
        """Close the cursor, releasing database resources."""
        self._cursor.close()

    def __iter__(self) -> Generator[dict[str, object], None, None]:
        """Iterate over result rows as dicts."""
        return (dict(row) for row in self._cursor)


class DatabaseService:
    """Manages SQLite connections with thread-local isolation and WAL mode."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        """Initialize the service and ensure the database directory exists.

        Args:
            db_path: Absolute path to the SQLite file. Defaults to the
                hard-coded ``FileMapperService.get_db_path()``. Tests pass a temp
                path explicitly.
        """
        self.db_path = db_path or str(FileMapperService.get_db_path())
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

            conn = sqlite3.connect(self.db_path, timeout=30)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=15000")
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

    def get_connection(self) -> "sqlite3.Connection":
        """Get a connection (thread-local). Compatible with old API."""
        return self._get_connection()

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
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

    def execute(self, sql: str, params: "_Params" = None) -> None:
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

    def fetch_all(self, sql: str, params: "_Params" = None) -> list[dict[str, object]]:
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
    def get_session(self) -> Generator["SessionProxy", None, None]:
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

    def close_pool(self) -> None:
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

