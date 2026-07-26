"""Thread-local SQLite connection gateway — the WAL-mode terminal that owns
the sole ``sqlite3`` import on the MP spine (Rule-3 depth: terminal, no
``mp``).

Opens/owns each thread's connection(s), binds a connection getter onto
:class:`models.model.Model` once at boot, and owns generic
transaction/commit mechanics. It issues no domain SQL of its own — every
domain query lives in :class:`models.model.Model` / :class:`models.query.Query`,
which run on the connection this class hands them.

Two access shapes, one registry:

* **On-spine** code holds a :class:`Database` instance (``self.mp.db``), binds
  it at boot, and reaches the DB through models + ``self.mp.db.transaction()``.
* **Off-spine** code (mp-less: tool abilities, decay/GC, doc/list CRUD, …)
  reaches the DB through the static accessors :meth:`conn` / :meth:`transaction`,
  passing a custom path for a dedicated db-file (``skills.sqlite``,
  ``mcp_tools.sqlite``). It NEVER runs on-spine.

Connections are per-thread and per-path: one slot per ``(thread, db_path)``.
They are never closed in the hot path — a connection lives as long as its
thread and dies with it (CPython releases the thread-local, ``__del__`` closes
the handle). :meth:`close` exists only for deterministic teardown (tests,
graceful shutdown). Idle connections hold no lock: the handle is autocommit,
and every ``transaction()`` always commits or rolls back, so nothing is left
open to pin WAL checkpointing or block another writer.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from collections.abc import Generator
from contextlib import contextmanager

from models.model import Model
from services.file_mapper_service import FileMapperService

logger = logging.getLogger(__name__)

# Per-thread registry, shared across every Database instance: ``conns`` maps
# db_path -> live connection, ``depths`` maps db_path -> transaction nesting
# depth. Keying by path lets the default chalie.db connection and a dedicated
# ability db-file coexist on one thread without thrashing a single slot.
_local = threading.local()

#: Sugar the off-spine accessors accept in place of the real chalie.db path.
DEFAULT = "default"


class Database:
    """Owns the sqlite3 connections; the only sqlite3 importer on the spine."""

    def __init__(self) -> None:
        self.db_path: str = str(FileMapperService.get_db_path())

    # ── static per-(thread, path) connection registry ────────────────────────

    @staticmethod
    def conn(path: str = DEFAULT) -> sqlite3.Connection:
        """Return this thread's connection for ``path``, opening it on first
        use and caching it for the thread's life. ``path=DEFAULT`` resolves to
        chalie.db, so an off-spine ``Database.conn()`` and the spine's bound
        connection land on the SAME handle; pass an explicit path for a
        dedicated db-file. The off-spine accessor — on-spine code uses
        ``self.mp.db`` + models instead."""
        resolved = str(FileMapperService.get_db_path()) if path == DEFAULT else path
        conns = getattr(_local, 'conns', None)
        if conns is None:
            conns = {}
            _local.conns = conns
        conn = conns.get(resolved)
        if conn is None:
            conn = Database._open(resolved)
            conns[resolved] = conn
        return conn

    @staticmethod
    def _open(path: str) -> sqlite3.Connection:
        """Open one connection with the WAL/foreign-key/busy-timeout/NORMAL/
        in-memory-temp pragmas and the ``sqlite-vec`` extension."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        conn = sqlite3.connect(path, timeout=30)
        conn.isolation_level = None  # autocommit; transaction() issues an explicit BEGIN to group multi-write blocks
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=15000")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=MEMORY")

        try:
            conn.enable_load_extension(True)
            import sqlite_vec
            sqlite_vec.load(conn)
        except Exception as exc:
            logger.warning(f"[Database] sqlite-vec not available: {exc}")

        logger.debug(f"[Database] opened {path} for thread {threading.current_thread().name}")
        return conn

    def get_connection(self) -> sqlite3.Connection:
        """This instance's (default-path) connection — the getter bound onto
        models. Delegates to the shared registry so the spine and off-spine
        share one chalie.db handle per thread."""
        return Database.conn(self.db_path)

    def bind(self) -> None:
        """Boot step: bind this instance's connection getter onto the Model
        base (``Model.bind``) so every model subclass re-derives its own
        thread's connection on every access, instead of one thread's
        connection being frozen onto the class forever."""
        Model.bind(self.get_connection)

    @staticmethod
    @contextmanager
    def transaction(path: str = DEFAULT) -> Generator[sqlite3.Connection, None, None]:
        """Open an explicit, **reentrant** transaction on ``path``'s (autocommit)
        connection: the outermost ``with`` issues ``BEGIN IMMEDIATE`` and owns
        the commit/rollback; a nested ``with`` on the same thread+path just
        joins it (SQLite has no true nested transactions). Depth is tracked
        per-path, so a transaction on chalie.db and one on a dedicated db-file
        never share a counter. Since the connection autocommits by default,
        single model writes persist on their own; wrapping them here is what
        makes a multi-write block atomic — and the reentrancy lets one
        coordinating service's transaction call another's without a ``cannot
        start a transaction within a transaction`` crash (I6: ``Database`` owns
        transaction/commit).

        ``IMMEDIATE`` (not deferred ``BEGIN``) takes SQLite's write lock at
        statement one, so a read-then-write block — ``begin()``'s per-channel
        turn-id allocation (``SELECT MAX(turn_id)+1`` then ``INSERT``, §6.8) —
        is the single-writer critical section the spine relies on: two
        concurrent fresh turns on one channel serialize here and can never read
        the same max and collide. This is the atomicity, not a Python lock.

        On-spine callers reach this via ``self.mp.db.transaction()`` (default
        path); off-spine callers pass an explicit path for a dedicated db-file."""
        resolved = str(FileMapperService.get_db_path()) if path == DEFAULT else path
        conn = Database.conn(resolved)
        depths = getattr(_local, 'depths', None)
        if depths is None:
            depths = {}
            _local.depths = depths
        entry_depth: int = depths.get(resolved, 0)
        if entry_depth == 0:
            conn.execute("BEGIN IMMEDIATE")
        depths[resolved] = entry_depth + 1
        try:
            yield conn
        except Exception:
            if entry_depth == 0:
                conn.rollback()
            raise
        else:
            if entry_depth == 0:
                conn.commit()
        finally:
            depths[resolved] = entry_depth

    @staticmethod
    def close() -> None:
        """Close every connection this thread has opened and clear its
        registry — deterministic teardown only (tests, graceful shutdown),
        never the request hot path."""
        conns = getattr(_local, 'conns', None)
        if conns:
            for conn in conns.values():
                try:
                    conn.close()
                except sqlite3.Error:
                    pass
        _local.conns = {}
        _local.depths = {}
