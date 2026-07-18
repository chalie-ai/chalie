# Baseline: 2249 passed, 65 failed, 499 errors (2026-03-27)
# Errors are pre-existing: 15 files excluded (numpy import failure in this env),
# and 499 test-setup errors caused by missing sqlite-vec extension (vec0 module).
import shutil
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
    from services.memory_store import MemoryStore


# ── Real-SQLite fixtures ──────────────────────────────────────────
# Session-scoped template: full schema + migrations applied once.
# Function-scoped `db`: fresh copy per test, patched as the singleton.

@pytest.fixture(scope='session')
def _db_template(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Build a fully-converged SQLite database file (once per session).

    Runs the real production boot sequence — SchemaConvergenceService.converge()
    plus the policy seed (PolicyManager.apply_seed, as run.py does at boot) —
    against a temp file.  The result is a "golden" database that
    function-scoped fixtures copy cheaply.
    """
    import services.database as _newdb
    from services.file_mapper_service import FileMapperService
    from services.policy_manager import PolicyManager
    from services.schema_convergence_service import SchemaConvergenceService

    template_dir = tmp_path_factory.mktemp('db_template')
    template_path = str(template_dir / 'template.db')

    # The convergence + seed services reach the DB through the Database gateway,
    # which resolves FileMapperService.get_db_path() at call time. Point that at
    # the template file for the build so the golden db lands there, not chalie.db.
    with patch.object(FileMapperService, 'get_db_path', return_value=Path(template_path)):
        _newdb.Database.close()  # drop any thread connection bound to another path
        convergence = SchemaConvergenceService()
        convergence.converge()
        # Mirror production boot (run.py / consumer.py): converge() applies only
        # static column DEFAULTs, never value backfills, so the deterministic
        # redesign-column backfill runs as a separate step right after it. Without
        # this the template diverges from a real boot — last_relevant_at / valid_from
        # / valid_to stay NULL where production would have populated them.
        convergence.backfill_redesign_columns()

        # Mirror boot: seed the flat policy table so gated tool calls on non-chat
        # channels (e.g. subconscious bash.* / timer) resolve to their real defaults
        # instead of an empty-table lazy 'ask'→deny. (PolicyManager.INTERNAL tools
        # bypass the gate entirely and carry no seed rows.)
        PolicyManager().apply_seed()

        # Flush WAL into main file so shutil.copy2 gets a self-contained copy
        _newdb.Database.conn().execute("PRAGMA wal_checkpoint(TRUNCATE)")
        _newdb.Database.close()
    return template_path


@pytest.fixture
def db(_db_template: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[sqlite3.Connection]:
    """Fresh, fully-migrated SQLite database — one per test.

    Copies the session-scoped template, points the ``Database`` gateway
    (``FileMapperService.get_db_path``) at it, and drops any stale thread
    connection so every ``Database.conn()`` call — on-spine and off-spine alike
    — lands on this test's file. Yields the raw ``sqlite3.Connection`` for
    seeding data.

    Usage::

        def test_something(self, db):
            db.execute("INSERT INTO lists (id, name) VALUES ('l1', 'Groceries')")
            db.commit()
            result = my_service.get_list('Groceries')
            assert result['name'] == 'Groceries'
    """
    import services.database as _newdb
    from services.file_mapper_service import FileMapperService

    test_db_path = str(tmp_path / 'test.db')
    shutil.copy2(_db_template, test_db_path)

    # Point the Database gateway at this test's file (the default path resolves
    # through FileMapperService) so every Database.conn() call lands on it, then
    # drop any stale thread connection bound to another path.
    monkeypatch.setattr(FileMapperService, 'get_db_path', lambda *_: Path(test_db_path))
    _newdb.Database.close()
    # Bind the connection getter onto Model — the boot step run.py runs once at
    # startup (``Database().bind()``). Repeated per test because each Database()
    # captures this test's patched path, so the getter must point at the current
    # test's file, not a prior test's or the real chalie.db.
    _newdb.Database().bind()

    # Invalidate heartbeat cache so it reads from this test's DB.
    from services.heartbeat_service import heartbeat_service
    heartbeat_service._ctx = None

    conn = _newdb.Database.conn()
    try:
        yield conn
    finally:
        _newdb.Database.close()
        heartbeat_service._ctx = None


# ── Vault backup isolation ────────────────────────────────────────
# Any test that initialises the real vault (directly or via /auth/register)
# writes a permanent vault_backup_*.json. Redirect the secure dir to a per-test
# temp path so backups never accumulate in the repo's data/secure/.

@pytest.fixture(autouse=True)
def _isolate_vault_backups(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from services.file_mapper_service import FileMapperService
    monkeypatch.setattr(FileMapperService, "_SECURE_DIR", tmp_path / "secure")


# ── Non-DB mock fixtures ──────────────────────────────────────────

@pytest.fixture
def store() -> "Iterator[MemoryStore]":
    """Isolated MemoryStore — same implementation used in production.

    Patches both the canonical ``get_shared_store()`` in memory_store and the
    legacy ``MemoryClientService.create_connection()`` shim so every code path
    sees the same isolated instance.

    Yields:
        MemoryStore: A fresh, fully-functional in-process store instance.
    """
    from services.memory_store import MemoryStore
    _store = MemoryStore()
    with patch('services.memory_store.get_shared_store', return_value=_store), \
         patch('services.memory_client.MemoryClientService.create_connection', return_value=_store):
        yield _store


@pytest.fixture
def authed_client(db: sqlite3.Connection) -> Iterator[tuple[object, sqlite3.Connection, object]]:
    """Flask test client with real blueprints registered, auth bypassed.

    Uses the real ``db`` fixture (which points the ``Database`` gateway at a
    per-test SQLite file), so Flask route handlers hit a real SQLite database.  The memory store is a
    real ``MemoryStore`` instance (not a ``MagicMock``), so route handlers that
    read or write store state work correctly in integration tests.

    Yields:
        tuple[FlaskClient, sqlite3.Connection, MemoryStore]: A 3-tuple of the
        Flask test client, the raw SQLite connection for seeding data, and the
        isolated in-process memory store.

    Usage::

        def test_endpoint(self, authed_client):
            client, db_conn, store = authed_client
            db_conn.execute("INSERT INTO ...")
            db_conn.commit()
            response = client.get('/health')
    """
    from api import create_app
    from services.memory_store import MemoryStore

    real_store = MemoryStore()

    with patch('services.auth_session_service.validate_session', return_value=True), \
         patch('services.memory_store.get_shared_store', return_value=real_store), \
         patch('services.memory_client.MemoryClientService.create_connection', return_value=real_store):
        app = create_app()
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield (client, db, real_store)


@pytest.fixture
def tmp_state_file(tmp_path: Path) -> Path:
    """Temporary state file path for tools using JSON state."""
    state_file = tmp_path / "state.json"
    return state_file
