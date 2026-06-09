# Baseline: 2249 passed, 65 failed, 499 errors (2026-03-27)
# Errors are pre-existing: 15 files excluded (numpy import failure in this env),
# and 499 test-setup errors caused by missing sqlite-vec extension (vec0 module).
"""
Shared test fixtures — full sandbox isolation.

No real external connections. MemoryStore IS the production implementation.
"""

import shutil
from unittest.mock import patch

import pytest


# ── NOTE: get_shared_db_service import-time-binding hazard ────────────────────
# Modules that do `from services.database_service import get_shared_db_service`
# at module scope copy the function reference at import time. If such a module's
# FIRST import happens inside a `patch('services.database_service
# .get_shared_db_service', ...)` block, its local name binds permanently to the
# MagicMock and never recovers — polluting every later test. The `db` fixture
# below is immune: it rebinds the `_shared_db_service` singleton (not the
# function). But if a test `patch()`es the function directly, also patch the
# CONSUMING module's own reference — see test_policies_api.py (mcp_client_service).
# A blanket pre-load guard used to live here; it rotted (pointed at the deleted
# `services.dmn_service`) and was removed (TKT-645).


# ── Real-SQLite fixtures ──────────────────────────────────────────
# Session-scoped template: full schema + migrations applied once.
# Function-scoped `db`: fresh copy per test, patched as the singleton.

@pytest.fixture(scope='session')
def _db_template(tmp_path_factory):
    """Build a fully-converged SQLite database file (once per session).

    Runs the real production boot sequence — SchemaConvergenceService.converge()
    plus the policy seed (PolicyManager.apply_seed, as run.py does at boot) —
    against a temp file.  The result is a "golden" database that
    function-scoped fixtures copy cheaply.
    """
    from services.database_service import DatabaseService
    from services.policy_manager import PolicyManager
    from services.schema_convergence_service import SchemaConvergenceService

    template_dir = tmp_path_factory.mktemp('db_template')
    template_path = str(template_dir / 'template.db')

    db = DatabaseService(template_path)
    convergence = SchemaConvergenceService(db, embedding_dimensions=256)
    convergence.converge()

    # Mirror boot: seed the flat policy table so gated tool calls on non-chat
    # channels (e.g. subconscious email.* / timer) resolve to their real defaults
    # instead of an empty-table lazy 'ask'→deny. (PolicyManager.INTERNAL tools
    # bypass the gate entirely and carry no seed rows.)
    PolicyManager(db).apply_seed()

    # Flush WAL into main file so shutil.copy2 gets a self-contained copy
    with db.connection() as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    db.close_pool()
    return template_path


@pytest.fixture
def db(_db_template, tmp_path):
    """Fresh, fully-migrated SQLite database — one per test.

    Copies the session-scoped template, creates a real DatabaseService, and
    patches ``get_shared_db_service`` so every service in the call chain sees
    this database.  Yields the raw ``sqlite3.Connection`` for seeding data.

    Usage::

        def test_something(self, db):
            db.execute("INSERT INTO lists (id, name) VALUES ('l1', 'Groceries')")
            db.commit()
            result = my_service.get_list('Groceries')
            assert result['name'] == 'Groceries'
    """
    import services.database_service as _db_mod
    from services.database_service import DatabaseService

    test_db_path = str(tmp_path / 'test.db')
    shutil.copy2(_db_template, test_db_path)

    db_service = DatabaseService(test_db_path)

    # Clear thread-local cache so _get_connection() opens the new file
    _db_mod._local.conn = None
    _db_mod._local.db_path = None

    # Inject as the process-wide singleton
    original = _db_mod._shared_db_service
    _db_mod._shared_db_service = db_service

    # Reset data_graph singleton so it binds to this test's db on next access
    import services.data_graph_service as _dgs_mod
    original_dgs_instance = _dgs_mod._instance
    _dgs_mod._instance = None

    # Invalidate heartbeat cache so it reads from this test's DB
    from services.heartbeat_service import heartbeat_service
    heartbeat_service._ctx = None

    conn = db_service._get_connection()
    try:
        yield conn
    finally:
        db_service.close_pool()
        _db_mod._shared_db_service = original
        _db_mod._local.conn = None
        _db_mod._local.db_path = None
        _dgs_mod._instance = original_dgs_instance
        heartbeat_service._ctx = None


# ── Non-DB mock fixtures ──────────────────────────────────────────

@pytest.fixture
def store():
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
def authed_client(db):
    """Flask test client with real blueprints registered, auth bypassed.

    Uses the real ``db`` fixture (which patches ``get_shared_db_service``),
    so Flask route handlers hit a real SQLite database.  The memory store is a
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
            response = client.get('/system/health')
    """
    from api import create_app
    from services.memory_store import MemoryStore

    real_store = MemoryStore()

    with patch('services.auth_session_service.validate_session', return_value=True), \
         patch('services.memory_store.get_shared_store', return_value=real_store), \
         patch('services.memory_client.MemoryClientService.create_connection', return_value=real_store), \
         patch('api._get_or_generate_session_secret', return_value='test-secret'):
        app = create_app()
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield (client, db, real_store)


@pytest.fixture
def tmp_state_file(tmp_path):
    """Temporary state file path for tools using JSON state."""
    state_file = tmp_path / "state.json"
    return state_file
