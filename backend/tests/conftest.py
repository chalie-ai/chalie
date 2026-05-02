# Baseline: 2249 passed, 65 failed, 499 errors (2026-03-27)
# Errors are pre-existing: 15 files excluded (numpy import failure in this env),
# and 499 test-setup errors caused by missing sqlite-vec extension (vec0 module).
"""
Shared test fixtures — full sandbox isolation.

No real external connections. MemoryStore IS the production implementation.
"""

import shutil
from unittest.mock import MagicMock, patch

import pytest


# ── Pre-import modules that bind get_shared_db_service at module level ────────
# `services.dmn_service` does `from services.database_service import
# get_shared_db_service` at import time. If the very first import of
# `services.dmn_service` happens INSIDE a `patch('services.database_service
# .get_shared_db_service', ...)` block (e.g. tests that mock the DB before
# referencing dmn_service), the local reference inside dmn_service binds
# permanently to the MagicMock and never recovers — polluting every later
# test that touches DMNService.
#
# Pre-importing here forces the binding to happen against the real function
# before any test patches can interfere.
def _preload_db_bound_modules():
    try:
        import services.dmn_service  # noqa: F401
    except Exception:
        pass  # If something fails to import in this env, individual tests will surface it.


_preload_db_bound_modules()


# ── Real-SQLite fixtures ──────────────────────────────────────────
# Session-scoped template: full schema + migrations applied once.
# Function-scoped `db`: fresh copy per test, patched as the singleton.

@pytest.fixture(scope='session')
def _db_template(tmp_path_factory):
    """Build a fully-converged SQLite database file (once per session).

    Runs the real production boot sequence — SchemaConvergenceService.converge()
    — against a temp file.  The result is a "golden" database that
    function-scoped fixtures copy cheaply.
    """
    from services.database_service import DatabaseService
    from services.schema_convergence_service import SchemaConvergenceService

    template_dir = tmp_path_factory.mktemp('db_template')
    template_path = str(template_dir / 'template.db')

    db = DatabaseService(template_path)
    convergence = SchemaConvergenceService(db, embedding_dimensions=256)
    convergence.converge()

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

    conn = db_service._get_connection()
    try:
        yield conn
    finally:
        db_service.close_pool()
        _db_mod._shared_db_service = original
        _db_mod._local.conn = None
        _db_mod._local.db_path = None
        _dgs_mod._instance = original_dgs_instance


# ── Non-DB mock fixtures ──────────────────────────────────────────

@pytest.fixture
def store():
    """Isolated MemoryStore — same implementation used in production.

    Patches both the canonical ``get_shared_store()`` in memory_store and the
    legacy ``MemoryClientService.create_connection()`` shim so every code path
    sees the same isolated instance.

    Prefer this fixture over ``mock_store`` for new tests.  The ``mock_store``
    name is kept as a backward-compat alias so existing tests continue to work
    without a big-bang rename.

    Yields:
        MemoryStore: A fresh, fully-functional in-process store instance.
    """
    from services.memory_store import MemoryStore
    _store = MemoryStore()
    with patch('services.memory_store.get_shared_store', return_value=_store), \
         patch('services.memory_client.MemoryClientService.create_connection', return_value=_store):
        yield _store


@pytest.fixture
def mock_store(store):
    """Backward-compat alias for the ``store`` fixture.

    .. deprecated::
        Use ``store`` directly in new tests.
        # TODO: remove after all references migrated

    Yields:
        MemoryStore: Delegates to the ``store`` fixture unchanged.
    """
    yield store


@pytest.fixture
def mock_config():
    """Test config — no file I/O."""
    agent_configs = {
        'frontal-cortex': {
            'model': 'test-model',
            'cost_base': 1.0,
            'cost_growth_factor': 1.5,
        },
    }
    agent_prompts = {}
    connections = {
        'memory': {},
        'rest_api': {'host': '0.0.0.0', 'port': 8081},
        'voice': {'enabled': False},
    }

    with patch('services.config_service.ConfigService.get_agent_config', side_effect=lambda name: agent_configs.get(name, {})), \
         patch('services.config_service.ConfigService.get_agent_prompt', side_effect=lambda name: agent_prompts.get(name, '')), \
         patch('services.config_service.ConfigService.connections', return_value=connections):
        yield agent_configs


@pytest.fixture
def mock_ollama():
    """Mock OllamaService — no real LLM calls."""
    from services.llm_service import LLMResponse
    mock = MagicMock()
    mock.send_message.return_value = LLMResponse(
        text='{"gists": [], "scope": "test"}',
        model='test-model',
        provider='ollama',
    )
    mock.generate_embedding.return_value = [0.0] * 256
    with patch('services.ollama_service.OllamaService', return_value=mock):
        yield mock




@pytest.fixture
def mock_llm():
    """Configurable LLM mock — set mock_llm.response_text before calling.

    Usage:
        def test_something(self, mock_llm):
            mock_llm.response_text = '{"verdict": "good"}'
            # Now any service calling create_llm_service().send_message() gets that text
    """
    from services.llm_service import LLMResponse
    mock = MagicMock()

    # Default response — override via mock.response_text
    mock.response_text = '{"result": "ok"}'

    def _send_message(*args, **kwargs):
        return LLMResponse(
            text=mock.response_text,
            model='test-model',
            provider='mock',
        )

    mock.send_message.side_effect = _send_message
    mock.generate_embedding.return_value = [0.0] * 256

    with patch('services.llm_service.create_llm_service', return_value=mock):
        yield mock


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
         patch('api._init_dashboard_gateway'), \
         patch('api._get_or_generate_session_secret', return_value='test-secret'):
        app = create_app()
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield (client, db, real_store)


@pytest.fixture
def mock_requests():
    """Mock requests.get/post/head for HTTP tool handlers."""
    with patch('requests.get') as mock_get, \
         patch('requests.post') as mock_post, \
         patch('requests.head') as mock_head:
        yield {'get': mock_get, 'post': mock_post, 'head': mock_head}


@pytest.fixture
def tmp_state_file(tmp_path):
    """Temporary state file path for tools using JSON state."""
    state_file = tmp_path / "state.json"
    return state_file


@pytest.fixture
def flask_test_client(mock_store):
    """Flask test client with mocked session for API tests."""
    from flask import Flask

    app = Flask(__name__)
    app.config['TESTING'] = True

    # Mock session in test client context
    @app.before_request
    def setup_session():
        from flask import g
        g.session = MagicMock()
        g.session.get.return_value = 'test_user_id'

    with app.test_client() as client:
        yield client
