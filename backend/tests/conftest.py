"""
Shared test fixtures — full sandbox isolation.

No real external connections. MemoryStore IS the production implementation.
"""

import shutil
import sys
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── Real-SQLite fixtures ──────────────────────────────────────────
# Session-scoped template: full schema + migrations applied once.
# Function-scoped `db`: fresh copy per test, patched as the singleton.

@pytest.fixture(scope='session')
def _db_template(tmp_path_factory):
    """Build a fully-migrated SQLite database file (once per session).

    Runs the real production boot sequence — SchemaService.initialize_schema()
    + DatabaseService.run_pending_migrations() — against a temp file.  The
    result is a "golden" database that function-scoped fixtures copy cheaply.
    """
    from services.database_service import DatabaseService
    from services.schema_service import SchemaService

    template_dir = tmp_path_factory.mktemp('db_template')
    template_path = str(template_dir / 'template.db')

    db = DatabaseService(template_path)
    schema = SchemaService(db, embedding_dimensions=256)
    schema.initialize_schema()
    db.run_pending_migrations()

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

    conn = db_service._get_connection()
    try:
        yield conn
    finally:
        db_service.close_pool()
        _db_mod._shared_db_service = original
        _db_mod._local.conn = None
        _db_mod._local.db_path = None


# ── Non-DB mock fixtures ──────────────────────────────────────────

@pytest.fixture
def mock_store():
    """Isolated MemoryStore — same implementation used in production.

    Patches both the canonical ``get_shared_store()`` in memory_store and the
    legacy ``MemoryClientService.create_connection()`` shim so every code path
    sees the same isolated instance.
    """
    from services.memory_store import MemoryStore
    store = MemoryStore()
    with patch('services.memory_store.get_shared_store', return_value=store), \
         patch('services.memory_client.MemoryClientService.create_connection', return_value=store):
        yield store


@pytest.fixture
def mock_config():
    """Test config — no file I/O."""
    agent_configs = {
        'mode-router': {
            'base_scores': {
                'UNIFIED': 0.40,
                'ACT': 0.20,
                'IGNORE': -0.50,
            },
            'weights': {
                'respond.warmth_boost': 0.20,
                'respond.fact_density': 0.15,
                'respond.gist_density': 0.10,
                'respond.question_warm': 0.15,
                'respond.cold_penalty': 0.15,
                'act.question_moderate_context': 0.20,
                'act.interrogative_gap': 0.15,
                'act.implicit_reference': 0.15,
                'act.very_cold_penalty': 0.10,
                'act.warm_facts_penalty': 0.10,
                'ignore.empty_input': 1.00,
            },
            'tiebreaker_base_margin': 0.20,
            'tiebreaker_min_margin': 0.08,
        },
        'frontal-cortex': {
            'model': 'test-model',
            'cost_base': 1.0,
            'cost_growth_factor': 1.5,
        },
    }
    agent_prompts = {
        'trait-extraction': 'Test trait extraction prompt {{message}}',
    }
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
    so Flask route handlers hit a real SQLite database.

    Usage::

        def test_endpoint(self, authed_client):
            client, db_conn, mock_store = authed_client
            db_conn.execute("INSERT INTO ...")
            db_conn.commit()
            response = client.get('/system/health')
    """
    from api import create_app

    mock_store = MagicMock()

    with patch('services.auth_session_service.validate_session', return_value=True), \
         patch('services.memory_store.get_shared_store', return_value=mock_store), \
         patch('services.memory_client.MemoryClientService.create_connection', return_value=mock_store), \
         patch('api._init_dashboard_gateway'):
        app = create_app()
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield (client, db, mock_store)


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
def tmp_sqlite_db(tmp_path):
    """Temporary SQLite database for scheduler tool/service tests."""
    db_path = tmp_path / "test.db"

    # Create tables if needed
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Create scheduled_items table for scheduler tests
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scheduled_items (
            id TEXT PRIMARY KEY,
            message TEXT NOT NULL,
            due_at TEXT NOT NULL,
            type TEXT DEFAULT 'reminder',
            recurrence TEXT,
            status TEXT DEFAULT 'pending',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()

    yield str(db_path)


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
