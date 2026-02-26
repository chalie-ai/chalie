"""
Shared test fixtures — full sandbox isolation.

No real Redis, PostgreSQL, or Ollama connections are made.
"""

import sys
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, MagicMock
from io import BytesIO

import pytest
import fakeredis


@pytest.fixture
def mock_redis():
    """Isolated in-memory Redis — no real Redis touched."""
    r = fakeredis.FakeRedis(decode_responses=True)
    with patch('services.redis_client.RedisClientService.create_connection', return_value=r):
        yield r
    r.flushall()


@pytest.fixture
def mock_config():
    """Test config — no file I/O."""
    agent_configs = {
        'memory-chunker': {
            'model': 'test-model',
            'attention_span_minutes': 30,
            'min_gist_confidence': 7,
            'max_gists': 8,
            'gist_similarity_threshold': 0.7,
            'max_gists_per_type': 2,
            'timeout': 120,
        },
        'mode-router': {
            'base_scores': {
                'RESPOND': 0.40,
                'CLARIFY': 0.30,
                'ACT': 0.20,
                'ACKNOWLEDGE': 0.10,
                'IGNORE': -0.50,
            },
            'weights': {
                'respond.warmth_boost': 0.20,
                'respond.fact_density': 0.15,
                'respond.gist_density': 0.10,
                'respond.question_warm': 0.15,
                'respond.cold_penalty': 0.15,
                'respond.greeting_penalty': 0.20,
                'respond.feedback_penalty': 0.15,
                'clarify.cold_boost': 0.25,
                'clarify.question_no_facts': 0.20,
                'clarify.new_topic_question': 0.10,
                'clarify.cold_question': 0.05,
                'clarify.warm_penalty': 0.20,
                'act.question_moderate_context': 0.20,
                'act.interrogative_gap': 0.15,
                'act.implicit_reference': 0.15,
                'act.very_cold_penalty': 0.10,
                'act.warm_facts_penalty': 0.10,
                'acknowledge.greeting': 0.80,
                'acknowledge.positive_feedback': 0.55,
                'acknowledge.question_penalty': 0.30,
                'ignore.empty_input': 1.00,
            },
            'tiebreaker_base_margin': 0.20,
            'tiebreaker_min_margin': 0.08,
        },
        'mode-tiebreaker': {
            'model': 'test-model',
            'temperature': 0.1,
            'max_tokens': 32,
        },
        'fact-store': {
            'model': 'test-model',
            'ttl_minutes': 1440,
            'max_facts_per_topic': 50,
            'min_confidence': 0.5,
        },
        'frontal-cortex': {
            'model': 'test-model',
            'cost_base': 1.0,
            'cost_growth_factor': 1.5,
        },
    }
    agent_prompts = {
        'memory-chunker': 'Test chunker prompt {{world_state}}',
        'fact-extraction': 'Test fact extraction prompt {{user_message}} {{system_response}}',
        'mode-tiebreaker': 'Test tiebreaker prompt',
    }
    connections = {
        'redis': {'host': 'localhost', 'port': 6379},
        'postgres': {'host': 'localhost', 'port': 5432, 'database': 'test'},
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
def mock_db():
    """Mock DatabaseService — no real PostgreSQL touched."""
    mock = MagicMock()
    ctx = MagicMock()
    cursor = MagicMock()
    cursor.fetchone.return_value = None
    cursor.fetchall.return_value = []
    ctx.__enter__ = MagicMock(return_value=cursor)
    ctx.__exit__ = MagicMock(return_value=False)
    mock.connection.return_value = ctx
    yield mock


@pytest.fixture
def mock_db_rows():
    """Extended DB mock with programmable cursor returns.

    Supports both patterns used across services:
      1. with db.connection() as conn: cursor = conn.cursor()
      2. with db.get_session() as session: session.execute(...)

    Usage:
        def test_something(self, mock_db_rows):
            db, cursor = mock_db_rows
            cursor.fetchone.return_value = make_task_row(status='active')
            cursor.fetchall.return_value = [make_task_row(), make_task_row(task_id=2)]
    """
    db = MagicMock()
    cursor = MagicMock()
    cursor.fetchone.return_value = None
    cursor.fetchall.return_value = []
    cursor.rowcount = 0

    # Pattern 1: db.connection() → conn → conn.cursor() → cursor
    conn = MagicMock()
    conn.cursor.return_value = cursor
    conn.execute.return_value = cursor  # some services do conn.execute() directly
    conn_ctx = MagicMock()
    conn_ctx.__enter__ = MagicMock(return_value=conn)
    conn_ctx.__exit__ = MagicMock(return_value=False)
    db.connection.return_value = conn_ctx

    # Pattern 2: db.get_session() → session → session.execute() → result
    session = MagicMock()
    session_result = MagicMock()
    session_result.fetchone.return_value = None
    session_result.fetchall.return_value = []
    session.execute.return_value = session_result
    session_ctx = MagicMock()
    session_ctx.__enter__ = MagicMock(return_value=session)
    session_ctx.__exit__ = MagicMock(return_value=False)
    db.get_session.return_value = session_ctx

    # Expose session_result for tests that need SQLAlchemy-style control
    db._test_session = session
    db._test_session_result = session_result

    yield (db, cursor)


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
def authed_client():
    """Flask test client with real blueprints registered, auth bypassed.

    Usage:
        def test_endpoint(self, authed_client):
            client, mock_db, mock_redis = authed_client
            mock_db._test_session_result.fetchall.return_value = [...]
            response = client.get('/system/health')
    """
    from api import create_app

    mock_db = MagicMock()
    mock_r = MagicMock()

    # Wire up db.connection() context manager
    cursor = MagicMock()
    cursor.fetchone.return_value = None
    cursor.fetchall.return_value = []
    cursor.rowcount = 0
    conn = MagicMock()
    conn.cursor.return_value = cursor
    conn.execute.return_value = cursor
    conn_ctx = MagicMock()
    conn_ctx.__enter__ = MagicMock(return_value=conn)
    conn_ctx.__exit__ = MagicMock(return_value=False)
    mock_db.connection.return_value = conn_ctx

    # Wire up db.get_session() context manager
    session = MagicMock()
    session_result = MagicMock()
    session_result.fetchone.return_value = None
    session_result.fetchall.return_value = []
    session.execute.return_value = session_result
    session_ctx = MagicMock()
    session_ctx.__enter__ = MagicMock(return_value=session)
    session_ctx.__exit__ = MagicMock(return_value=False)
    mock_db.get_session.return_value = session_ctx

    # Expose internals for test control
    mock_db._test_cursor = cursor
    mock_db._test_conn = conn
    mock_db._test_session = session
    mock_db._test_session_result = session_result

    with patch('services.auth_session_service.validate_session', return_value=True), \
         patch('services.database_service.get_shared_db_service', return_value=mock_db), \
         patch('services.redis_client.RedisClientService.create_connection', return_value=mock_r):
        app = create_app()
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield (client, mock_db, mock_r)


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
def flask_test_client(mock_redis):
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
