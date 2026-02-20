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
