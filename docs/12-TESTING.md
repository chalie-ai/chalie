# Chalie Testing Documentation - Test Conventions, Fixtures & Mock Strategies

This comprehensive guide covers testing guide, test conventions, mock strategies, providing essential information for developers and users. For related topics, see: [Documentation Index](../docs/INDEX.md) | [[Data Schemas](08-DATA-SCHEMAS.md) Reference](../docs/08-DATA-SCHEMAS.md) | [[Tools](09-TOOLS.md) System Documentation](../docs/09-[TOOLS](09-TOOLS.md).md)


This comprehensive guide covers Chalie documentation, technical guide, providing essential information for developers and users. For related topics, see: 


## [Quick Start](01-QUICK-START.md)

```bash
cd backend

# Run all unit tests (fast, no external deps)
pytest -m unit

# Run a specific test file
pytest tests/test_mode_router.py

# Run a single test
pytest tests/test_mode_router.py::TestModeRouter::test_respond_wins_on_warm_question

# Verbose output
pytest -m unit -v
```

## Conventions

### Naming

```
test_{behavior}_when_{condition}
```

Omit `_when_` for obvious cases:

```python
def test_empty_query_returns_empty(self):       # obvious
def test_site_goes_down(self):                   # obvious
def test_accept_task_enforces_max_active_limit(self):  # with condition
```

### Structure (AAA)

Every test follows **Arrange / Act / Assert**:

```python
def test_get_returns_value_when_found(self, mock_db):
    # Arrange
    db, _, result = mock_db
    result.fetchone.return_value = ('my-value',)
    service = SettingsService(db)

    # Act
    value = service.get('some_key')

    # Assert
    assert value == 'my-value'
```

### Markers

```python
@pytest.mark.unit          # No external deps ([MemoryStore](08-DATA-SCHEMAS.md), SQLite, LLM)
@pytest.mark.integration   # Requires running [services](04-ARCHITECTURE.md)
```

### Banned Patterns

- `assert True` — always passes, tests nothing
- `assert isinstance(x, dict)` as sole assertion — verify content, not type
- `assert x is not None` without content check — verify the actual value
- Tests with zero assertions — every test must assert specific behavior

## Fixture Catalog

All fixtures live in `tests/conftest.py`.

### `mock_memory_store`
In-memory [MemoryStore](08-DATA-SCHEMAS.md) instance. Patches `[MemoryStore](08-DATA-SCHEMAS.md)` connections. Flushes on teardown.

### `mock_config`
Patches `ConfigService.get_agent_config`, `get_agent_prompt`, and `connections`. Provides realistic agent configs for memory-chunker, mode-router, fact-store, frontal-cortex.

### `mock_ollama`
Returns `LLMResponse(text='{"gists": [], "scope": "test"}', model='test-model', provider='[ollama](02-PROVIDERS-SETUP.md)')`. Also mocks `generate_embedding` → `[0.0] * 256`.

### `mock_db`
Basic DB mock. Context manager yields cursor directly. Good for simple [services](04-ARCHITECTURE.md).

### `mock_db_rows`
Extended DB mock with programmable cursor returns. Supports both patterns:
- `db.connection()` → conn → `conn.cursor()` → cursor
- `db.get_session()` → session → `session.execute()` → result

Returns `(db, cursor)` tuple.

### `mock_llm`
Configurable LLM mock. Set `mock_llm.response_text` before calling:
```python
def test_something(self, mock_llm):
    mock_llm.response_text = '{"verdict": "good"}'
    # [Services](04-ARCHITECTURE.md) calling create_llm_service().send_message() get that text
```

### `authed_client`
Full Flask app with all blueprints, auth bypassed, DB/[MemoryStore](08-DATA-SCHEMAS.md) mocked:
```python
def test_endpoint(self, authed_client):
    client, mock_db, mock_store = authed_client
    response = client.get('/system/health')
    assert response.status_code == 200
```

## Test Data Factories

Located in `tests/helpers.py`. Return tuples matching actual DB column orders:

```python
from tests.helpers import make_task_row, make_scheduled_item, make_trait_row

# 18-element tuple matching persistent_tasks SELECT
row = make_task_row(status='accepted', goal='Monitor weather')

# 11-element tuple matching scheduled_items SELECT
row = make_scheduled_item(message='Take medicine', recurrence='daily')

# 4-element tuple matching user_traits SELECT
row = make_trait_row(trait_key='timezone', trait_value='CET')

# Dict matching episodic retrieval service output
episode = make_episode_row(gist='Discussed morning routine')

# 9-element tuple matching providers SELECT
provider = make_provider_row(platform='[ollama](02-PROVIDERS-SETUP.md)', model='qwen3:4b')
```

## Mock Strategies

### LLM Mocking

Never call real LLMs in unit tests. Two approaches:

**Via `mock_llm` fixture** (patches `create_llm_service`):
```python
def test_critic(self, mock_llm):
    mock_llm.response_text = '{"safe": true, "verdict": "ok"}'
    result = critic.evaluate(action_result)
```

**Via `mock_ollama` fixture** (patches `OllamaService`):
```python
def test_chunker(self, mock_ollama, mock_config):
    mock_ollama.send_message.return_value = LLMResponse(
        text='{"gists": [{"content": "test"}]}',
        model='test', provider='[ollama](02-PROVIDERS-SETUP.md)'
    )
```

### DB Mocking — Connection Pattern

For [services](04-ARCHITECTURE.md) using `db.connection()` → cursor (most [services](04-ARCHITECTURE.md)):

```python
@pytest.fixture
def mock_db(self):
    db = MagicMock()
    cursor = MagicMock()
    conn = MagicMock()
    conn.cursor.return_value = cursor
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=conn)
    ctx.__exit__ = MagicMock(return_value=False)
    db.connection.return_value = ctx
    return db, cursor
```

### DB Mocking — Session Pattern

For [services](04-ARCHITECTURE.md) using `db.get_session()` → session (SettingsService, ProviderDbService, auth):

```python
@pytest.fixture
def mock_db(self):
    db = MagicMock()
    session = MagicMock()
    result = MagicMock()
    session.execute.return_value = result
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=session)
    ctx.__exit__ = MagicMock(return_value=False)
    db.get_session.return_value = ctx
    return db, session, result
```

### Auth Bypass for API Tests

The `@require_session` decorator checks `validate_session(request)` at runtime:

```python
@pytest.fixture(autouse=True)
def bypass_auth(self):
    with patch('[services](04-ARCHITECTURE.md).auth_session_service.validate_session', return_value=True):
        yield
```

### Optional Dependencies

Use `pytest.importorskip` for [tools](09-TOOLS.md) with optional deps:

```python
feedparser = pytest.importorskip('feedparser', reason='feedparser not installed')
from [tools](09-TOOLS.md).reddit_digest.handler import execute
```

### Module Injection for Missing Packages

When a dependency is lazily imported inside a function and not installed:

```python
MockWebPushException = type('WebPushException', (Exception,), {})
mock_pywebpush = MagicMock()
mock_pywebpush.webpush = MagicMock()
mock_pywebpush.WebPushException = MockWebPushException

with patch.dict('sys.modules', {'pywebpush': mock_pywebpush}):
    send_push_to_all("Test", "Body")
```

## Adding Tests for a New Service

1. Create `tests/test_my_service.py`
2. Import the service: `from [services](04-ARCHITECTURE.md).my_service import MyService`
3. Add `@pytest.mark.unit` to the test class
4. Mock external deps (DB, [MemoryStore](08-DATA-SCHEMAS.md), LLM) in fixtures
5. Write tests following AAA structure
6. Run: `pytest tests/test_my_service.py -v`
7. Verify: `pytest -m unit` still passes

## Adding Tests for a New API Blueprint

1. Create `tests/test_api_my_endpoint.py`
2. Register only the blueprint you're testing:
   ```python
   @pytest.fixture
   def client(self):
       app = Flask(__name__)
       app.register_blueprint(my_bp)
       app.config['TESTING'] = True
       return app.test_client()
   ```
3. Add `bypass_auth` autouse fixture
4. Patch service constructors at their lazy-import points
5. Test HTTP status codes, JSON response shape, and key field values
6. Test validation (400), not-found (404), and error (500) paths

## Adding Tests for a New Tool Handler

1. Create `tests/test_tool_my_tool.py`
2. Import: `from [tools](09-TOOLS.md).my_tool.handler import execute`
3. Mock external HTTP calls via `patch('requests.get')`
4. Test state round-trip: `result['_state']` → feed back as `params['_state']`
5. Test error cases: missing config, API failures, invalid responses

## Related Documentation
- [Vision & Philosophy](00-VISION.md)
- [Quick Start Guide](01-QUICK-START.md)
- [LLM Providers Setup](02-PROVIDERS-SETUP.md)
- [Web Interface](03-WEB-INTERFACE.md)
- [System Architecture](04-ARCHITECTURE.md)
- [Workflow Guide](05-WORKFLOW.md)
- [Workers Overview](06-WORKERS.md)
- [Cognitive Architecture](07-COGNITIVE-ARCHITECTURE.md)
- [Data Schemas](08-DATA-SCHEMAS.md)
- [Tools & Extensions](09-TOOLS.md)
- [Context Relevance](10-CONTEXT-RELEVANCE.md)
- [Message Flow Diagrams](13-MESSAGE-FLOW.md)
- [Default Tools](14-DEFAULT-TOOLS.md)