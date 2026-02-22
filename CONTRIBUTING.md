# Contributing to Chalie

Thank you for your interest in contributing to Chalie! This document outlines guidelines for developers working on the project.

## Code Organization

### Frontend (Web Interface)

The frontend consists of three independent applications:

```
frontend/interface/
├── index.html          # Main HTML file
├── app.js              # Main application logic
├── api.js              # API client
├── renderer.js         # Card and message rendering
├── voice.js            # Voice interaction
├── presence.js         # Presence/status indicators
├── heartbeat.js        # Connection heartbeat
├── sse.js              # Server-sent events
├── sw.js               # Service worker
├── style.css           # All styling
├── manifest.json       # PWA manifest
├── icons/              # Icon files
└── cards/              # Reusable card components

frontend/brain/
└── Admin/cognitive dashboard

frontend/on-boarding/
└── Account setup wizard
```

**Important**: Do NOT create interface code elsewhere. Each frontend directory is self-contained.

### Backend

Code organization follows a service-oriented architecture:

```
backend/
├── services/           # Business logic and service classes
├── workers/            # Async worker processes
├── api/                # REST API blueprints
├── configs/            # Configuration files
├── migrations/         # Database migrations
├── prompts/            # LLM prompt templates
├── tests/              # Test suite
└── consumer.py         # Main supervisor process
```

## Development Workflow

### Setup

```bash
cd backend
pip install -r requirements.txt
source .venv/bin/activate  # or use poetry
cp .env.example .env
```

### Running Tests

```bash
# Run all tests
pytest

# Run only unit tests (fast, no external dependencies)
pytest -m unit

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_file.py
```

### Running Locally

```bash
# Terminal 1: Start consumer (all workers)
python consumer.py

# Terminal 2: Or run Flask directly for debugging
python -c "from api import create_app; app = create_app(); app.run(host='0.0.0.0', port=8080)"
```

### Docker Development

```bash
docker-compose build
docker-compose up -d
docker-compose logs -f backend
```

## Code Style

- **Python**: Follow PEP 8
- **JavaScript**: Follow standard ES6+ conventions
- **Documentation**: Keep docstrings clear and concise
- **Comments**: Explain the "why", not the "what"

## Adding New Features

### Adding a New REST API Endpoint

1. Create blueprint in `backend/api/my_endpoint.py`
2. Register in `backend/api/__init__.py`
3. Add auth via `@require_session` decorator
4. Test via curl or REST client

### Adding a New Service

1. Create `backend/services/my_service.py`
2. Implement class with clear public interface
3. Add unit tests in `backend/tests/test_my_service.py`
4. Register in `consumer.py` if background execution needed
5. Document in this file's "Architecture" section

### Adding a New Worker

1. Create `backend/workers/my_worker.py` extending `WorkerBase`
2. Implement process main loop
3. Register in `consumer.py`
4. Add integration tests

### Modifying the Web Interface

1. All changes go in `frontend/interface/` only
2. Keep styling in `css/styles.css`
3. Keep logic in `js/` files
4. Ensure mobile-first responsive design
5. Test on multiple device sizes

## Testing Requirements

- **New features**: Must include tests (unit + integration as applicable)
- **Bug fixes**: Include test that reproduces the bug
- **Refactoring**: Existing tests must pass
- **Run tests locally** before pushing: `pytest`

## Git Workflow

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make commits with clear messages
3. Push branch: `git push origin feature/your-feature`
4. Create Pull Request with description of changes
5. Ensure all tests pass before merging

## Configuration

- **Precedence**: Environment variables > .env file > JSON config files
- **Secrets**: Never commit API keys or credentials
- **Changes**: Document in `.env.example`

See `docs/02-PROVIDERS-SETUP.md` for provider configuration details.

## Documentation

- Update relevant docs when code changes
- Document new endpoints in API blueprints
- Add docstrings to services and workers
- Update `docs/` files for user-facing changes
- Keep `CLAUDE.md` updated with recent work

## Debugging

- **Worker logs**: `docker-compose logs -f backend` or terminal output
- **Database**: Enable SQLAlchemy echo in `database_service.py`
- **API requests**: Add logging to Flask blueprints
- **Memory state**: Use REST API endpoints to inspect state

## Performance Considerations

- **Memory system**: Uses hierarchical compression (working memory → gists → episodes → concepts)
- **Routing**: Deterministic ~5ms router before LLM generation
- **Timeouts**: All operations have hard timeouts (ACT loop 60s, actions 10s each)
- **Database queries**: Use PostgreSQL efficiently, leverage pgvector for semantic search

## Safety & Constraints

When modifying the system, maintain these guardrails:

- **Prompt immutability**: Prompts are marked as "authoritative and final"
- **Skill registry**: Fixed at startup (no runtime registration)
- **Data scope**: All queries scoped to topic (no cross-topic leakage)
- **Safety gates**: AND logic for proactive messaging (quality + timing + engagement)
- **Operational limits**: Hard timeouts, fatigue budgets, cooldowns

## Questions?

- Check `CLAUDE.md` for project guidance
- Review architecture docs in `docs/`
- Look at existing code patterns in similar files
- Ask in project discussions or create an issue

---

**Thanks for contributing!** 🎉
