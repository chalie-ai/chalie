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
├── ws.js               # WebSocket client
├── renderer.js         # Card and message rendering
├── voice.js            # Voice interaction
├── presence.js         # Presence/status indicators
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
├── workers/            # Background worker threads
├── api/                # REST API + WebSocket blueprints
├── configs/            # Configuration files
├── data/               # SQLite database (auto-created)
├── prompts/            # LLM prompt templates
├── tests/              # Test suite
├── schema.sql          # Database schema
└── run.py              # Single entry point
```

## Development Workflow

### Setup

```bash
cd backend
pip install -r requirements.txt
source .venv/bin/activate  # or use poetry
# Optional: copy .env.example to .env only if you need a non-default PORT or VOICE_ENABLED=false
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
# Start everything (SQLite auto-initializes, no external services required)
python backend/run.py

# Custom port
PORT=9000 python backend/run.py

# Run Flask directly for debugging (without background workers)
cd backend && python -c "from api import create_app; app = create_app(); app.run(host='0.0.0.0', port=8081)"
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
4. Register in `run.py` if background execution needed
5. Document in this file's "Architecture" section

### Adding a New Worker

1. Create `backend/workers/my_worker.py` extending `WorkerBase`
2. Implement worker main loop
3. Register in `run.py` as a daemon thread
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
- **New env vars**: Document in root `.env.example` (only PORT and VOICE_ENABLED belong there; all secrets auto-generate)

See `docs/02-PROVIDERS-SETUP.md` for provider configuration details.

## Documentation

- Update relevant docs when code changes
- Document new endpoints in API blueprints
- Add docstrings to services and workers
- Update `docs/` files for user-facing changes
- Keep `CLAUDE.md` updated with recent work

## Debugging

- **All logs**: `python backend/run.py` (single process, all logs to stdout)
- **Database**: SQLite file at `backend/data/chalie.db` — inspect with `sqlite3` CLI
- **API requests**: Add logging to Flask blueprints
- **WebSocket**: Browser DevTools → Network → WS tab
- **Memory state**: Use REST API endpoints to inspect state

## Performance Considerations

- **Memory system**: Uses hierarchical compression (working memory → gists → episodes → concepts)
- **Routing**: Deterministic ~5ms router before LLM generation
- **Timeouts**: All operations have hard timeouts (ACT loop 60s, actions 10s each)
- **Database queries**: Use SQLite efficiently, leverage sqlite-vec for semantic search

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
