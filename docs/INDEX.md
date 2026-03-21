# Chalie Documentation

## Start Here

| Document | What You'll Learn |
|---|---|
| [00-VISION.md](00-VISION.md) | Product vision, design principles, feature decision filter |
| [01-QUICK-START.md](01-QUICK-START.md) | Install, configure, and start using Chalie |
| [02-PROVIDERS-SETUP.md](02-PROVIDERS-SETUP.md) | Configure LLM providers (Ollama, Anthropic, OpenAI, Gemini) |
| [FAQ.md](FAQ.md) | Common questions and answers |

## For Developers

**Recommended reading order:**

1. [13-MESSAGE-FLOW.md](13-MESSAGE-FLOW.md) — Visual map of every path, storage hit, and LLM call
2. [04-ARCHITECTURE.md](04-ARCHITECTURE.md) — All services, workers, database schema, and data flow
3. [09-TOOLS.md](09-TOOLS.md) — How tools work and how to add new ones
4. [14-DEFAULT-TOOLS.md](14-DEFAULT-TOOLS.md) — Built-in tools reference
5. [15-INTERFACES.md](15-INTERFACES.md) — External app integration protocol
6. [12-TESTING.md](12-TESTING.md) — Test conventions and mock strategies

## Specialized Topics

| Document | Contents |
|---|---|
| [03-WEB-INTERFACE.md](03-WEB-INTERFACE.md) | UI spec and Radiant design system |
| [16-AMBIENT-AWARENESS.md](16-AMBIENT-AWARENESS.md) | Place inference, attention tracking, event bridge |
| [17-CURIOSITY-SYSTEM.md](17-CURIOSITY-SYSTEM.md) | Self-directed exploration threads |
| [18-SIGNAL-CONTRACT.md](18-SIGNAL-CONTRACT.md) | Signal-driven continuous reasoning spec |

## Key Directories

```
backend/
├── services/          # Business logic
├── workers/           # Background daemon threads
├── api/               # REST API blueprints
├── tools/             # First-party tool modules
├── prompts/           # LLM prompt templates
├── tests/             # Test suite
└── run.py             # Single entry point

frontend/
├── interface/         # Main chat UI
├── brain/             # Admin dashboard
└── on-boarding/       # Setup wizard
```

## Quick Reference

- **Start Chalie**: `./run.sh` or `python backend/run.py`
- **Run tests**: `cd backend && pytest`
- **Database**: `backend/data/chalie.db` (SQLite)
- **Default port**: 8081
- **Schema**: `backend/schema.sql`

## Other Project Files

- [CLAUDE.md](../CLAUDE.md) — Development guidance for Claude Code
- [CONTRIBUTING.md](../CONTRIBUTING.md) — Contribution guidelines
- [README.md](../README.md) — Project overview
