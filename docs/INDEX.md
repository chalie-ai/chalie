# Chalie Documentation

## Start Here

| Document | What You'll Learn |
|---|---|
| [00-VISION.md](00-VISION.md) | What Chalie is, why it exists, how to evaluate new ideas |
| [01-QUICK-START.md](01-QUICK-START.md) | Install, run, and onboard |
| [02-PROVIDERS-SETUP.md](02-PROVIDERS-SETUP.md) | Wire up an LLM provider |
| [FAQ.md](FAQ.md) | Common questions |

## For Developers

1. [13-MESSAGE-FLOW.md](13-MESSAGE-FLOW.md) — How a message travels through the system
2. [04-ARCHITECTURE.md](04-ARCHITECTURE.md) — Runtime shape, memory, background reasoning
3. [09-TOOLS.md](09-TOOLS.md) — How tools work, how to add one
4. [14-DEFAULT-TOOLS.md](14-DEFAULT-TOOLS.md) — Tools shipped with Chalie
5. [15-INTERFACES.md](15-INTERFACES.md) — External app integration contract
6. [12-TESTING.md](12-TESTING.md) — Test philosophy and discipline

## Specialized

| Document | Contents |
|---|---|
| [03-WEB-INTERFACE.md](03-WEB-INTERFACE.md) | Radiant design system and the four SPAs |
| [16-AMBIENT-AWARENESS.md](16-AMBIENT-AWARENESS.md) | WorldState Signal contract — typed snapshot of last user message, heartbeat, device, local time |
| [18-SIGNAL-CONTRACT.md](18-SIGNAL-CONTRACT.md) | Signal-driven reasoning spine |

## Quick Reference

- Start: `./run.sh` or `python backend/run.py`
- Tests: `cd backend && pytest`
- Default port: 8081
