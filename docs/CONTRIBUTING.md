# Contributing to Chalie

[MANIFESTO.md](MANIFESTO.md) is the only mandatory read — it carries the vision and the twelve principles every change is reviewed against. This file is the mechanics.

## Repository layout

- `backend/` — Python backend (Flask + SQLite). Entry point: `backend/run.py`. Key packages: `services/` (business logic), `abilities/` (built-in tools), `capabilities/` (external-system adapters), `api/` (REST + WebSocket), `models/`, `workers/`, `migrations/` + `schema.sql`.
- `frontend/` — pnpm workspace (Vue 3 + TypeScript + Vite): `apps/interface` (the user-facing UI), `apps/brain` (the admin/cognitive dashboard), shared code under `packages/`.

## Setup & running

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e backend/                 # core dependencies
pip install -e 'backend/[voice-cpu]'    # optional TTS/STT (GPU/ROCm variants in backend/pyproject.toml)
python backend/run.py                   # SQLite auto-initializes; no external services required
```

There is no `.env` and no environment-variable configuration: code-level config is Python constants, runtime settings live in the Brain interface, and secrets auto-generate on first run.

Frontend:

```bash
pnpm install
pnpm dev:interface    # or dev:brain
pnpm build            # builds all apps; `pnpm typecheck` and `pnpm lint` before pushing
```

Frontend changes ship with the rebuilt `dist/` committed.

## Tests

Three markers, three jobs (`backend/pytest.ini`): `unit` (deterministic, no external dependencies), `integration` (needs local services), `e2e` (live network).

```bash
cd backend && pytest -m unit -q    # must pass before any merge — no exceptions
```

Write feature tests, not mock theater: drive the real entry point on the real stack and assert what a user would feel (MANIFESTO P11). A bug fix includes the test that reproduces the bug. A test that breaks without behavior changing asserts implementation — delete it.

## Code conventions

- All datetimes are timezone-aware UTC via `services.time_utils` (`utc_now()` / `parse_utc()`) — never `datetime.now()`, `utcnow()`, or `fromisoformat()`.
- Both themes, always: frontend styling uses the shared theme's CSS variables, never hardcoded single-theme colors.
- Repo paths resolve through `FileMapperService` — no `Path(__file__)` outside it; runtime `os.path.join(root, …)` is fine, and `sys.path.insert` bootstraps are exempt.
- Schema changes ship with a migration in `backend/migrations/` in the same commit.
- No single-use variables — inline `call_func(y)`, not `x = y; call_func(x)`.
- Match the surrounding code's style and comment density; comments explain *why*, never *what*.
- A feature that introduces a new concept adds its term to [VOCABULARY.md](VOCABULARY.md).

## Git workflow

1. Branch from `main`: `git checkout -b feature/your-feature`.
2. Keep the diff lean — smallest change that works, zero residue (MANIFESTO P5, P8).
3. Open a PR describing the change and the evidence behind it.
4. The unit suite must be green. Merging and releasing are always maintainer-manual.

## Security

Read [SECURITY.md](SECURITY.md) before touching credentials, tool execution, or network surfaces. Never commit secrets — credentials live in the encrypted vault, never in files.
