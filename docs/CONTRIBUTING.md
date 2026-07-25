# Contributing to Chalie

[MANIFESTO.md](MANIFESTO.md) is the only mandatory read — it carries the vision and the twelve principles every change is reviewed against. This file is the mechanics.

## Repository layout

- `backend/` — Python backend (Flask + SQLite). Entry point: `backend/run.py`. Key packages: `services/` (business logic), `abilities/` (built-in tools), `capabilities/` (external-system adapters), `api/` (REST + WebSocket), `models/`, `workers/`, `migrations/` + `schema.sql`.
- `frontend/` — pnpm workspace (Vue 3 + TypeScript + Vite): `apps/interface` (the user-facing UI), `apps/brain` (the admin/cognitive dashboard), shared code under `packages/`.

## Setup & running

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e backend/                 # all dependencies, voice (TTS/STT/VAD) included
python backend/run.py                   # SQLite auto-initializes; no external services required
```

There is no `.env` and no environment-variable configuration: code-level config is Python constants, runtime settings live in the Brain interface, and secrets auto-generate on first run.

This manual flow syncs Python dependencies only. The Playwright browser and the on-device voice models are fetched once by `installer/install.sh` (or the Docker build), not by `pip install -e backend/` or `python backend/run.py` — abilities that need them fail loudly with a reinstall hint until you've run the installer at least once.

`run.py` will not start with a dependency missing. Before any heavy import it checks every entry in `backend/pyproject.toml` against the installed distributions; if one is absent it names it on stderr, serves a terminal error page on the public port instead of the starting page, and exits non-zero. A backend that boots without its dependencies silently takes the degraded branch of some `except ImportError` and lies about its own state. So after a pull that changes `pyproject.toml`, re-run `pip install -e backend/`.

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
- Schema changes ship with a migration in `backend/migrations/` in the same commit. A migration module exposes `needed(conn)` (self-contained precondition — False on databases already in target shape) and `apply(db_path)`, and registers in `migrations/runner.py`'s `_STEPS`; the runner records each outcome once in the `schema_migrations` ledger at boot.
- No single-use variables — inline `call_func(y)`, not `x = y; call_func(x)`.
- Match the surrounding code's style and comment density; comments explain *why*, never *what*.
- New REST endpoints subclass `api.endpoint.Endpoint` (CRUD groups, `backend/api/endpoints/`) or `api.action.Action` (verb operations, `backend/api/actions/<slug>/<verb>.py`) — the base owns routing, auth, and response envelopes; handlers never build them. A controller declares no path of its own: mount it in `backend/api/routes.py`, the single table every URL comes from. Legacy module-level Namespaces are being migrated onto this contract.
- A feature that introduces a new concept adds its term to [VOCABULARY.md](VOCABULARY.md).

## Git workflow

1. Branch from `main`: `git checkout -b feature/your-feature`.
2. Keep the diff lean — smallest change that works, zero residue (MANIFESTO P5, P8).
3. Open a PR describing the change and the evidence behind it.
4. The unit suite must be green. Merging and releasing are always maintainer-manual.

## Security

Read [SECURITY.md](SECURITY.md) before touching credentials, tool execution, or network surfaces. Never commit secrets — credentials live in the encrypted vault, never in files.
