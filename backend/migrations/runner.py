"""Startup migration runner — the single authority for one-time data migrations.

Boot calls :func:`run_all` once per start (run.py), AFTER SchemaConvergenceService
has converged the schema: the ``schema_migrations`` ledger table is declared in
``schema.sql`` and must already exist, and worker threads are not running yet so
steps execute against a quiet database.

Each step is a module in this package exposing the migration contract:

- ``needed(conn) -> bool`` — self-encapsulated precondition: does the database
  still carry the legacy shape/data this migration exists to fix? A fresh
  install (schema.sql already in target shape) answers False everywhere, so
  every step records as a no-op without executing. An upgrading database
  answers True only for the steps its vintage actually needs.
- ``apply(db_path) -> None`` — the migration body (owns its own connection;
  also runnable standalone via the module's CLI).

Guard order per step, cheapest first:

1. ledger row exists → skip (completed on an earlier boot);
2. legacy boot sentinel file (``.<key>.done`` beside the DB, written by the
   pre-ledger per-migration wrappers) → record as ``'sentinel'`` without
   executing. The file is deliberately left in place so a code downgrade to
   the wrapper era stays gated too;
3. ``needed()`` False → record as ``'noop'`` without executing;
4. otherwise ``apply()``, then record as ``'applied'``.

The ledger lives inside the database so applied state travels with it through
snapshot export/restore, and destructive steps (the scheduled_items wipe and
rebuild) re-check ``needed()`` instead of ever re-running blind when tracking
is absent. A failing step logs a warning and retries next boot (no ledger row
is written); boot never aborts. Recording a step also (re)writes its
wrapper-era sentinel file(s) beside the database — fresh installs included —
so a code downgrade to the pre-ledger wrappers stays gated everywhere, not
just on machines that upgraded through that era. To re-fire a shipped
migration against already-upgraded databases (migration_008's v1→v2 pattern),
give it a new key and list the old stem in ``_LEGACY_STEMS`` so the old
wrapper's gate stays satisfied too.
"""

from __future__ import annotations

import importlib
import logging
import os
import sqlite3

from migrations.connection import connect

logger = logging.getLogger(__name__)

# Execution order is positional, preserved from the pre-ledger boot wrappers.
# Each key doubles as the legacy sentinel stem: '.{key}.done' beside the DB.
_STEPS: tuple[tuple[str, str], ...] = (
    ("compactions-table-migration-v1", "migrations.migration_002_compactions_table"),
    ("tool-calls-drop-invoked-by-v1", "migrations.migration_011_drop_tool_calls_invoked_by"),
    ("tool-calls-drop-ephemeral-v1", "migrations.migration_012_drop_tool_calls_ephemeral"),
    ("tool-calls-id-primary-key-v1", "migrations.migration_014_tool_calls_id_primary_key"),
    ("transcript-settled-backfill-v1", "migrations.migration_006_transcript_settled"),
    ("episodes-fts-rebuild-v1", "migrations.migration_013_rebuild_episodes_fts"),
    ("scheduled-items-cron-wipe-v1", "migrations.migration_007_scheduler_cron"),
    ("scheduled-items-prompt-only-v2", "migrations.migration_008_scheduler_prompt_only"),
    ("system-kind-split-migration-v1", "migrations.migration_009_system_kind_split"),
    ("episode-search-queries-migration-v1", "migrations.migration_010_episode_search_queries"),
)

# Extra wrapper-era sentinel stems a step's resolution must also satisfy:
# migration_008's ledger key was bumped v1→v2 to re-fire, but the old boot
# wrapper still gates its destructive rebuild on the v1 stem.
_LEGACY_STEMS: dict[str, tuple[str, ...]] = {
    "scheduled-items-prompt-only-v2": ("scheduled-items-prompt-only-v1",),
}


def run_all(db_path: str) -> None:
    """Run every unrecorded migration step against ``db_path``, recording each
    outcome in the ``schema_migrations`` ledger. Never raises — a failing step
    warns and is retried on the next boot."""
    conn = connect(db_path)
    try:
        if not _ledger_exists(conn):
            logger.warning(
                "[migrations] schema_migrations table missing — schema convergence "
                "has not run against %s; skipping all startup migrations", db_path,
            )
            return
        recorded = {row[0] for row in conn.execute("SELECT key FROM schema_migrations")}
        for key, module_path in _STEPS:
            if key not in recorded:
                _run_step(conn, db_path, key, module_path)
    finally:
        conn.close()


def _run_step(conn: sqlite3.Connection, db_path: str, key: str, module_path: str) -> None:
    """Apply one step through the sentinel-import → needed() → apply() guards."""
    try:
        if _legacy_sentinel_exists(db_path, key):
            _record(conn, db_path, key, "sentinel")
            logger.info("[migrations] %s: imported pre-ledger sentinel — already applied", key)
            return
        module = importlib.import_module(module_path)
        if not module.needed(conn):
            _record(conn, db_path, key, "noop")
            logger.info("[migrations] %s: not needed — database already in target shape", key)
            return
        module.apply(db_path)
        _record(conn, db_path, key, "applied")
        logger.info("[migrations] %s: applied", key)
    except Exception as exc:
        logger.warning(
            "[migrations] %s skipped (will retry next boot): %s", key, exc, exc_info=True
        )


def _record(conn: sqlite3.Connection, db_path: str, key: str, outcome: str) -> None:
    """Write the ledger row and (re)write the wrapper-era sentinel file(s), so
    a code downgrade to the pre-ledger boot wrappers stays gated for every
    resolved step — including fresh installs, which never had the files."""
    for stem in (key, *_LEGACY_STEMS.get(key, ())):
        open(os.path.join(os.path.dirname(db_path), f".{stem}.done"), "a").close()
    conn.execute(
        "INSERT INTO schema_migrations (key, outcome) VALUES (?, ?)", (key, outcome)
    )
    conn.commit()


def _legacy_sentinel_exists(db_path: str, key: str) -> bool:
    return os.path.exists(os.path.join(os.path.dirname(db_path), f".{key}.done"))


def _ledger_exists(conn: sqlite3.Connection) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
    ).fetchone() is not None
