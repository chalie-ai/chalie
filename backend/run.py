#!/usr/bin/env python3
"""Single entry point for Chalie.

CLI: `python backend/run.py [--port=9000] [--host=127.0.0.1]`
All worker threads, database initialisation, and the Flask+WebSocket server
run in a single process. Voice runs natively when deps are installed.
"""

import argparse
import os
import sys
import logging
from typing import TYPE_CHECKING, cast

from utils.logger import Logger

if TYPE_CHECKING:
    import ssl
    from services.embedding_service import EmbeddingService
    from services.onnx_inference_service import OnnxInferenceService
    from services.database_service import DatabaseService
    from consumer import WorkerManager as _WorkerManager

# Force numpy/transformers to fully initialize before any background thread
# imports them. Python's import system isn't fully thread-safe for nested
# imports — concurrent first-imports from multiple threads cause a circular
# import in numpy._typing (NDArray not yet available from the
# partially-initialized module), which poisons sys.modules and makes every
# subsequent embedding call fail with "maximum recursion depth exceeded".
try:
    import numpy  # noqa: F401 — thread-safety warm-up
    import transformers  # noqa: F401 — thread-safety warm-up
except Exception as _e:
    import sys as _sys
    print(f"[BOOT] CRITICAL: import failed: {_e}", file=_sys.stderr, flush=True)

# Ensure backend/ is on the Python path
_backend_dir = os.path.dirname(os.path.abspath(__file__))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

Logger.start()
logger = logging.getLogger(__name__)




def _start_model_preload() -> None:
    """Kick off background model preloading (embedding + ONNX classifiers)."""
    def _preload_models() -> None:
        try:
            logger.info("[System] Preloading embedding model (background)...")
            from services.embedding_service import get_embedding_service
            svc: "EmbeddingService | OnnxInferenceService | None" = get_embedding_service()
            cast("EmbeddingService", svc).generate_embedding("warmup")
            logger.info("[System] Embedding model ready (inference warm)")
        except Exception as e:
            import traceback
            logger.warning(f"[System] Embedding model preload failed: {e}")
            logger.warning(f"[System] Preload traceback:\n{traceback.format_exc()}")

        svc = None
        try:
            logger.info("[System] Registering ONNX classifier heads (background)...")
            from services.onnx_inference_service import get_onnx_inference_service
            svc = get_onnx_inference_service()
            from services.onnx_inference_service import MODEL_REGISTRY as _CLASSIFIER_REGISTRY
            _failures: list[tuple[str, str]] = []
            for _task, _ in _CLASSIFIER_REGISTRY:
                try:
                    svc._get_head(_task)
                except RuntimeError as _reg_err:
                    logger.error(
                        f"[System] CLASSIFIER REGISTRATION FAILED — task={_task} "
                        f"reason={_reg_err}"
                    )
                    _failures.append((_task, str(_reg_err)))
            svc._failed_registrations = _failures
            svc._ready = True
            if _failures:
                logger.error(
                    f"[System] ONNX classifier DEGRADED — {len(_failures)} task(s) "
                    f"failed registration: {[t for t, _ in _failures]} — health "
                    f"endpoint will report not-ready until resolved"
                )
            else:
                logger.info("[System] ONNX classifier heads registered")
        except Exception as e:
            if svc is not None:
                svc._failed_registrations = [("preload", str(e))]
                svc._ready = True
            logger.exception("[System] ONNX preload failed")

    import threading as _threading
    _threading.stack_size(2 * 1024 * 1024)
    _threading.Thread(target=_preload_models, name="model-preload", daemon=True).start()


def _migrate_legacy_policy_rules(database_service: "DatabaseService") -> None:
    """Upgrade-path only — on a fresh install this is a no-op. Critically does
    NOT create the ``policy`` table early: making the DB look non-empty to
    convergence's freshness check would skip the schema.sql seed pass (incl.
    the row that marks ``api_key`` sensitive, persisting the REST API key in
    cleartext on fresh installs)."""
    with database_service.connection() as conn:
        has_legacy = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='policy_rules'"
        ).fetchone() is not None
        if not has_legacy:
            return
        conn.execute("""
            CREATE TABLE IF NOT EXISTS policy (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel TEXT NOT NULL, permission TEXT NOT NULL,
                setting TEXT NOT NULL CHECK (setting IN ('internal','allow','ask','deny')),
                UNIQUE (channel, permission)
            )
        """)
        conn.execute(
            "INSERT OR IGNORE INTO policy (channel, permission, setting) "
            "SELECT context, action_id, state FROM policy_rules"
        )
        conn.commit()
    logger.info("[Startup] Legacy policy_rules copied into flat policy table")


def _init_database() -> "DatabaseService":
    from services.database_service import get_shared_db_service
    from services.schema_convergence_service import SchemaConvergenceService

    database_service = get_shared_db_service()

    # Provider deletion is permanent — there is no longer a soft-delete flag.
    # Any rows still parked at is_active=0 are previously-deleted providers;
    # purge them BEFORE convergence drops the column, otherwise they resurface
    # as active providers (and collide on the UNIQUE name index). Self-disabling:
    # once convergence removes is_active this block is a no-op.
    try:
        with database_service.connection() as _conn:
            _cols = [r[1] for r in _conn.execute("PRAGMA table_info(providers)").fetchall()]
            if 'is_active' in _cols:
                _purged = _conn.execute("DELETE FROM providers WHERE is_active = 0").rowcount
                _conn.commit()
                if _purged:
                    logger.info("[Startup] Purged %d soft-deleted provider row(s)", _purged)
    except Exception as _prov_err:
        logger.warning(f"[Startup] soft-deleted provider purge skipped: {_prov_err}")

    # Policy (upgrade path): copy legacy policy_rules → policy BEFORE convergence
    # drops policy_rules.  No-op on a fresh DB so it stays "fresh" and convergence
    # runs schema.sql's seed pass (incl. the api_key is_sensitive row).
    try:
        _migrate_legacy_policy_rules(database_service)
    except Exception as _pol_err:
        logger.warning(f"[Startup] legacy policy_rules copy skipped: {_pol_err}")

    convergence = SchemaConvergenceService(database_service)
    convergence.converge()
    # Separate deterministic value backfill — convergence applies only static
    # column DEFAULTs, never derived values (last_relevant_at, valid_from, etc.).
    convergence.backfill_redesign_columns()

    # Policy: apply the declarative seed (idempotent) AFTER convergence has created
    # the policy table.  INSERT OR IGNORE preserves any copied/user rows.
    try:
        from services.policy_manager import PolicyManager
        inserted = PolicyManager(database_service).apply_seed()
        logger.info("[Startup] Policy seed applied (%d new rows)", inserted)
    except Exception as _seed_err:
        logger.warning(f"[Startup] policy seed skipped: {_seed_err}")

    return database_service


def _run_startup_migrations(database_service: "DatabaseService") -> None:
    """Run all one-time startup migrations and data backfills."""
    import os as _os

    # Token-limit backfill
    try:
        from services.provider_token_limits import backfill_all
        with database_service.connection() as _conn:
            _stats = backfill_all(_conn)
            _conn.commit()
        logger.info(
            "[Startup] providers token-limit backfill: total=%d succeeded=%d failed=%d",
            _stats['total'], _stats['succeeded'], _stats['failed'],
        )
    except Exception as _bf_err:
        logger.warning(f"[Startup] providers max_tokens/compact_at backfill skipped: {_bf_err}")

    # One-time transcript rebuild
    try:
        from migrate_transcript_rebuild import run_once_on_boot
        run_once_on_boot(db_path=database_service.db_path)
    except Exception as _mig_err:
        logger.warning(f"[Startup] Transcript migration skipped: {_mig_err}")

    # One-time compactions-table migration — purge legacy role='compaction'
    # transcript rows now that compaction state lives in its own table. Runs after
    # the rebuild above (which leaves those rows in place). Idempotent; sentinel
    # gates it to one run per database.
    try:
        _compaction_sentinel = _os.path.join(
            _os.path.dirname(database_service.db_path),
            '.compactions-table-migration-v1.done',
        )
        if not _os.path.exists(_compaction_sentinel):
            from migrations.migration_002_compactions_table import apply as _apply_compactions
            _apply_compactions(database_service.db_path)
            with open(_compaction_sentinel, 'w') as _f:
                _f.write('done')
    except Exception as _comp_err:
        logger.warning(f"[Startup] compactions-table migration skipped: {_comp_err}")

    # Drop zombie invoked_by column
    try:
        _drop_sentinel = _os.path.join(
            _os.path.dirname(database_service.db_path),
            '.tool-calls-drop-invoked-by-v1.done',
        )
        if not _os.path.exists(_drop_sentinel):
            with database_service.connection() as _conn:
                _cols = [r[1] for r in _conn.execute("PRAGMA table_info(tool_calls)").fetchall()]
                if 'invoked_by' in _cols:
                    _conn.execute("ALTER TABLE tool_calls DROP COLUMN invoked_by")
                    _conn.commit()
                    logger.info("[Startup] Dropped zombie invoked_by column from tool_calls")
            with open(_drop_sentinel, 'w') as _f:
                _f.write('done')
    except Exception as _drop_err:
        logger.warning(f"[Startup] tool_calls invoked_by drop skipped: {_drop_err}")

    # Drop legacy ephemeral column — every tool call is now durable; rows older
    # than 7 days are removed by DecayEngineService._purge_tool_calls() instead.
    try:
        _ephem_sentinel = _os.path.join(
            _os.path.dirname(database_service.db_path),
            '.tool-calls-drop-ephemeral-v1.done',
        )
        if not _os.path.exists(_ephem_sentinel):
            with database_service.connection() as _conn:
                _cols = [r[1] for r in _conn.execute("PRAGMA table_info(tool_calls)").fetchall()]
                if 'ephemeral' in _cols:
                    _conn.execute("ALTER TABLE tool_calls DROP COLUMN ephemeral")
                    _conn.commit()
                    logger.info("[Startup] Dropped ephemeral column from tool_calls (durable retention)")
            with open(_ephem_sentinel, 'w') as _f:
                _f.write('done')
    except Exception as _ephem_err:
        logger.warning(f"[Startup] tool_calls ephemeral drop skipped: {_ephem_err}")

    # One-time transcript.settled backfill. SchemaConvergence adds the column as
    # all-zeros (NOT NULL DEFAULT 0); without this every pre-existing assistant
    # row reads settled=0, so settle0 is NULL for all historical turns and the
    # spine/feed/GC break. migration_006 re-derives settled from tool_calls to the
    # exact runtime predicate. Idempotent; sentinel-gated to one run per database.
    try:
        _settled_sentinel = _os.path.join(
            _os.path.dirname(database_service.db_path),
            '.transcript-settled-backfill-v1.done',
        )
        if not _os.path.exists(_settled_sentinel):
            from migrations.migration_006_transcript_settled import apply as _apply_settled
            _apply_settled(database_service.db_path)
            with open(_settled_sentinel, 'w') as _f:
                _f.write('done')
    except Exception as _settled_err:
        logger.warning(f"[Startup] transcript.settled backfill skipped: {_settled_err}")

    # One-time episodes FTS rebuild
    try:
        _sentinel = _os.path.join(_os.path.dirname(database_service.db_path), '.episodes-fts-rebuild-v1.done')
        if not _os.path.exists(_sentinel):
            with database_service.connection() as _conn:
                _conn.execute("INSERT INTO episodes_fts(episodes_fts) VALUES('rebuild')")
                _conn.commit()
            with open(_sentinel, 'w') as _f:
                _f.write('done')
            logger.info("[Startup] episodes_fts rebuilt from content table")
    except Exception as _fts_err:
        logger.warning(f"[Startup] episodes FTS rebuild skipped: {_fts_err}")

    # Purge stale AdaptiveLayer data_graph rows
    try:
        _adaptive_keys = (
            'prefers_concise', 'prefers_depth', 'enjoys_challenge',
            'prefers_bullet_format', 'challenge_tolerance',
        )
        with database_service.connection() as _conn:
            _placeholders = ','.join('?' * len(_adaptive_keys))
            _conn.execute(
                f"DELETE FROM data_graph WHERE kind='user_specific' AND key IN ({_placeholders})",
                _adaptive_keys,
            )
            _conn.commit()
    except Exception as _adl_err:
        logger.warning(f"[Startup] AdaptiveLayer data_graph purge skipped: {_adl_err}")


def _init_services(database_service: "DatabaseService") -> None:
    """Initialize singleton services and seed default configuration."""
    # Clean up expired auth sessions
    try:
        from services.auth_session_service import cleanup_expired_sessions
        cleanup_expired_sessions()
    except Exception:
        pass

    logger.info("[Startup] Encryption key deferred to post-login (vault mode)")
    logger.info("[Startup] Capability reconnection deferred to post-login (vault mode)")

    # Initialize API key
    try:
        from services.settings_service import SettingsService
        settings_service = SettingsService(database_service)
        api_key = settings_service.get_api_key_or_generate()
        logger.info(f"[Settings] API key initialized (key: ...{api_key[-8:]})")
    except Exception as e:
        logger.warning(f"Settings initialization failed: {e}")

    from services.write_queue_service import get_write_queue as _get_write_queue
    _get_write_queue()
    logger.info("[Startup] WriteQueueService started")

    from services.telemetry_service import get_telemetry_collector as _get_telemetry_collector
    _get_telemetry_collector()
    logger.info("[Startup] TelemetryCollector initialized")


def _register_workers(manager: "_WorkerManager", host: str, port: int) -> None:
    """Register all service workers with the WorkerManager."""
    from services.scheduler_service import scheduler_worker
    from workers.document_worker import document_purge_worker
    from services.world_awareness_service import world_awareness_worker

    manager.register_service("scheduler-service", scheduler_worker)
    manager.register_service("document-purge-service", document_purge_worker)
    manager.register_service("world-awareness-service", world_awareness_worker)

    from workers.folder_watcher_worker import folder_watcher_worker
    manager.register_service("folder-watcher-service", folder_watcher_worker)

    from services.subconscious_worker import subconscious_worker
    manager.register_service("subconscious-worker", subconscious_worker)

    from workers.tmp_cleanup_worker import tmp_cleanup_worker
    manager.register_service("tmp-cleanup-service", tmp_cleanup_worker)

    _bootstrap_capability_sync()
    _try_register(manager, "search-expander-service",
                  "services.search_expander_service", "search_expander_worker")
    _try_register(manager, "mcp-server", "mcp_server.server", "run_mcp_server")
    _try_register(manager, "mcp-client-heartbeat",
                  "workers.mcp_client_worker", "mcp_client_worker")

    def _flask_worker() -> None:
        from api import create_app
        app = create_app()
        ssl_context = _resolve_ssl_context()
        scheme = "https" if ssl_context else "http"
        logger.info(f"[Chalie] Starting on {scheme}://{host}:{port}")
        app.run(host=host, port=port, debug=False, threaded=True, ssl_context=ssl_context)

    manager.register_service("rest-api-worker-1", _flask_worker)


def _disable_ssl_setting() -> None:
    """Clear ``ssl_enabled`` after a TLS fallback so the cookie Secure-scope tracks the real scheme.

    Without this a vanished cert leaves ``ssl_enabled=true`` in the DB while the
    server runs HTTP — issuing ``Secure`` cookies the browser drops, locking out login.
    """
    try:
        from services.settings_service import SettingsService
        from services.database_service import get_shared_db_service
        SettingsService(get_shared_db_service()).set_bool(SettingsService.SSL_ENABLED, False)
    except Exception as exc:
        logger.error("[SSL] could not clear ssl_enabled after TLS fallback: %s", exc)


def _resolve_ssl_context() -> "ssl.SSLContext | None":
    """Build the server TLS context when SSL is enabled and a valid cert/key exist.

    Returns ``None`` (plain HTTP) when SSL is off. If SSL is enabled but the cert/key
    are missing or fail to load, clears ``ssl_enabled`` and serves HTTP — degrading
    rather than crash-looping, and keeping the cookie Secure-scope honest.
    """
    try:
        from services.settings_service import SettingsService
        from services.database_service import get_shared_db_service
        if not SettingsService(get_shared_db_service()).get_bool(SettingsService.SSL_ENABLED):
            return None
        from services.file_mapper_service import FileMapperService
        cert = FileMapperService.get_ssl_cert_path()
        key = FileMapperService.get_ssl_key_path()
        if cert.is_file() and key.is_file():
            import ssl
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(str(cert), str(key))
            return context
        logger.warning("[SSL] enabled but certificate/key missing — disabling SSL, serving HTTP")
    except Exception as exc:
        logger.error("[SSL] context load failed — disabling SSL, serving HTTP: %s", exc)
    _disable_ssl_setting()
    return None


def _check_asset_caches() -> None:
    """Verify search routing and concept LUT assets are present."""
    import sqlite3 as _sql
    from services.file_mapper_service import FileMapperService

    try:
        _search_db = FileMapperService.get_search_providers_db_path()
        if _search_db.exists():
            _c = _sql.connect(str(_search_db))
            _tables = [r[0] for r in _c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            _c.close()
            if "example_embeddings" in _tables:
                logger.info("[Startup] Search router cache ready")
            else:
                logger.warning("[Startup] Search embeddings missing — run 'cd backend && python -m utils.generate_search_cache'")
        else:
            logger.warning("[Startup] search_tool_providers.sqlite not found")
    except Exception as e:
        logger.warning(f"[Startup] Search cache check failed: {e}")

    try:
        _lut_db = FileMapperService.get_concept_lut_db_path()
        if _lut_db.exists():
            _c = _sql.connect(str(_lut_db))
            _tables = [r[0] for r in _c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            _c.close()
            if "lut_embeddings" in _tables:
                logger.info("[Startup] Concept LUT ready: %s", _lut_db)
            else:
                logger.warning("[Startup] Concept LUT embeddings missing — run 'cd backend && python -m utils.generate_concept_lut'")
        else:
            logger.warning("[Startup] concept_lut.sqlite not found — run 'cd backend && python -m utils.generate_concept_lut'")
    except Exception as e:
        logger.warning(f"[Startup] Concept LUT check failed: {e}")


def _warmup_models() -> None:
    """Warm up voice and embedding models in background daemon threads."""
    import threading as _t
    try:
        from api.voice import _ensure_models as _voice_warm
        _t.Thread(target=_voice_warm, name="voice-warmup", daemon=True).start()
    except Exception as e:
        logger.warning(f"[Startup] Voice warm-up skipped: {e}")
    try:
        from services.embedding_service import _get_session_and_tokenizer as _embed_warm
        _t.Thread(target=_embed_warm, name="embed-warmup", daemon=True).start()
    except Exception as e:
        logger.warning(f"[Startup] Embedding warm-up skipped: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chalie — personal intelligence layer")
    parser.add_argument("--port", type=int, default=31025, help="Server port (default: 31025)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    args = parser.parse_args()

    port = args.port
    host = args.host

    import runtime_config
    runtime_config.set({"port": port, "host": host})

    # Apply any staged whole-instance restore BEFORE the DB is opened. The first
    # chalie.db open happens inside _init_database() → converge() below, so the
    # artifact swap must complete here or convergence would run against the old
    # DB. SnapshotService.apply_pending owns its own try/except (boot safety):
    # a failed restore is rolled back + logged and never aborts startup.
    # Paired with services/snapshot_service.py (Phase B) and api/snapshot.py.
    from services.snapshot_service import SnapshotService
    SnapshotService.apply_pending()

    # Guarantee a usable onnxruntime BEFORE any warmup thread imports it. On a
    # host whose GPU/ROCm wheel can't load its native libs (e.g. libcudart
    # missing), this swaps to the CPU wheel and logs an actionable ERROR hint
    # (visible in the Cognition → Errors panel); a no-op when import already works.
    from services.runtime_deps_service import RuntimeDepsService
    RuntimeDepsService.ensure_onnxruntime()

    _start_model_preload()

    database_service = _init_database()
    _run_startup_migrations(database_service)
    _init_services(database_service)

    _check_asset_caches()
    _warmup_models()

    # Background-install optional runtime deps (playwright, voice if enabled)
    RuntimeDepsService.ensure_playwright()
    RuntimeDepsService.init_voice_from_settings(database_service)

    from consumer import WorkerManager
    manager = WorkerManager()
    _register_workers(manager, host, port)

    manager.run()


def _bootstrap_capability_sync() -> None:
    """Bootstrap connected capabilities at startup.

    For each capability, call connect() — it checks credentials internally and
    registers its sync handler + ensures recurring scheduled_items exist.
    Capability tools are surfaced to the LLM via Ability subclasses (email,
    calendar, contacts) which are auto-discovered by AbilityRegistry.
    """
    try:
        from capabilities import load_capabilities
        all_caps = load_capabilities()
        for cap_id, cap in all_caps.items():
            try:
                if not cap.is_connected():
                    cap.connect()
                if cap.is_connected():
                    logger.info("[bootstrap] Auto-connected capability: %s", cap_id)
            except Exception as exc:
                logger.warning("[bootstrap] Failed to auto-connect %s: %s", cap_id, exc)
    except Exception as exc:
        logger.warning("[bootstrap] Capability sync bootstrap failed: %s", exc)


def _try_register(manager: "_WorkerManager", name: str, module_path: str, func_name: str) -> None:
    """Try to import and register a service, logging failure gracefully."""
    try:
        import importlib
        mod = importlib.import_module(module_path)
        func = getattr(mod, func_name)
        manager.register_service(name, func)
    except Exception as e:
        logger.warning(f"[Startup] {name} registration failed: {e}")


if __name__ == "__main__":
    main()
