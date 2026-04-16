#!/usr/bin/env python3
"""
Single entry point for Chalie.

Start with:
    python backend/run.py

CLI options:
    python backend/run.py --port=9000
    python backend/run.py --host=127.0.0.1

All worker threads, database initialization, and the Flask+WebSocket server
run in a single process. Voice runs natively when deps are installed.
"""

import argparse
import os
import sys
import logging

from utils.logger import Logger

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




def main():
    parser = argparse.ArgumentParser(description="Chalie — personal intelligence layer")
    parser.add_argument("--port", type=int, default=8081, help="Server port (default: 8081)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--models-dir", default=None, help="ONNX models directory (default: /models or MODELS_DIR env)")
    args = parser.parse_args()

    port = args.port
    host = args.host

    # Store in runtime_config so any module can access these values
    import runtime_config
    config = {"port": port, "host": host}
    if args.models_dir:
        config["models_dir"] = args.models_dir
    runtime_config.set(config)

    # Preload models in a single background thread so Flask starts immediately.
    # Models are loaded sequentially to avoid concurrent memory spikes that
    # exceed the Docker VM limit (embedding + ONNX loaded in parallel = OOM).
    def _preload_models():
        try:
            logger.info("[System] Preloading embedding model (background)...")
            from services.embedding_service import get_embedding_service
            svc = get_embedding_service()
            svc.generate_embedding("warmup")
            logger.info("[System] Embedding model ready (inference warm)")
        except Exception as e:
            import traceback
            logger.warning(f"[System] Embedding model preload failed: {e}")
            logger.warning(f"[System] Preload traceback:\n{traceback.format_exc()}")

        try:
            logger.info("[System] Checking ONNX models (background)...")
            from services.onnx_inference_service import get_onnx_inference_service
            svc = get_onnx_inference_service()
            svc.ensure_models()
            # Trigger task registration now so boot markers ([CLASSIFIER BOOT] ...)
            # fire before any user request arrives. The encoder session is already
            # warm (loaded by embedding preload above), so registration is fast.
            # _get_head() calls _register_task() which emits the boot marker and
            # validates the sha256 pin — any mismatch raises RuntimeError here.
            from services.onnx_inference_service import MODEL_REGISTRY as _CLASSIFIER_REGISTRY
            _failures: list[tuple[str, str]] = []
            for _task, _ in _CLASSIFIER_REGISTRY:
                try:
                    svc._get_head(_task)
                except RuntimeError as _reg_err:
                    # sha256 gate refused this task. Surface this loudly — silent
                    # fallback in production would defeat the boot-pin invariant.
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
            logger.warning(f"[System] ONNX preload failed: {e}")

    import threading as _threading
    _threading.stack_size(2 * 1024 * 1024)
    _threading.Thread(target=_preload_models, name="model-preload", daemon=True).start()

    # Initialize SQLite database — declarative convergence
    from services.database_service import get_shared_db_service
    from services.schema_convergence_service import SchemaConvergenceService

    database_service = get_shared_db_service()
    convergence = SchemaConvergenceService(database_service)
    convergence.converge()

    # One-time transcript rebuild (runs exactly once; sentinel file prevents re-runs)
    try:
        from migrate_transcript_rebuild import run_once_on_boot
        run_once_on_boot(db_path=database_service.db_path)
    except Exception as _mig_err:
        logger.warning(f"[Startup] Transcript migration skipped: {_mig_err}")

    # Seed data_graph from legacy knowledge table (runs once — idempotent check inside)
    try:
        from services.data_graph_service import seed_from_legacy_knowledge
        seed_from_legacy_knowledge(database_service)
    except Exception as _seed_err:
        logger.warning(f"[Startup] data_graph seed skipped: {_seed_err}")

    # Drop zombie `invoked_by` column from tool_calls (NOT NULL, never populated by new code)
    try:
        import os as _os
        _drop_sentinel = _os.path.join(
            _os.path.dirname(database_service.db_path),
            '.tool-calls-drop-invoked-by-v1.done',
        )
        if not _os.path.exists(_drop_sentinel):
            with database_service.connection() as _conn:
                # Check if column still exists before attempting drop
                _cols = [r[1] for r in _conn.execute("PRAGMA table_info(tool_calls)").fetchall()]
                if 'invoked_by' in _cols:
                    _conn.execute("ALTER TABLE tool_calls DROP COLUMN invoked_by")
                    _conn.commit()
                    logger.info("[Startup] Dropped zombie invoked_by column from tool_calls")
            with open(_drop_sentinel, 'w') as _f:
                _f.write('done')
    except Exception as _drop_err:
        logger.warning(f"[Startup] tool_calls invoked_by drop skipped: {_drop_err}")

    # One-time episodes FTS rebuild — external-content FTS5 was never synced on insert
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

    # Clean up expired auth sessions from SQLite
    try:
        from services.auth_session_service import cleanup_expired_sessions
        cleanup_expired_sessions()
    except Exception:
        pass

    # Encryption key initialisation and capability reconnection are deferred to
    # the post-login hook in user_auth.py (_reconnect_capabilities).  The vault
    # requires an interactive password to unseal, so neither step can run at
    # boot time in vault mode.
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

    # Start the write-queue singleton so its daemon drain thread is running before
    # any service worker tries to submit writes.  The singleton is created here
    # (rather than lazily on first use) so the background thread is alive for the
    # full lifetime of the process.
    from services.write_queue_service import get_write_queue as _get_write_queue
    _get_write_queue()
    logger.info("[Startup] WriteQueueService started")

    # Initialise the telemetry collector singleton before any service worker
    # starts emitting events.  Services call get_telemetry_collector().record()
    # directly (thread-safe), so the singleton must exist first to ensure the
    # ring buffer and per-type counters are ready from boot.
    from services.telemetry_service import get_telemetry_collector as _get_telemetry_collector
    _get_telemetry_collector()
    logger.info("[Startup] TelemetryCollector initialized")

    # Import consumer's WorkerManager and all services
    from consumer import WorkerManager

    # Import worker functions
    from services.decay_engine_service import decay_engine_worker
    from services.dmn_service import dmn_worker
    from services.tool_synthesis_processor import tool_synthesis_worker
    from services.scheduler_service import scheduler_worker
    from workers.document_worker import document_purge_worker
    from services.world_awareness_service import world_awareness_worker

    # Initialize worker manager
    manager = WorkerManager()

    # Register service workers
    manager.register_service("decay-engine-service", decay_engine_worker)
    manager.register_service("dmn-service", dmn_worker)
    manager.register_service("tool-synthesis-service", tool_synthesis_worker)
    manager.register_service("scheduler-service", scheduler_worker)
    manager.register_service("document-purge-service", document_purge_worker)
    manager.register_service("world-awareness-service", world_awareness_worker)

    from workers.folder_watcher_worker import folder_watcher_worker
    manager.register_service("folder-watcher-service", folder_watcher_worker)

    from workers.interface_health_worker import interface_health_worker
    manager.register_service("interface-health-monitor", interface_health_worker)

    from workers.interface_daemon_worker import interface_daemon_worker
    manager.register_service("interface-daemon-watcher", interface_daemon_worker)

    # Moment context enrichment service (6h worker)
    from services.moment_context_service import moment_context_worker
    manager.register_service("moment-context-service", moment_context_worker)

    # Self-model service (interoception — epistemic, operational, capability awareness)
    from services.self_model_service import self_model_worker
    manager.register_service("self-model-service", self_model_worker)

    # Background LLM worker
    from workers.background_llm_worker import background_llm_worker
    manager.register_service("background-llm-worker", background_llm_worker)

    # Capability sync — bootstrap connected capabilities into scheduler system handlers
    _bootstrap_capability_sync()

    # Optional services (fail gracefully)
    _try_register(manager, "growth-pattern-service",
                  "services.growth_pattern_service", "growth_pattern_worker")
    _try_register(manager, "routing-stability-regulator-service",
                  "services.routing_stability_regulator_service", "routing_stability_regulator_worker")
    _try_register(manager, "triage-calibration-service",
                  "services.triage_calibration_service", "triage_calibration_worker")
    _try_register(manager, "profile-enrichment-service",
                  "services.profile_enrichment_service", "profile_enrichment_worker")
    # Load tool registry
    try:
        from services.tool_registry_service import ToolRegistryService
        tool_count = len(ToolRegistryService().get_tool_names())
        if tool_count > 0:
            logger.info(f"[Startup] Tool registry loaded: {tool_count} tools")
    except Exception as e:
        logger.warning(f"[Startup] Tool registry load failed: {e}")

    # Bootstrap tool profiles (background thread)
    try:
        import threading
        from services.tool_profile_service import ToolProfileService
        def _run_bootstrap():
            try:
                ToolProfileService().bootstrap_all()
                logger.info("[Startup] Tool profile bootstrap complete")
            except Exception as e:
                logger.warning(f"[Startup] Tool profile bootstrap failed: {e}")
        threading.Thread(target=_run_bootstrap, daemon=True, name="profile-bootstrap").start()
    except Exception as e:
        logger.warning(f"[Startup] Tool profile bootstrap start failed: {e}")

    # Bootstrap user trait sentence — load from DB or synthesize if traits exist
    try:
        def _bootstrap_trait_sentence():
            try:
                from services.data_graph_service import get_data_graph_service

                dgs = get_data_graph_service()

                # Already exists — nothing to do
                existing = dgs.fetch(kinds=['system'], limit=1)
                summary_row = next((r for r in existing if r.get('key') == 'user_summary'), None)
                if summary_row and summary_row.get('value'):
                    logger.info("[Startup] User trait sentence exists")
                    return

                # No sentence yet — synthesize from user_specific rows
                traits = dgs.fetch(kinds=['user_specific'], order_by='retrieval_weight DESC', limit=50)
                if not traits:
                    return

                trait_lines = [
                    f"{r['key']}: {r['value']} (weight: {r.get('retrieval_weight', 1.0):.2f})"
                    for r in traits if r.get('key') and r.get('value')
                ]
                if not trait_lines:
                    return

                import os
                prompt_path = os.path.join(os.path.dirname(__file__), 'prompts', 'trait-synthesis.md')
                with open(prompt_path, 'r') as f:
                    template = f.read()
                prompt_text = template.replace('{{traits}}', '\n'.join(trait_lines))

                from services.provider_cache_service import ProviderCacheService
                provider_config = ProviderCacheService.resolve_for_job('trait-extraction')
                if not provider_config:
                    return

                from services.llm_service import create_llm_service
                llm = create_llm_service(provider_config)
                resp = llm.send_message(prompt_text, "Output only the sentence. No preamble, no explanation.")
                sentence = resp.text.strip() if resp and resp.text else None
                if not sentence:
                    return

                dgs.store(kind='system', key='user_summary', value=sentence, source='trait_synthesis')
                logger.info("[Startup] User trait sentence synthesized from %d traits", len(trait_lines))
            except Exception as e:
                logger.warning(f"[Startup] Trait sentence bootstrap failed: {e}")
        threading.Thread(target=_bootstrap_trait_sentence, daemon=True, name="trait-sentence-bootstrap").start()
    except Exception as e:
        logger.warning(f"[Startup] Trait sentence bootstrap start failed: {e}")

    # Verify search routing embeddings are present
    try:
        import os as _os
        import sqlite3 as _sql
        _search_db = _os.path.join(
            _os.path.dirname(__file__), "tools", "search", "assets", "search_tool_providers.sqlite"
        )
        if _os.path.exists(_search_db):
            _c = _sql.connect(_search_db)
            _tables = [r[0] for r in _c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            _c.close()
            if "example_embeddings" in _tables:
                logger.info("[Startup] Search router cache ready")
            else:
                logger.warning("[Startup] Search embeddings missing — run 'python -m utils.generate_search_cache'")
        else:
            logger.warning("[Startup] search_tool_providers.sqlite not found")
    except Exception as e:
        logger.warning(f"[Startup] Search cache check failed: {e}")

    # Verify concept LUT is present and has embeddings
    try:
        _lut_db = _os.path.join(
            _os.path.dirname(__file__), "services", "data_graph", "assets", "concept_lut.sqlite"
        )
        if _os.path.exists(_lut_db):
            _c = _sql.connect(_lut_db)
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

    # Register the Flask API worker (this is the main thread's HTTP server)
    def _flask_worker(shared_state=None):
        from api import create_app
        app = create_app()
        logger.info(f"[Chalie] Starting on http://{host}:{port}")
        app.run(host=host, port=port, debug=False, threaded=True)

    manager.register_service("rest-api-worker-1", _flask_worker)

    # Start everything
    manager.run()


def _bootstrap_capability_sync():
    """Bootstrap connected capabilities into the scheduler's system handler registry.

    For each capability, call connect() — it checks credentials internally and
    registers its sync handler + ensures recurring scheduled_items exist.
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
                    # Re-register dynamic tools so find_tools can discover them after restart
                    from services.tool_library_service import register_tool
                    for tool_def in cap.get_tools():
                        tool_name = tool_def["name"]
                        handler = tool_def.get("handler")
                        if handler is None:
                            continue
                        metadata = {k: v for k, v in tool_def.items() if k != "handler"}
                        try:
                            register_tool(tool_name, handler, metadata)
                            logger.info("[bootstrap] Registered tool '%s' for capability '%s'", tool_name, cap_id)
                        except Exception as reg_exc:
                            logger.warning("[bootstrap] Failed to register tool '%s': %s", tool_name, reg_exc)
            except Exception as exc:
                logger.warning("[bootstrap] Failed to auto-connect %s: %s", cap_id, exc)
        # If ToolRegistryService was already initialised before this bootstrap ran,
        # its in-memory tools dict is stale. Reload so find_tools can discover
        # the freshly registered capability tools.
        try:
            from services.tool_registry_service import ToolRegistryService
            reg = ToolRegistryService()
            if reg._initialized:
                reg._load_tools()
                logger.info("[bootstrap] Tool registry reloaded after capability tool registration")
        except Exception as reg_exc:
            logger.warning("[bootstrap] Tool registry reload failed: %s", reg_exc)
    except Exception as exc:
        logger.warning("[bootstrap] Capability sync bootstrap failed: %s", exc)


def _try_register(manager, name, module_path, func_name):
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
