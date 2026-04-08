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

# Force numpy to fully initialize before any background thread imports it.
# Python's import system isn't fully thread-safe for nested imports — concurrent
# first-imports of numpy from multiple threads cause a circular import in
# numpy._typing (NDArray not yet available from the partially-initialized module),
# which poisons sys.modules and makes every subsequent embedding call fail with
# "maximum recursion depth exceeded".
try:
    import numpy  # noqa: F401 — thread-safety warm-up
    import torch  # noqa: F401 — thread-safety warm-up
    import transformers  # noqa: F401 — thread-safety warm-up
    # These heavy imports must complete in the main thread before any background
    # thread tries to import them. Python's import system isn't fully thread-safe
    # for complex nested imports — concurrent first-imports from multiple threads
    # cause circular import errors in numpy._typing that poison sys.modules.
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

    # Preload embedding model in a background thread so Flask starts immediately.
    # On first run the model (~438MB) may need to download from HuggingFace;
    # blocking here would prevent the onboarding page from loading for 5+ minutes.
    def _preload_embedding_model():
        try:
            logger.info("[System] Preloading embedding model (background)...")
            from services.embedding_service import get_embedding_service
            svc = get_embedding_service()
            # Warm the ONNX session — first encode() triggers model load and,
            # on first run, a ~300MB HuggingFace download. Running here so the
            # user never hits that delay during an actual conversation.
            svc.generate_embedding("warmup")
            logger.info("[System] Embedding model ready (inference warm)")
        except Exception as e:
            import traceback
            logger.warning(f"[System] Embedding model preload failed: {e}")
            logger.warning(f"[System] Preload traceback:\n{traceback.format_exc()}")

    import threading as _threading
    _threading.Thread(target=_preload_embedding_model, name="embedding-preload", daemon=True).start()

    # Download/update ONNX classifiers, then warm the inference path.
    def _preload_onnx_models():
        try:
            logger.info("[System] Checking ONNX models (background)...")
            from services.onnx_inference_service import get_onnx_inference_service
            svc = get_onnx_inference_service()
            # Download missing models / version-check existing ones
            svc.ensure_models()
            # Warm the mode-tiebreaker — load session + tokenizer + throwaway inference
            label, _ = svc.predict("mode-tiebreaker", "warmup")
            if label is not None:
                logger.info("[System] ONNX mode-tiebreaker ready (inference warm)")
            else:
                logger.info("[System] ONNX mode-tiebreaker not available — higher-score fallback active")
            svc._ready = True
        except Exception as e:
            logger.warning(f"[System] ONNX preload failed: {e}")

    _threading.Thread(target=_preload_onnx_models, name="onnx-preload", daemon=True).start()

    # Initialize SQLite database
    from services.database_service import get_shared_db_service
    from services.schema_service import SchemaService
    from services.config_service import ConfigService

    episodic_config = ConfigService.resolve_agent_config("episodic-memory")
    embedding_dimensions = episodic_config.get('embedding_dimensions', 768)

    database_service = get_shared_db_service()
    schema_service = SchemaService(database_service, embedding_dimensions)

    if not schema_service.database_exists():
        logger.info("Initializing database...")

    # Always apply schema.sql — every CREATE TABLE/INDEX uses IF NOT EXISTS, so this is
    # fully idempotent. Running it on every startup ensures new tables added in any commit
    # are created in existing databases without requiring an explicit migration.
    schema_service.initialize_schema()
    current_version = schema_service.schema_version()
    logger.info(f"Schema applied (version {current_version})")

    # Always ensure vec tables exist — idempotent, repairs existing DBs missing new tables
    schema_service.ensure_vec_tables()

    # Run pending migrations
    logger.info("Checking for pending database migrations...")
    database_service.run_pending_migrations()

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
    from services.experience_assimilation_service import experience_assimilation_worker
    from services.scheduler_service import scheduler_worker
    from services.autobiography_service import autobiography_synthesis_worker
    from workers.persistent_task_worker import persistent_task_worker
    from workers.document_worker import document_purge_worker
    from services.world_awareness_service import world_awareness_worker

    # Initialize worker manager
    manager = WorkerManager()

    # Register service workers
    manager.register_service("decay-engine-service", decay_engine_worker)
    manager.register_service("dmn-service", dmn_worker)
    manager.register_service("experience-assimilation-service", experience_assimilation_worker)
    manager.register_service("scheduler-service", scheduler_worker)
    manager.register_service("autobiography-synthesis-service", autobiography_synthesis_worker)
    manager.register_service("persistent-task-worker", persistent_task_worker)
    manager.register_service("document-purge-service", document_purge_worker)
    manager.register_service("world-awareness-service", world_awareness_worker)

    from workers.folder_watcher_worker import folder_watcher_worker
    manager.register_service("folder-watcher-service", folder_watcher_worker)

    from workers.interface_health_worker import interface_health_worker
    manager.register_service("interface-health-monitor", interface_health_worker)

    from workers.interface_daemon_worker import interface_daemon_worker
    manager.register_service("interface-daemon-watcher", interface_daemon_worker)

    # Moment enrichment service
    from services.moment_enrichment_service import moment_enrichment_worker
    manager.register_service("moment-enrichment-service", moment_enrichment_worker)

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
    # Register cron-triggered tools
    registry = None
    try:
        from services.tool_registry_service import ToolRegistryService
        registry = ToolRegistryService()
        for tool in registry.get_cron_tools():
            worker_func = registry.create_cron_worker(tool)
            manager.register_service(f"tool-{tool['name']}-service", worker_func)
        tool_count = len(registry.get_tool_names())
        if tool_count > 0:
            logger.info(f"[Startup] Tool registry loaded: {tool_count} tools")
    except Exception as e:
        logger.warning(f"[Startup] Tool cron registration failed: {e}")

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
                from services.database_service import get_shared_db_service
                from services.knowledge_service import KnowledgeService
                db = get_shared_db_service()
                ks = KnowledgeService(db)

                # Already exists — nothing to do
                existing = ks.get('system', 'user_summary')
                if existing and existing.get('value'):
                    logger.info("[Startup] User trait sentence exists")
                    return

                # No sentence yet — check if traits exist and synthesize
                traits = db.fetch_all(
                    "SELECT key, value, confidence, decay_class FROM knowledge "
                    "WHERE entity = 'user' AND kind = 'trait' AND deleted_at IS NULL "
                    "ORDER BY decay_class DESC, confidence DESC"
                )
                if not traits:
                    return

                trait_lines = []
                for row in traits:
                    key = row['key'] if isinstance(row, dict) else row[0]
                    value = row['value'] if isinstance(row, dict) else row[1]
                    confidence = row['confidence'] if isinstance(row, dict) else row[2]
                    decay_class = row['decay_class'] if isinstance(row, dict) else row[3]
                    trait_lines.append(f"{key}: {value} (confidence: {confidence:.2f}, {decay_class})")

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

                ks.store(
                    kind='fact', entity='system', key='user_summary',
                    value=sentence, decay_class='permanent',
                    confidence=1.0, source='trait_synthesis',
                )
                logger.info("[Startup] User trait sentence synthesized from %d traits", len(traits))
            except Exception as e:
                logger.warning(f"[Startup] Trait sentence bootstrap failed: {e}")
        threading.Thread(target=_bootstrap_trait_sentence, daemon=True, name="trait-sentence-bootstrap").start()
    except Exception as e:
        logger.warning(f"[Startup] Trait sentence bootstrap start failed: {e}")

    # Warm search router embedding cache (background thread)
    if os.environ.get('CHALIE_SKIP_SEARCH_CACHE') != '1':
        try:
            def _warm_search_cache():
                try:
                    from tools.search.router import _ensure_cache
                    _ensure_cache()
                    logger.info("[Startup] Search router cache ready")
                except Exception as e:
                    logger.warning(f"[Startup] Search router cache warmup failed: {e}")
            threading.Thread(target=_warm_search_cache, daemon=True, name="search-cache-warmup").start()
        except Exception as e:
            logger.warning(f"[Startup] Search cache warmup start failed: {e}")

    # Hourly cleanup for stale pending contradictions
    def _pending_contradiction_cleanup_loop():
        import time
        from services.pending_contradiction_service import PendingContradictionService
        from services.database_service import get_shared_db_service
        while True:
            try:
                db = get_shared_db_service()
                svc = PendingContradictionService(db)
                count = svc.cleanup_stale()
                if count > 0:
                    logging.info(f"[PENDING_CONTRADICTION] Cleanup: processed {count} stale records")
            except Exception as e:
                logging.warning(f"[PENDING_CONTRADICTION] Cleanup error: {e}")
            time.sleep(3600)

    threading.Thread(target=_pending_contradiction_cleanup_loop, daemon=True, name="pending-contradiction-cleanup").start()

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
