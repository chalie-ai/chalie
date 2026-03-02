#!/usr/bin/env python3
"""
Single entry point for Chalie.

Start with:
    python backend/run.py

CLI options:
    python backend/run.py --port=9000
    python backend/run.py --host=127.0.0.1

All worker threads, database initialization, and the Flask+WebSocket server
run in a single process. Docker is optional — used only for sandboxed tool
execution. Voice runs natively when deps are installed.
"""

import argparse
import os
import sys
import logging

# Ensure backend/ is on the Python path
_backend_dir = os.path.dirname(os.path.abspath(__file__))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Chalie — personal intelligence layer")
    parser.add_argument("--port", type=int, default=8081, help="Server port (default: 8081)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    args = parser.parse_args()

    port = args.port
    host = args.host

    # Store in runtime_config so any module can access these values
    import runtime_config
    runtime_config.set({"port": port, "host": host})

    # Ensure encryption key
    from services.encryption_key_service import get_encryption_key
    get_encryption_key()

    # Preload embedding model singleton (eagerly load the actual model, not just the wrapper)
    try:
        logger.info("[System] Preloading embedding model...")
        from services.embedding_service import get_embedding_service, _get_st_model
        svc = get_embedding_service()
        _get_st_model(svc.model_name)
        logger.info("[System] Embedding model ready")
    except Exception as e:
        logger.warning(f"[System] Embedding model preload failed: {e}")

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
        schema_service.create_database()
        logger.info("Database initialized")
    else:
        current_version = schema_service.schema_version()
        if current_version == 0:
            logger.info("Initializing schema...")
            schema_service.initialize_schema()
            logger.info("Schema initialized")
        else:
            logger.info(f"Schema up to date (version {current_version})")

    # Always ensure vec tables exist — idempotent, repairs existing DBs missing new tables
    schema_service.ensure_vec_tables()

    # Run pending migrations
    logger.info("Checking for pending database migrations...")
    database_service.run_pending_migrations()

    # Initialize API key
    try:
        from services.settings_service import SettingsService
        settings_service = SettingsService(database_service)
        api_key = settings_service.get_api_key_or_generate()
        logger.info(f"[Settings] API key initialized (key: ...{api_key[-8:]})")
    except Exception as e:
        logger.warning(f"Settings initialization failed: {e}")

    # Import consumer's WorkerManager and all services
    from consumer import WorkerManager, ToolScannerThread

    # Import worker functions
    from services.idle_consolidation_service import idle_consolidation_process
    from services.decay_engine_service import decay_engine_worker
    from services.growth_pattern_service import growth_pattern_worker
    from services.topic_stability_regulator_service import topic_stability_regulator_worker
    from services.cognitive_drift_engine import cognitive_drift_worker
    from services.routing_stability_regulator_service import routing_stability_regulator_worker
    from services.routing_reflection_service import routing_reflection_worker
    from services.experience_assimilation_service import experience_assimilation_worker
    from services.thread_expiry_service import thread_expiry_worker
    from services.scheduler_service import scheduler_worker
    from services.autobiography_service import autobiography_synthesis_worker
    from services.curiosity_pursuit_service import curiosity_pursuit_worker
    from workers.persistent_task_worker import persistent_task_worker
    from workers.document_worker import document_purge_worker

    # Initialize worker manager
    manager = WorkerManager()

    # Register service workers
    manager.register_service("idle-consolidation-service", idle_consolidation_process)
    manager.register_service("decay-engine-service", decay_engine_worker)
    manager.register_service("growth-pattern-service", growth_pattern_worker)
    manager.register_service("topic-stability-regulator-service", topic_stability_regulator_worker)
    manager.register_service("cognitive-drift-engine", cognitive_drift_worker)
    manager.register_service("routing-stability-regulator-service", routing_stability_regulator_worker)
    manager.register_service("routing-reflection-service", routing_reflection_worker)
    manager.register_service("experience-assimilation-service", experience_assimilation_worker)
    manager.register_service("thread-expiry-service", thread_expiry_worker)
    manager.register_service("scheduler-service", scheduler_worker)
    manager.register_service("autobiography-synthesis-service", autobiography_synthesis_worker)
    manager.register_service("curiosity-pursuit-service", curiosity_pursuit_worker)
    manager.register_service("persistent-task-worker", persistent_task_worker)
    manager.register_service("document-purge-service", document_purge_worker)

    # Moment enrichment service
    from services.moment_enrichment_service import moment_enrichment_worker
    manager.register_service("moment-enrichment-service", moment_enrichment_worker)

    # Background LLM worker
    from workers.background_llm_worker import background_llm_worker
    manager.register_service("background-llm-worker", background_llm_worker)

    # Optional services (fail gracefully)
    _try_register(manager, "triage-calibration-service",
                  "services.triage_calibration_service", "triage_calibration_worker")
    _try_register(manager, "profile-enrichment-service",
                  "services.profile_enrichment_service", "profile_enrichment_worker")
    _try_register(manager, "temporal-pattern-service",
                  "services.temporal_pattern_service", "temporal_pattern_worker")
    _try_register(manager, "tool-update-checker",
                  "services.tool_update_service", "tool_update_worker")

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

    # Wire up hot-reload tool scanner
    if registry:
        try:
            scanner = ToolScannerThread(manager=manager, tools_dir=registry.tools_dir)
            manager._tool_scanner = scanner
            manager._tool_scanner_registry = registry
        except Exception as e:
            logger.warning(f"[Startup] Tool scanner setup failed: {e}")

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

    # Register the Flask API worker (this is the main thread's HTTP server)
    def _flask_worker(shared_state=None):
        from api import create_app
        app = create_app()
        logger.info(f"[Chalie] Starting on http://{host}:{port}")
        app.run(host=host, port=port, debug=False, threaded=True)

    manager.register_service("rest-api-worker-1", _flask_worker)

    # Start everything
    manager.run()


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
