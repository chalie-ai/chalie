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

# Force numpy to fully initialize before any background thread imports it.
# Python's import system isn't fully thread-safe for nested imports — concurrent
# first-imports of numpy from multiple threads cause a circular import in
# numpy._typing (NDArray not yet available from the partially-initialized module),
# which poisons sys.modules and makes every subsequent embedding call fail with
# "maximum recursion depth exceeded".
try:
    import numpy  # noqa: F401
    import torch  # noqa: F401
    import transformers  # noqa: F401
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── Default Tool Auto-Install ───────────────────────────────────────────────

def _safe_tar_extract(tf, dest):
    """Extract tarball with path traversal and symlink protection."""
    dest_resolved = str(dest.resolve())
    sep = os.sep
    for member in tf.getmembers():
        member_path = (dest / member.name).resolve()
        if not str(member_path).startswith(dest_resolved + sep) and str(member_path) != dest_resolved:
            raise RuntimeError(f"Unsafe tar member rejected: {member.name}")
        if member.issym() or member.islnk():
            raise RuntimeError(f"Tar symlink/hardlink rejected: {member.name}")
    tf.extractall(dest)


def _install_embodiment(name, repo, version, tools_dir):
    """Download and install an embodiment at a pinned version.

    Uses stdlib only (urllib + tarfile). Short timeouts; atomic move via staging dir.
    """
    import tarfile
    import tempfile
    import shutil
    from pathlib import Path as _Path
    from urllib.request import urlopen
    from urllib.error import URLError

    logger.info(f"[Startup] Installing embodiment '{name}' ({version}) from {repo}")

    tmp_dir = None
    staging = tools_dir / f".{name}_installing"
    try:
        # 1. Download source tarball at pinned version (10s timeout)
        tarball_url = f"https://github.com/{repo}/archive/refs/tags/{version}.tar.gz"
        tmp_dir = _Path(tempfile.mkdtemp(prefix=f"chalie_embody_{name}_"))
        tarball_path = tmp_dir / "tool.tar.gz"
        with urlopen(tarball_url, timeout=10) as resp:
            tarball_path.write_bytes(resp.read())

        # 2. Safe extraction (path traversal + symlink protection)
        extract_dir = tmp_dir / "extracted"
        extract_dir.mkdir()
        with tarfile.open(tarball_path) as tf:
            _safe_tar_extract(tf, extract_dir)

        # GitHub tarballs have a single top-level dir like "repo-tag/"
        children = list(extract_dir.iterdir())
        source_dir = children[0] if len(children) == 1 and children[0].is_dir() else extract_dir

        # 3. Validate manifest exists
        if not (source_dir / "manifest.json").exists():
            logger.warning(f"[Startup] '{name}' ({version}) has no manifest.json, skipping")
            return

        # 4. Atomic install: move to staging, then rename to final path
        if staging.exists():
            shutil.rmtree(staging)
        final_path = tools_dir / name
        if final_path.exists():
            shutil.rmtree(final_path)
        shutil.move(str(source_dir), str(staging))
        staging.rename(final_path)
        logger.info(f"[Startup] Installed embodiment '{name}' ({version})")

        # 5. Record source metadata
        try:
            from services.tool_config_service import ToolConfigService
            from services.database_service import get_shared_db_service
            cfg = ToolConfigService(get_shared_db_service())
            cfg.set_tool_config(name, "_source_type", "embodiment")
            cfg.set_tool_config(name, "_source_url", f"https://github.com/{repo}")
            cfg.set_tool_config(name, "_installed_tag", version)
        except Exception:
            pass  # non-fatal

    except (URLError, OSError) as e:
        logger.warning(f"[Startup] Failed to install embodiment '{name}': {e}")
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    except Exception as e:
        logger.warning(f"[Startup] Unexpected error installing embodiment '{name}': {e}")
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    finally:
        if tmp_dir is not None and tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


def _sync_embodiments():
    """Ensure all embodiments are installed at their pinned versions.

    Downloads missing embodiments and upgrades/downgrades any that don't match
    the version declared in embodiment_library.json. Blocks until complete —
    Chalie is not fully functional without its embodiments.
    """
    import json as _json
    from pathlib import Path as _Path

    backend_dir = _Path(__file__).parent
    lib_path = backend_dir / "configs" / "embodiment_library.json"
    if not lib_path.exists():
        return

    tools_dir = backend_dir / "tools"

    with open(lib_path) as f:
        library = _json.load(f)

    pending = []
    for e in library:
        name = e.get("name")
        repo = e.get("repo")
        version = e.get("version")
        if not name or not repo or not version:
            continue

        tool_path = tools_dir / name
        if tool_path.exists():
            # Check if installed version matches pinned version
            try:
                from services.tool_config_service import ToolConfigService
                from services.database_service import get_shared_db_service
                cfg = ToolConfigService(get_shared_db_service())
                installed_tag = cfg.get_tool_config(name).get("_installed_tag")
                if installed_tag == version:
                    continue  # Already at correct version
                logger.info(f"[Startup] Embodiment '{name}' version mismatch: {installed_tag} → {version}")
            except Exception:
                continue  # Can't check version, assume OK
        pending.append((name, repo, version))

    if not pending:
        return

    logger.info(
        f"[Startup] Syncing {len(pending)} embodiment(s): "
        f"{[name for name, _, _ in pending]}"
    )
    for name, repo, version in pending:
        _install_embodiment(name, repo, version, tools_dir)
    installed = sum(1 for name, _, _ in pending if (tools_dir / name).exists())
    logger.info(f"[Startup] Embodiment sync complete ({installed}/{len(pending)} ready)")


# ─────────────────────────────────────────────────────────────────────────────

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

    # Ensure encryption key
    from services.encryption_key_service import get_encryption_key
    get_encryption_key()

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
    from services.cognitive_drift_engine import cognitive_drift_worker
    from services.experience_assimilation_service import experience_assimilation_worker
    from services.thread_expiry_service import thread_expiry_worker
    from services.episodic_memory_observer import episodic_memory_observer_worker
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
    manager.register_service("cognitive-drift-engine", cognitive_drift_worker)
    manager.register_service("experience-assimilation-service", experience_assimilation_worker)
    manager.register_service("thread-expiry-service", thread_expiry_worker)
    manager.register_service("episodic-memory-observer", episodic_memory_observer_worker)
    manager.register_service("scheduler-service", scheduler_worker)
    manager.register_service("autobiography-synthesis-service", autobiography_synthesis_worker)
    manager.register_service("curiosity-pursuit-service", curiosity_pursuit_worker)
    manager.register_service("persistent-task-worker", persistent_task_worker)
    manager.register_service("document-purge-service", document_purge_worker)

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

    # Optional services (fail gracefully)
    _try_register(manager, "growth-pattern-service",
                  "services.growth_pattern_service", "growth_pattern_worker")
    _try_register(manager, "topic-stability-regulator-service",
                  "services.topic_stability_regulator_service", "topic_stability_regulator_worker")
    _try_register(manager, "routing-stability-regulator-service",
                  "services.routing_stability_regulator_service", "routing_stability_regulator_worker")
    _try_register(manager, "triage-calibration-service",
                  "services.triage_calibration_service", "triage_calibration_worker")
    _try_register(manager, "profile-enrichment-service",
                  "services.profile_enrichment_service", "profile_enrichment_worker")
    _try_register(manager, "temporal-pattern-service",
                  "services.temporal_pattern_service", "temporal_pattern_worker")
    # tool-update-checker removed — embodiment versions are pinned in
    # embodiment_library.json and synced on boot, no periodic polling needed

    # Sync embodiments to pinned versions (synchronous, blocks until complete)
    _sync_embodiments()

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
