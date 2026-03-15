# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Consumer — Master supervisor for Chalie's single-process architecture.

All workers run as daemon threads in one Python process.
PromptQueue handles job dispatch via threads (no RQ dependency).
SQLite replaces PostgreSQL, MemoryStore replaces Redis.
"""

import os
import logging
import time
import signal
import sys
import threading
from typing import Dict, List, Tuple

def _read_version():
    """Read version from the VERSION file — single source of truth."""
    try:
        from pathlib import Path
        return Path(__file__).parent.parent.joinpath("VERSION").read_text().strip()
    except Exception:
        return "0.0.0"

APP_VERSION = _read_version()


def _thread_excepthook(args):
    """Global exception handler for threads — threads die silently by default."""
    logging.error(
        f"[ThreadException] Uncaught exception in thread '{args.thread.name}': "
        f"{args.exc_type.__name__}: {args.exc_value}",
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback)
    )


# Install global thread exception handler
threading.excepthook = _thread_excepthook


class WorkerManager:
    """Master supervisor for managing worker threads."""

    def __init__(self):
        """Initialise the WorkerManager with empty shared state and thread registry."""
        self.shared_state: Dict = {}
        self._state_lock = threading.Lock()
        self.threads: Dict[str, threading.Thread] = {}
        self.service_definitions: List[Tuple[str, callable]] = []
        self.running = True
        self._tool_scanner = None
        self._tool_scanner_registry = None

    def register_service(self, worker_id: str, worker_func):
        """Register a worker service definition for future spawning.

        The service is appended to the internal ``service_definitions`` list.
        It will be started when :meth:`spawn_all_services` is called, or
        immediately via :meth:`spawn_service`.

        Args:
            worker_id: Unique string identifier for the worker, also used as
                the daemon thread name (e.g., ``'decay-engine-service'``).
            worker_func: Callable with signature
                ``(shared_state: dict) -> None`` that implements the worker's
                blocking run loop.
        """
        self.service_definitions.append((worker_id, worker_func))
        logging.info(f"[Manager] Registered service definition: {worker_id}")

    def spawn_service(self, worker_id: str, worker_func):
        """Spawn a worker as a daemon thread, skipping if one is already alive.

        If a thread with the given ``worker_id`` already exists and is alive,
        this method returns immediately without creating a duplicate.  The
        thread passes ``shared_state`` to ``worker_func`` and logs any
        unhandled exception before terminating.

        Args:
            worker_id: Unique identifier and daemon thread name for the worker.
            worker_func: Callable with signature
                ``(shared_state: dict) -> None`` that implements the worker's
                blocking run loop.
        """
        if worker_id in self.threads and self.threads[worker_id].is_alive():
            return

        def _run():
            try:
                worker_func(self.shared_state)
            except Exception:
                logging.exception(f"[Manager] Service {worker_id} crashed")

        t = threading.Thread(target=_run, daemon=True, name=worker_id)
        t.start()
        self.threads[worker_id] = t
        logging.info(f"[Manager] Spawned service: {worker_id} (thread)")

    def spawn_all_services(self):
        """Spawn all registered service definitions as daemon threads.

        Iterates ``service_definitions`` in registration order and calls
        :meth:`spawn_service` for each entry.  Individual spawn failures are
        logged as errors but do not abort spawning of subsequent services.
        """
        logging.info("[Manager] Spawning all services...")
        for worker_id, worker_func in self.service_definitions:
            try:
                self.spawn_service(worker_id, worker_func)
            except Exception as e:
                logging.error(f"[Manager] Failed to spawn service '{worker_id}': {e}")

    def check_health(self):
        """Check service thread health and restart dead threads."""
        for worker_id, worker_func in self.service_definitions:
            try:
                t = self.threads.get(worker_id)
                if not t or not t.is_alive():
                    logging.warning(f"[Manager] Service {worker_id} is dead. Restarting...")
                    self.spawn_service(worker_id, worker_func)
            except Exception as e:
                logging.error(f"[Manager] Health check failed for service {worker_id}: {e}")

        # Publish thread health summary to MemoryStore for self-model consumption
        try:
            import json
            from services.memory_client import MemoryClientService
            store = MemoryClientService.create_connection()
            alive = [wid for wid, t in self.threads.items() if t.is_alive()]
            dead = [wid for wid, t in self.threads.items() if not t.is_alive()]
            store.setex("self_model:thread_health", 15, json.dumps({
                "alive": alive, "dead": dead, "total": len(self.threads),
            }))
        except Exception:
            pass  # never let health publication crash the health check

    def shutdown_all(self):
        """Initiate a graceful shutdown by clearing the running flag.

        Sets ``self.running = False`` so the :meth:`run` loop exits cleanly.
        Worker threads are daemon threads and are terminated automatically
        when the main thread finishes.
        """
        logging.info("\n[Manager] Initiating graceful shutdown...")
        self.running = False
        # Daemon threads will be killed when main thread exits
        logging.info("[Manager] All services stopped")

    def run(self):
        """Run the main supervisor loop, spawning services and monitoring thread health.

        Installs ``SIGINT`` and ``SIGTERM`` handlers that call
        :meth:`shutdown_all`.  Spawns all registered services via
        :meth:`spawn_all_services`, optionally starts the hot-reload tool
        scanner, then enters a 5-second polling loop.  Every iteration calls
        :meth:`check_health` to restart dead threads.  A periodic summary is
        logged every 5 minutes (60 × 5 s intervals).

        Blocks until :meth:`shutdown_all` is called or a
        ``KeyboardInterrupt`` is received, after which it ensures
        :meth:`shutdown_all` is invoked in the ``finally`` block.
        """
        signal.signal(signal.SIGINT, lambda sig, frame: self.shutdown_all())
        signal.signal(signal.SIGTERM, lambda sig, frame: self.shutdown_all())

        logging.info("[Manager] Starting Worker Manager (single-process, threaded)")
        self.spawn_all_services()

        # Start hot-reload tool scanner (daemon thread, in-process)
        if self._tool_scanner is not None:
            self._tool_scanner.start(self._tool_scanner_registry)

        try:
            health_check_counter = 0
            while self.running:
                time.sleep(5)
                self.check_health()

                health_check_counter += 1
                if health_check_counter >= 60:
                    alive = sum(1 for t in self.threads.values() if t.is_alive())
                    logging.info(f"[Manager] Health: {alive}/{len(self.threads)} threads alive")
                    health_check_counter = 0

        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown_all()


def _migrate_tools_disabled(tools_dir, db):
    """One-time startup migration: move tools from legacy tools_disabled/ back to tools/
    and write _enabled=false to DB for each."""
    import json as _json
    import shutil as _shutil
    from pathlib import Path as _Path
    disabled_dir = _Path(tools_dir).parent / "tools_disabled"
    if not disabled_dir.exists():
        return
    from services.tool_config_service import ToolConfigService
    config_svc = ToolConfigService(db)
    for entry in sorted(disabled_dir.iterdir()):
        if not entry.is_dir():
            continue
        dest = _Path(tools_dir) / entry.name
        if dest.exists():
            _shutil.rmtree(str(dest))
        _shutil.move(str(entry), str(dest))
        try:
            with open(dest / "manifest.json") as f:
                tool_name = _json.load(f).get("name", entry.name)
        except Exception:
            tool_name = entry.name
        config_svc._set_enabled_flag(tool_name, enabled=False)
        logging.info(f"[Startup] Migrated disabled tool '{tool_name}' -> tools/ with _enabled=false")
    try:
        disabled_dir.rmdir()
    except OSError:
        pass


class ToolScannerThread:
    """
    Daemon thread that scans backend/tools/ for new tool directories
    and registers them without a restart.
    """

    DEFAULT_INTERVAL = 30

    def __init__(self, manager: WorkerManager, tools_dir):
        """Initialise the tool scanner for the given tools directory.

        Args:
            manager: The :class:`WorkerManager` instance used to register and
                spawn cron workers discovered during directory scans.
            tools_dir: Filesystem path to the ``backend/tools/`` directory
                that is monitored for new tool sub-directories.  The scan
                interval can be overridden via the
                ``TOOL_SCANNER_INTERVAL_SECONDS`` environment variable
                (default: ``30`` seconds).
        """
        self._manager = manager
        self._tools_dir = tools_dir
        self._interval = int(os.environ.get("TOOL_SCANNER_INTERVAL_SECONDS", self.DEFAULT_INTERVAL))
        self._registry = None

    def start(self, registry):
        """Start the background scan loop and wire up the tool-registered callback.

        Stores a reference to ``registry``, installs
        :meth:`_on_tool_registered` as the post-build callback, then starts
        the ``_scan_loop`` as a daemon thread named ``'tool-scanner'``.

        Args:
            registry: ``ToolRegistryService`` instance used to inspect known
                tools, query build statuses, and create cron worker callables.
        """
        self._registry = registry
        registry.set_on_tool_registered(self._on_tool_registered)
        t = threading.Thread(target=self._scan_loop, name="tool-scanner", daemon=True)
        t.start()
        logging.info(f"[ToolScanner] Started (interval={self._interval}s)")

    def _on_tool_registered(self, tool_name: str):
        """Callback fired after a successful build. Spawns cron worker if needed."""
        tool = self._registry.tools.get(tool_name)
        if not tool:
            return
        trigger = tool["manifest"].get("trigger", {})
        if trigger.get("type") != "cron":
            return

        worker_id = f"tool-{tool_name}-service"
        existing = self._manager.threads.get(worker_id)
        if existing and existing.is_alive():
            return

        tool_config = {
            "name": tool_name,
            "schedule": trigger["schedule"],
            "prompt": trigger["prompt"],
            "image": tool["image"],
            "sandbox": tool.get("sandbox", {}),
            "dir": tool["dir"],
            "manifest": tool["manifest"],
        }
        worker_func = self._registry.create_cron_worker(tool_config)
        self._manager.register_service(worker_id, worker_func)
        self._manager.spawn_service(worker_id, worker_func)
        logging.info(f"[ToolScanner] Spawned cron worker: {worker_id}")

    def _scan_loop(self):
        """Run periodic scans at the configured interval, logging errors without crashing.

        Sleeps for one full interval before the first scan to allow other
        services to finish initialising.  Errors inside :meth:`_scan_once`
        are caught and logged so the scanner thread never terminates
        unexpectedly.
        """
        time.sleep(self._interval)
        while True:
            try:
                self._scan_once()
            except Exception as e:
                logging.error(f"[ToolScanner] Scan error: {e}")
            time.sleep(self._interval)

    def _scan_once(self):
        """Scan ``tools_dir`` once for new tool directories and trigger async builds.

        A directory is skipped if any of the following conditions apply:

        * It is already present in the registry (``known``).
        * It is currently being built (``building``) or has an active install
          lock (``locked``).
        * It is missing a ``manifest.json`` or ``Dockerfile``.
        * It is administratively disabled via ``ToolConfigService``.

        Eligible new directories are handed off to
        ``registry.register_tool_async`` for background image builds.
        """
        import json as _json
        from pathlib import Path
        if not self._tools_dir.exists():
            return

        known = set(self._registry.tools.keys())
        building = {n for n, s in self._registry.get_all_build_statuses().items()
                    if s.get("status") == "building"}
        locked = set(self._registry._install_locks)
        # Track tools that failed to build — don't retry every cycle
        failed = {n for n, s in self._registry.get_all_build_statuses().items()
                  if s.get("status") in ("failed", "error")}

        for entry in sorted(self._tools_dir.iterdir()):
            if not entry.is_dir() or entry.name.startswith(("_", ".")):
                continue
            if not (entry / "manifest.json").exists():
                continue
            # Trusted tools use runner.py (no Docker); sandboxed tools use Dockerfile.
            # Accept either — reject dirs that have neither.
            if not (entry / "runner.py").exists() and not (entry / "Dockerfile").exists():
                continue
            try:
                with open(entry / "manifest.json") as f:
                    tool_name = _json.load(f).get("name", "").strip()
            except Exception:
                continue
            if not tool_name or tool_name in known or tool_name in building or tool_name in locked:
                continue
            if tool_name in failed:
                continue
            try:
                from services.tool_config_service import ToolConfigService
                from services.database_service import get_shared_db_service
                if not ToolConfigService(get_shared_db_service()).is_tool_enabled(tool_name):
                    logging.debug(f"[ToolScanner] Ignoring disabled tool '{tool_name}'")
                    continue
            except Exception:
                pass
            logging.info(f"[ToolScanner] Discovered new tool '{tool_name}', starting build")
            try:
                self._registry.register_tool_async(entry)
            except Exception as e:
                logging.warning(f"[ToolScanner] Build start failed for '{tool_name}': {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Deferred imports
    from services import PromptQueue, DatabaseService, SchemaService
    from workers import digest_worker, episodic_memory_worker, semantic_consolidation_worker, rest_api_worker, tool_worker
    from services.config_service import ConfigService
    from services.idle_consolidation_service import idle_consolidation_process
    from services.decay_engine_service import decay_engine_worker
    from services.cognitive_drift_engine import cognitive_drift_worker
    from services.routing_reflection_service import routing_reflection_worker
    from services.experience_assimilation_service import experience_assimilation_worker
    from services.thread_expiry_service import thread_expiry_worker
    from services.scheduler_service import scheduler_worker
    from services.autobiography_service import autobiography_synthesis_worker
    from services.curiosity_pursuit_service import curiosity_pursuit_worker
    from workers.persistent_task_worker import persistent_task_worker
    from workers.document_worker import process_document_job, document_purge_worker

    # Ensure encryption key
    from services.encryption_key_service import get_encryption_key
    get_encryption_key()

    # Preload embedding model singleton
    try:
        logging.info("[System] Preloading embedding model...")
        from services.embedding_service import get_embedding_service
        get_embedding_service()
        logging.info("[System] Embedding model ready")
    except Exception as e:
        logging.warning(f"[System] Embedding model preload failed: {e}")

    # Initialize SQLite database
    from services.database_service import get_shared_db_service, get_db_path
    from services.config_service import ConfigService

    episodic_config = ConfigService.resolve_agent_config("episodic-memory")
    embedding_dimensions = episodic_config.get('embedding_dimensions', 768)

    database_service = get_shared_db_service()
    schema_service = SchemaService(database_service, embedding_dimensions)

    if not schema_service.database_exists():
        logging.info("Initializing database...")
        schema_service.create_database()
        logging.info("Database initialized")
    else:
        current_version = schema_service.schema_version()
        if current_version == 0:
            logging.info("Initializing schema...")
            schema_service.initialize_schema()
            logging.info("Schema initialized")
        else:
            logging.info(f"Schema up to date (version {current_version})")

    # Run pending migrations
    logging.info("Checking for pending database migrations...")
    database_service.run_pending_migrations()

    # Initialize API key
    try:
        from services.settings_service import SettingsService
        settings_service = SettingsService(database_service)
        api_key = settings_service.get_api_key_or_generate()
        logging.info(f"[Settings] API key initialized (key: ...{api_key[-8:]})")
    except Exception as e:
        logging.warning(f"Settings initialization failed: {e}")

    # Initialize worker manager
    manager = WorkerManager()

    # Register service workers (all run as daemon threads)
    manager.register_service("idle-consolidation-service", idle_consolidation_process)
    manager.register_service("decay-engine-service", decay_engine_worker)
    manager.register_service("cognitive-drift-engine", cognitive_drift_worker)
    manager.register_service("routing-reflection-service", routing_reflection_worker)
    manager.register_service("rest-api-worker-1", rest_api_worker)
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

    # Profile enrichment service
    try:
        from services.profile_enrichment_service import profile_enrichment_worker
        manager.register_service("profile-enrichment-service", profile_enrichment_worker)
    except Exception as e:
        logging.warning(f"[Consumer] Profile enrichment service registration failed: {e}")

    # Temporal pattern service
    try:
        from services.temporal_pattern_service import temporal_pattern_worker
        manager.register_service("temporal-pattern-service", temporal_pattern_worker)
    except Exception as e:
        logging.warning(f"[Consumer] Temporal pattern service registration failed: {e}")

    # Tool update checker
    try:
        from services.tool_update_service import tool_update_worker
        manager.register_service("tool-update-checker", tool_update_worker)
    except Exception as e:
        logging.warning(f"[Consumer] Tool update checker registration failed: {e}")

    # One-time migration: tools_disabled → tools
    try:
        from pathlib import Path as _MigPath
        _tools_dir = _MigPath(__file__).parent / "tools"
        _migrate_tools_disabled(_tools_dir, get_shared_db_service())
    except Exception as _mig_err:
        logging.warning(f"[Consumer] tools_disabled migration failed: {_mig_err}")

    # Register cron-triggered tools
    try:
        from services.tool_registry_service import ToolRegistryService
        registry = ToolRegistryService()
        for tool in registry.get_cron_tools():
            worker_func = registry.create_cron_worker(tool)
            manager.register_service(f"tool-{tool['name']}-service", worker_func)
        tool_count = len(registry.get_tool_names())
        if tool_count > 0:
            logging.info(f"[Consumer] Tool registry loaded: {tool_count} tools")
    except Exception as e:
        logging.warning(f"[Consumer] Tool cron registration failed: {e}")

    # Wire up hot-reload tool scanner
    try:
        scanner = ToolScannerThread(manager=manager, tools_dir=registry.tools_dir)
        manager._tool_scanner = scanner
        manager._tool_scanner_registry = registry
    except Exception as e:
        logging.warning(f"[Consumer] Tool scanner setup failed: {e}")

    # Bootstrap tool profiles (background thread)
    try:
        from services.tool_profile_service import ToolProfileService
        def _run_bootstrap():
            try:
                ToolProfileService().bootstrap_all()
                logging.info("[Consumer] Tool profile bootstrap complete")
            except Exception as e:
                logging.warning(f"[Consumer] Tool profile bootstrap failed: {e}")
        threading.Thread(target=_run_bootstrap, daemon=True, name="profile-bootstrap").start()
    except Exception as e:
        logging.warning(f"[Consumer] Tool profile bootstrap start failed: {e}")

    manager.run()
