# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import os
import multiprocessing
import logging

import time
import signal
import logging
from typing import Dict, List, Tuple

APP_VERSION = "0.1.0"


class WorkerManager:
    """Master supervisor for managing worker processes"""

    def __init__(self):
        # We delay manager initialization until run() to ensure
        # the 'spawn' start method is fully active.
        self.shared_state = None
        self.processes: Dict[str, multiprocessing.Process] = {}
        self.worker_definitions: List[Tuple[str, str, 'PromptQueue']] = []
        self.service_definitions: List[Tuple[str, 'worker_func']] = []
        self.running = True
        self.health_monitor = None  # Initialized in run() after imports
        self._tool_scanner = None           # ToolScannerThread, set before run()
        self._tool_scanner_registry = None  # ToolRegistryService reference

    def register_worker(self, worker_id: str, worker_type: str, queue: 'PromptQueue'):
        self.worker_definitions.append((worker_id, worker_type, queue))
        logging.info(f"[Manager] Registered worker definition: {worker_id} ({worker_type})")

    def register_service(self, worker_id: str, worker_func):
        self.service_definitions.append((worker_id, worker_func))
        logging.info(f"[Manager] Registered service definition: {worker_id}")

    def spawn_worker(self, worker_id: str, worker_type: str, queue: 'PromptQueue'):
        if worker_id in self.processes and self.processes[worker_id].is_alive():
            return

        # Ensure manager is alive before spawning
        if self.shared_state is None:
            raise RuntimeError("Manager not initialized. Call run() first.")

        # Pass the shared_state (the dict created by the Manager) to the worker
        process = queue.consume_multiprocess(worker_id, worker_type, self.shared_state)
        self.processes[worker_id] = process
        if self.health_monitor:
            self.health_monitor.record_spawn(worker_id)
        logging.info(f"[Manager] Spawned worker: {worker_id} (PID: {process.pid})")

    def spawn_service(self, worker_id: str, worker_func):
        if worker_id in self.processes and self.processes[worker_id].is_alive():
            return

        # Ensure manager is alive before spawning
        if self.shared_state is None:
            raise RuntimeError("Manager not initialized. Call run() first.")

        # Create process directly for service workers
        process = multiprocessing.Process(target=worker_func, args=(self.shared_state,))
        process.start()
        self.processes[worker_id] = process
        logging.info(f"[Manager] Spawned service: {worker_id} (PID: {process.pid})")

    def spawn_all_workers(self):
        logging.info("[Manager] Spawning all workers...")
        for worker_id, worker_type, queue in self.worker_definitions:
            self.spawn_worker(worker_id, worker_type, queue)

        logging.info("[Manager] Spawning all services...")
        for worker_id, worker_func in self.service_definitions:
            try:
                self.spawn_service(worker_id, worker_func)
            except Exception as e:
                logging.error(f"[Manager] Failed to spawn service '{worker_id}': {e}")

    def check_worker_health(self):
        # Check RQ workers with comprehensive health monitoring
        for worker_id, worker_type, queue in self.worker_definitions:
            try:
                process = self.processes.get(worker_id)

                # Get queue name from queue object
                queue_name = queue.queue_name if hasattr(queue, 'queue_name') else 'unknown-queue'

                # Comprehensive health check (process + Redis heartbeat + job activity)
                if self.health_monitor:
                    is_healthy, reason = self.health_monitor.comprehensive_health_check(
                        worker_id, process, queue_name
                    )
                else:
                    # Fallback to simple check if health monitor not initialized
                    is_healthy = process and process.is_alive()
                    reason = "ok" if is_healthy else "process_dead"

                if not is_healthy:
                    logging.warning(f"[Manager] Worker {worker_id} is unhealthy (reason: {reason}). Restarting...")
                    if self.health_monitor:
                        self.health_monitor.record_restart(worker_id)
                    self.spawn_worker(worker_id, worker_type, queue)
            except Exception as e:
                logging.error(f"[Manager] Health check failed for worker {worker_id}: {e}")

        # Check service workers (simple process check only)
        for worker_id, worker_func in self.service_definitions:
            try:
                process = self.processes.get(worker_id)

                # Simple health check: if process is alive, it's healthy
                if not process or not process.is_alive():
                    logging.warning(f"[Manager] Service {worker_id} is dead. Restarting...")
                    if self.health_monitor:
                        self.health_monitor.record_restart(worker_id)
                    self.spawn_service(worker_id, worker_func)
            except Exception as e:
                logging.error(f"[Manager] Health check failed for service {worker_id}: {e}")

    def log_health_stats(self):
        """Log health statistics for all workers (called periodically)."""
        if not self.health_monitor:
            return

        stats = self.health_monitor.get_all_stats()
        restart_counts = stats['restart_counts']

        if restart_counts:
            restart_summary = ", ".join([f"{wid}: {count}" for wid, count in restart_counts.items()])
            logging.info(f"[HealthMonitor] Worker restarts: {restart_summary}")

    def shutdown_all(self):
        logging.info("\n[Manager] Initiating graceful shutdown...")
        self.running = False

        # Log final health stats
        self.log_health_stats()

        for worker_id, process in self.processes.items():
            if process.is_alive():
                process.terminate()
        logging.info("[Manager] All workers and services stopped")

    def run(self):
        # 2. INITIALIZE MANAGER HERE (inside the protected run)
        # Use context manager to ensure proper cleanup
        with multiprocessing.Manager() as manager:
            self.shared_state = manager.dict()

            signal.signal(signal.SIGINT, lambda sig, frame: self.shutdown_all())
            signal.signal(signal.SIGTERM, lambda sig, frame: self.shutdown_all())

            logging.info("[Manager] Starting Worker Manager")
            self.spawn_all_workers()

            # Start hot-reload tool scanner (daemon thread, in-process)
            if self._tool_scanner is not None:
                self._tool_scanner.start(self._tool_scanner_registry)

            try:
                health_check_counter = 0
                while self.running:
                    time.sleep(5)
                    self.check_worker_health()

                    # Log health stats every 5 minutes (60 health checks)
                    health_check_counter += 1
                    if health_check_counter >= 60:
                        self.log_health_stats()
                        health_check_counter = 0

            except KeyboardInterrupt:
                pass
            finally:
                self.shutdown_all()


class ToolScannerThread:
    """
    Daemon thread that scans backend/tools/ every TOOL_SCANNER_INTERVAL_SECONDS (default 30)
    for new tool directories and registers them without a restart.

    Runs in the main process to allow direct calls to manager.spawn_service().
    """

    DEFAULT_INTERVAL = 30

    def __init__(self, manager: WorkerManager, tools_dir):
        self._manager = manager
        self._tools_dir = tools_dir
        self._interval = int(os.environ.get("TOOL_SCANNER_INTERVAL_SECONDS", self.DEFAULT_INTERVAL))
        self._registry = None

    def start(self, registry):
        self._registry = registry
        registry.set_on_tool_registered(self._on_tool_registered)
        import threading
        t = threading.Thread(target=self._scan_loop, name="tool-scanner", daemon=True)
        t.start()
        logging.info(f"[ToolScanner] Started (interval={self._interval}s)")

    def _on_tool_registered(self, tool_name: str):
        """Callback fired by _build_worker after a successful build. Spawns cron worker if needed."""
        tool = self._registry.tools.get(tool_name)
        if not tool:
            return
        trigger = tool["manifest"].get("trigger", {})
        if trigger.get("type") != "cron":
            return

        worker_id = f"tool-{tool_name}-service"
        existing = self._manager.processes.get(worker_id)
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
        import time as _time
        _time.sleep(self._interval)  # Initial delay — startup build already ran
        while True:
            try:
                self._scan_once()
            except Exception as e:
                logging.error(f"[ToolScanner] Scan error: {e}")
            _time.sleep(self._interval)

    def _scan_once(self):
        import json as _json
        from pathlib import Path
        if not self._tools_dir.exists():
            return

        known = set(self._registry.tools.keys())
        building = {n for n, s in self._registry.get_all_build_statuses().items()
                    if s.get("status") == "building"}
        locked = set(self._registry._install_locks)

        for entry in sorted(self._tools_dir.iterdir()):
            if not entry.is_dir() or entry.name.startswith(("_", ".")):
                continue
            if not (entry / "manifest.json").exists() or not (entry / "Dockerfile").exists():
                continue
            try:
                with open(entry / "manifest.json") as f:
                    tool_name = _json.load(f).get("name", "").strip()
            except Exception:
                continue
            if not tool_name or tool_name in known or tool_name in building or tool_name in locked:
                continue
            logging.info(f"[ToolScanner] Discovered new tool '{tool_name}', starting build")
            try:
                self._registry.register_tool_async(entry)
            except Exception as e:
                logging.warning(f"[ToolScanner] Build start failed for '{tool_name}': {e}")


if __name__ == "__main__":
    # 3. FORCE SPAWN METHOD
    # This must be the very first thing called inside the name-main guard.
    try:
        multiprocessing.set_start_method('spawn', force=True)
        logging.info("[System] Multiprocessing start method set to 'spawn'")
    except RuntimeError:
        pass

    logging.basicConfig(level=logging.INFO)

    # 4. DEFERRED IMPORTS
    # We import these HERE to ensure no library (like Transformers) initializes
    # global state before the 'spawn' method is set.
    from services import PromptQueue, DatabaseService, SchemaService
    from workers import digest_worker, memory_chunker_worker, episodic_memory_worker, semantic_consolidation_worker, rest_api_worker, tool_worker
    from services.config_service import ConfigService
    from services.idle_consolidation_service import idle_consolidation_process
    from services.decay_engine_service import decay_engine_worker
    from services.topic_stability_regulator_service import topic_stability_regulator_worker
    from services.cognitive_drift_engine import cognitive_drift_worker
    from services.routing_stability_regulator_service import routing_stability_regulator_worker
    from services.routing_reflection_service import routing_reflection_worker
    from services.worker_health_monitor import WorkerHealthMonitor
    from services.experience_assimilation_service import experience_assimilation_worker
    from services.thread_expiry_service import thread_expiry_worker
    from services.scheduler_service import scheduler_worker
    from services.autobiography_service import autobiography_synthesis_worker

    # 5. RESOLVE HOSTNAMES (BEFORE FORKING)
    # This prevents DNS lookup segfaults in child processes on macOS
    ConfigService.resolve_hostnames()

    # 5.1. ENSURE ENCRYPTION KEY (REQUIRED FOR SENSITIVE DATABASE COLUMNS)
    # Auto-generates if not present, stores in .key with restrictive permissions (0600)
    from services.encryption_key_service import get_encryption_key
    get_encryption_key()

    # 5.4. PRELOAD EMBEDDING MODEL AND SERVICE SINGLETON
    # Load embedding model and initialize singleton before forking
    # This prevents HuggingFace requests in child processes
    try:
        logging.info("[System] Preloading embedding model and service singleton...")
        from services.embedding_service import get_embedding_service
        get_embedding_service()  # Lazy-loads model and creates singleton
        logging.info("[System] ✓ Embedding model and service singleton ready")
    except Exception as e:
        logging.warning(f"[System] Embedding model preload failed: {e}")
        logging.warning("[System] Continuing without preload (will load on first embedding request)")

    # 5.5. INITIALIZE DATABASE
    # Wait for postgres to be ready, then create database and schema if needed.
    # This is FATAL — the app cannot function without a database.
    import time
    from services.database_service import get_merged_db_config

    db_config = get_merged_db_config()
    episodic_config = ConfigService.resolve_agent_config("episodic-memory")

    # Create database service for admin connection
    admin_config = db_config.copy()
    admin_db = admin_config.pop('database')
    admin_config['database'] = 'postgres'

    # Wait for postgres to accept connections (up to 30s)
    max_retries = 15
    for attempt in range(1, max_retries + 1):
        try:
            admin_database_service = DatabaseService(admin_config)
            with admin_database_service.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
            logging.info(f"[DB] Postgres is ready (attempt {attempt}/{max_retries})")
            break
        except Exception as e:
            if attempt == max_retries:
                logging.error(f"[DB] Postgres not reachable after {max_retries} attempts: {e}")
                raise SystemExit(1)
            logging.warning(f"[DB] Waiting for postgres (attempt {attempt}/{max_retries}): {e}")
            time.sleep(2)

    embedding_dimensions = episodic_config.get('embedding_dimensions', 768)
    schema_service = SchemaService(admin_database_service, embedding_dimensions)

    # Check if database exists
    if not schema_service.database_exists(admin_db):
        logging.info(f"Database '{admin_db}' does not exist, creating...")
        schema_service.create_database(admin_db)
        logging.info(f"✓ Database '{admin_db}' created")
    else:
        logging.info(f"✓ Database '{admin_db}' exists")

    # Connect to the actual database
    database_service = DatabaseService(db_config)
    schema_service = SchemaService(database_service, embedding_dimensions)

    # Initialize schema (idempotent)
    current_version = schema_service.schema_version()
    if current_version == 0:
        logging.info("Initializing episodic memory schema...")
        schema_service.initialize_schema()
        logging.info("✓ Schema initialized")
    else:
        logging.info(f"✓ Schema up to date (version {current_version})")

    # Run pending migrations
    logging.info("Checking for pending database migrations...")
    database_service.run_pending_migrations()

    # Initialize API key in settings table (auto-generate if not present)
    try:
        from services.settings_service import SettingsService
        settings_service = SettingsService(database_service)
        api_key = settings_service.get_api_key_or_generate()
        logging.info(f"[Settings] API key initialized (key: ...{api_key[-8:]})")
    except Exception as e:
        logging.warning(f"Settings initialization failed: {e}")

    database_service.close_pool()
    admin_database_service.close_pool()

    # Initialize and run
    manager = WorkerManager()

    # Initialize enhanced health monitor
    manager.health_monitor = WorkerHealthMonitor(
        heartbeat_timeout_seconds=900,  # 15min — must far exceed RQ SimpleWorker's BLPOP timeout (~405s)
        activity_timeout_seconds=600    # 10 minutes for hung worker detection
    )
    logging.info("[Manager] Enhanced health monitoring initialized")

    digest_queue = PromptQueue(queue_name="prompt-queue", worker_func=digest_worker)
    manager.register_worker(
        worker_id="digest-worker-1",
        worker_type="idle-busy",
        queue=digest_queue
    )

    # Register memory chunker worker
    memory_queue = PromptQueue(queue_name="memory-chunker-queue", worker_func=memory_chunker_worker)
    manager.register_worker(
        worker_id="memory-chunker-worker-1",
        worker_type="idle-busy",
        queue=memory_queue
    )

    # Register episodic memory worker
    episodic_queue = PromptQueue(queue_name="episodic-memory-queue", worker_func=episodic_memory_worker)
    manager.register_worker(
        worker_id="episodic-memory-worker-1",
        worker_type="idle-busy",
        queue=episodic_queue
    )

    # Register semantic consolidation worker
    semantic_queue = PromptQueue(queue_name="semantic_consolidation_queue", worker_func=semantic_consolidation_worker)
    manager.register_worker(
        worker_id="semantic-consolidation-worker-1",
        worker_type="idle-busy",
        queue=semantic_queue
    )

    # Register tool worker (background ACT loop processing)
    tool_queue = PromptQueue(queue_name="tool-queue", worker_func=tool_worker)
    manager.register_worker(
        worker_id="tool-worker-1",
        worker_type="idle-busy",
        queue=tool_queue
    )

    # Register idle consolidation service (STORY-12)
    manager.register_service(
        worker_id="idle-consolidation-service",
        worker_func=idle_consolidation_process
    )

    # Register decay engine service
    manager.register_service(
        worker_id="decay-engine-service",
        worker_func=decay_engine_worker
    )

    # Register homeostatic stability regulator service (runs every 24h)
    manager.register_service(
        worker_id="topic-stability-regulator-service",
        worker_func=topic_stability_regulator_worker
    )

    # Register cognitive drift engine (spontaneous thought during idle)
    manager.register_service(
        worker_id="cognitive-drift-engine",
        worker_func=cognitive_drift_worker
    )

    # Register routing stability regulator (24h cycle, single authority for weight mutation)
    manager.register_service(
        worker_id="routing-stability-regulator-service",
        worker_func=routing_stability_regulator_worker
    )

    # Register routing reflection service (idle-time peer review of routing decisions)
    manager.register_service(
        worker_id="routing-reflection-service",
        worker_func=routing_reflection_worker
    )

    # Register REST API service
    manager.register_service(
        worker_id="rest-api-worker-1",
        worker_func=rest_api_worker
    )

    # Register experience assimilation service (tool output → episodic memory)
    manager.register_service(
        worker_id="experience-assimilation-service",
        worker_func=experience_assimilation_worker
    )

    # Register thread expiry service (5-minute cycle, expires stale threads)
    manager.register_service(
        worker_id="thread-expiry-service",
        worker_func=thread_expiry_worker
    )

    # Register scheduler service (60s poll cycle, fires due reminders/tasks)
    manager.register_service(
        worker_id="scheduler-service",
        worker_func=scheduler_worker
    )

    # Register autobiography synthesis service (6h cycle, synthesizes user narrative)
    manager.register_service(
        worker_id="autobiography-synthesis-service",
        worker_func=autobiography_synthesis_worker
    )

    # Register triage calibration service (24h cycle, computes correctness scores)
    try:
        from services.triage_calibration_service import triage_calibration_worker
        manager.register_service(
            worker_id="triage-calibration-service",
            worker_func=triage_calibration_worker
        )
    except Exception as e:
        logging.warning(f"[Consumer] Triage calibration service registration failed: {e}")

    # Register profile enrichment service (6h cycle, enriches tool profiles)
    try:
        from services.profile_enrichment_service import profile_enrichment_worker
        manager.register_service(
            worker_id="profile-enrichment-service",
            worker_func=profile_enrichment_worker
        )
    except Exception as e:
        logging.warning(f"[Consumer] Profile enrichment service registration failed: {e}")

    # Register cron-triggered tools as background services
    try:
        from services.tool_registry_service import ToolRegistryService
        registry = ToolRegistryService()
        for tool in registry.get_cron_tools():
            worker_func = registry.create_cron_worker(tool)
            manager.register_service(
                worker_id=f"tool-{tool['name']}-service",
                worker_func=worker_func
            )
        tool_count = len(registry.get_tool_names())
        if tool_count > 0:
            logging.info(f"[Consumer] Tool registry loaded: {tool_count} tools")
    except Exception as e:
        logging.warning(f"[Consumer] Tool cron registration failed: {e}")

    # Wire up hot-reload tool scanner
    try:
        from pathlib import Path
        scanner = ToolScannerThread(manager=manager, tools_dir=registry.tools_dir)
        manager._tool_scanner = scanner
        manager._tool_scanner_registry = registry
    except Exception as e:
        logging.warning(f"[Consumer] Tool scanner setup failed: {e}")

    # Bootstrap tool capability profiles on startup
    try:
        from services.tool_profile_service import ToolProfileService
        profile_svc = ToolProfileService()
        profile_svc.bootstrap_all()
        logging.info("[Consumer] Tool profile bootstrap complete")
    except Exception as e:
        logging.warning(f"[Consumer] Tool profile bootstrap failed: {e}")

    manager.run()