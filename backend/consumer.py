# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Consumer — master supervisor for the single-process architecture.

All workers run as daemon threads in one Python process. SQLite replaces
PostgreSQL, MemoryStore replaces Redis.
"""

import logging
import time
import signal
import threading
from typing import Dict, List, Tuple

from utils.logger import Logger

def _read_version():
    try:
        from services.file_mapper_service import FileMapperService
        return FileMapperService.get_version_path().read_text().strip()
    except Exception:
        return "0.0.0"

APP_VERSION = _read_version()


def _thread_excepthook(args):
    """Threads die silently by default — log uncaught exceptions globally."""
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
        self.threads: Dict[str, threading.Thread] = {}
        self.service_definitions: List[Tuple[str, callable]] = []
        self.running = True

    def register_service(self, worker_id: str, worker_func):
        self.service_definitions.append((worker_id, worker_func))
        logging.info(f"[Manager] Registered service definition: {worker_id}")

    def spawn_service(self, worker_id: str, worker_func):
        """Skip if a thread with the given worker_id is already alive."""
        if worker_id in self.threads and self.threads[worker_id].is_alive():
            return

        def _run():
            try:
                worker_func()
            except Exception:
                logging.exception(f"[Manager] Service {worker_id} crashed")

        t = threading.Thread(target=_run, daemon=True, name=worker_id)
        t.start()
        self.threads[worker_id] = t
        logging.info(f"[Manager] Spawned service: {worker_id} (thread)")

    def spawn_all_services(self):
        """Individual spawn failures are logged but do not abort subsequent
        spawns."""
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


    def shutdown_all(self):
        """Daemon threads are terminated automatically when the main thread
        finishes."""
        logging.info("\n[Manager] Initiating graceful shutdown...")
        self.running = False
        # Daemon threads will be killed when main thread exits
        logging.info("[Manager] All services stopped")

    def run(self):
        """Installs SIGINT/SIGTERM handlers, spawns all services, then polls
        every 5 s to restart dead threads. Logs a summary every 5 minutes.
        Blocks until shutdown_all() or KeyboardInterrupt."""
        signal.signal(signal.SIGINT, lambda _sig, _frame: self.shutdown_all())
        signal.signal(signal.SIGTERM, lambda _sig, _frame: self.shutdown_all())

        logging.info("[Manager] Starting Worker Manager (single-process, threaded)")
        self.spawn_all_services()

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




if __name__ == "__main__":
    Logger.start()

    # Deferred imports
    from workers import rest_api_worker
    from services.scheduler_service import scheduler_worker
    from workers.document_worker import document_purge_worker

    # Preload embedding model singleton
    try:
        logging.info("[System] Preloading embedding model...")
        from services.embedding_service import get_embedding_service
        get_embedding_service()
        logging.info("[System] Embedding model ready")
    except Exception as e:
        logging.warning(f"[System] Embedding model preload failed: {e}")

    # Initialize SQLite database — declarative convergence
    from services.database_service import get_shared_db_service
    from services.schema_convergence_service import SchemaConvergenceService

    database_service = get_shared_db_service()
    convergence = SchemaConvergenceService(database_service)
    convergence.converge()
    # Separate deterministic value backfill — convergence applies only static
    # column DEFAULTs, never derived values (last_relevant_at, valid_from, etc.).
    convergence.backfill_redesign_columns()

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
    manager.register_service("rest-api-worker-1", rest_api_worker)
    manager.register_service("scheduler-service", scheduler_worker)
    manager.register_service("document-purge-service", document_purge_worker)

    manager.run()
