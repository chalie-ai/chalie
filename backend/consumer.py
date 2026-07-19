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
import signal
import threading
import time
from typing import Callable, Dict, List, Tuple, cast


def _read_version() -> str:
    try:
        from services.file_mapper_service import FileMapperService
        return FileMapperService.get_version_path().read_text().strip()
    except Exception:
        return "0.0.0"

APP_VERSION = _read_version()


def _thread_excepthook(args: threading.ExceptHookArgs) -> None:
    """Threads die silently by default — log uncaught exceptions globally."""
    logging.error(
        f"[ThreadException] Uncaught exception in thread '{cast(threading.Thread, args.thread).name}': "
        f"{args.exc_type.__name__}: {args.exc_value}",
        exc_info=(args.exc_type, cast("BaseException", args.exc_value), args.exc_traceback)
    )


# Install global thread exception handler
threading.excepthook = _thread_excepthook


class WorkerManager:
    """Master supervisor for managing worker threads."""

    # Per-worker respawn backoff + circuit-breaker (see check_health).
    _RETRY_BASE_DELAY_S = 5.0     # first backoff delay; doubles each consecutive death
    _RETRY_MAX_DELAY_S = 300.0    # cap backoff at 5 minutes
    _MAX_FAILURES = 6             # consecutive deaths before the circuit-breaker trips

    def __init__(self) -> None:
        self.threads: Dict[str, threading.Thread] = {}
        self.service_definitions: List[Tuple[str, Callable[[], None]]] = []
        self.running = True
        self._failure_counts: Dict[str, int] = {}
        self._next_retry_at: Dict[str, float] = {}
        self._tripped: set[str] = set()

    def register_service(self, worker_id: str, worker_func: Callable[[], None]) -> None:
        self.service_definitions.append((worker_id, worker_func))
        logging.info(f"[Manager] Registered service definition: {worker_id}")

    def spawn_service(self, worker_id: str, worker_func: Callable[[], None]) -> None:
        """Skip if a thread with the given worker_id is already alive."""
        if worker_id in self.threads and self.threads[worker_id].is_alive():
            return

        def _run() -> None:
            try:
                worker_func()
            except Exception:
                logging.exception(f"[Manager] Service {worker_id} crashed")

        t = threading.Thread(target=_run, daemon=True, name=worker_id)
        t.start()
        self.threads[worker_id] = t
        logging.info(f"[Manager] Spawned service: {worker_id} (thread)")

    def spawn_all_services(self) -> None:
        """Individual spawn failures are logged but do not abort subsequent
        spawns."""
        logging.info("[Manager] Spawning all services...")
        for worker_id, worker_func in self.service_definitions:
            try:
                self.spawn_service(worker_id, worker_func)
            except Exception as e:
                logging.error(f"[Manager] Failed to spawn service '{worker_id}': {e}")

    def check_health(self) -> None:
        """Check service thread health and restart dead threads.

        A dead worker is only respawned once its per-worker backoff window has
        elapsed; each consecutive death doubles the delay (capped) and, after
        ``_MAX_FAILURES`` consecutive deaths, the circuit-breaker trips and the
        worker is abandoned for the lifetime of the process — preventing a
        deterministically-fatal worker from looping forever (log flooding, CPU
        churn, repeated import side effects). Failure counters reset the moment
        a respawned worker stays alive past one health-check tick.
        """
        now = time.monotonic()
        for worker_id, worker_func in self.service_definitions:
            try:
                t = self.threads.get(worker_id)
                if t and t.is_alive():
                    # A worker that survived since its last respawn clears its failures.
                    self._failure_counts.pop(worker_id, None)
                    self._next_retry_at.pop(worker_id, None)
                    continue

                if worker_id in self._tripped:
                    continue

                due = self._next_retry_at.get(worker_id)
                if due is not None and now < due:
                    continue

                failures = self._failure_counts.get(worker_id, 0)
                if failures >= self._MAX_FAILURES:
                    logging.error(
                        f"[Manager] Circuit-breaker tripped for {worker_id} after "
                        f"{failures} consecutive deaths; abandoning respawns"
                    )
                    self._tripped.add(worker_id)
                    continue

                delay = min(self._RETRY_BASE_DELAY_S * (2 ** failures), self._RETRY_MAX_DELAY_S)
                logging.warning(
                    f"[Manager] Service {worker_id} is dead. Restarting "
                    f"(attempt {failures + 1}/{self._MAX_FAILURES}, next backoff {delay:.0f}s)..."
                )
                self.spawn_service(worker_id, worker_func)
                self._failure_counts[worker_id] = failures + 1
                self._next_retry_at[worker_id] = now + delay
            except Exception as e:
                logging.error(f"[Manager] Health check failed for service {worker_id}: {e}")


    def shutdown_all(self) -> None:
        """Daemon threads are terminated automatically when the main thread
        finishes."""
        logging.info("\n[Manager] Initiating graceful shutdown...")
        self.running = False
        # Daemon threads will be killed when main thread exits
        logging.info("[Manager] All services stopped")

    def run(self) -> None:
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
