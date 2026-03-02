"""
Worker Base — Base class for worker threads with state management.

Replaced multiprocessing.Process state with threading-compatible dict.
"""

import threading
import logging
from typing import Dict, Any


class WorkerBase:
    """Base class for worker threads with state management."""

    def __init__(self, worker_id: str, worker_type: str, shared_state: Dict):
        self.worker_id = worker_id
        self.worker_type = worker_type
        self.shared_state = shared_state
        self.thread_name = threading.current_thread().name
        self.job_count = 0

    def _update_shared_state(self, updates: Dict[str, Any]):
        """Merge updates into the shared state dict for this worker (best-effort)."""
        try:
            current_state = dict(self.shared_state.get(self.worker_id, {}))
            current_state.update(updates)
            self.shared_state[self.worker_id] = current_state
        except Exception:
            pass

    def register(self):
        """Register this worker in the shared state."""
        try:
            self.shared_state[self.worker_id] = {
                "thread": self.thread_name,
                "type": self.worker_type,
                "state": "idle",
                "job_count": 0
            }
            logging.info(f"[{self.worker_id}] Registered on thread {self.thread_name}")
        except Exception as e:
            logging.warning(f"[{self.worker_id}] Shared-state registration failed: {e}")

    def update_state(self, new_state: str, extra_data: Dict[str, Any] = None):
        """Update worker state in shared dictionary."""
        try:
            if self.worker_id not in self.shared_state:
                self.register()

            state_update = {
                "state": new_state,
                "job_count": self.job_count
            }
            if extra_data:
                state_update.update(extra_data)
            self._update_shared_state(state_update)
        except Exception:
            pass
        logging.info(f"[{self.worker_id}] State: {new_state}")

    def increment_job_count(self):
        """Increment the job counter."""
        self.job_count += 1
        try:
            if self.worker_id in self.shared_state:
                self._update_shared_state({"job_count": self.job_count})
        except Exception:
            pass

    def mark_off(self):
        """Mark worker as off (terminated)."""
        self.update_state("off")
        logging.info(f"[{self.worker_id}] Shutting down")
