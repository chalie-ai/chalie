import multiprocessing
from rq import Queue, SimpleWorker

import logging
from .config_service import ConfigService
from .redis_client import RedisClientService
from .worker_base import WorkerBase

class PromptQueue:

    def __init__(self, queue_name=None, worker_func=None):
        self._config = ConfigService.connections().get("redis", {})

        # Use RedisClientService as single source of truth for Redis connections
        self._redis = RedisClientService.create_connection(decode_responses=False)

        self._queue_name = queue_name or self._config.get("topics", {}).get("prompt_queue", "prompt-queue")
        self._worker_func = worker_func

        # Get timeout from queue config, default to 600s (10 minutes)
        queue_configs = self._config.get("queues", {})
        queue_config = queue_configs.get(self._queue_name.replace("-", "_") + "_queue", {})
        timeout = queue_config.get("timeout", 600)

        self._queue = Queue(
            name=self._queue_name,
            connection=self._redis,
            default_timeout=timeout
        )

    @property
    def queue_name(self):
        return self._queue_name

    def enqueue(self, *args, **kwargs):
        if not self._worker_func:
            raise ValueError("No worker function configured for this queue")
        return self._queue.enqueue(self._worker_func, *args, **kwargs)

    def consume(self, burst=False):
        worker = SimpleWorker([self._queue], connection=self._redis)
        print(f"Starting worker for queue '{self._queue_name}'")
        worker.work(burst=burst)

    @staticmethod
    def _worker_process(config, queue_name, worker_id, worker_type, shared_state):
        """Static method to run in child process - creates its own Redis connection
        Runs continuously for idle-busy workers, processes one job at a time"""

        logging.basicConfig(level=logging.INFO)

        # Pre-resolve hostnames in this worker process (spawn doesn't inherit globals)
        from .config_service import ConfigService
        ConfigService.resolve_hostnames()

        # Initialize worker base for state management
        worker_base = WorkerBase(worker_id, worker_type, shared_state)
        worker_base.register()
        worker_base.update_state("idle")

        # Use RedisClientService as single source of truth for Redis connections
        redis_conn = RedisClientService.create_connection(decode_responses=False)

        # Get timeout from queue config
        queue_configs = config.get("queues", {})
        queue_config = queue_configs.get(queue_name.replace("-", "_") + "_queue", {})
        timeout = queue_config.get("timeout", 600)

        queue = Queue(name=queue_name, connection=redis_conn, default_timeout=timeout)

    # removed nested StatefulWorker definition
        class StatefulWorker(SimpleWorker):
            def perform_job(self, job, queue):
                """Override to add state management around job execution"""
                worker_base.update_state("busy", {"current_job": job.id})
                try:
                    result = super().perform_job(job, queue)
                    worker_base.increment_job_count()
                    return result
                finally:
                    worker_base.update_state("idle")

        worker = StatefulWorker([queue], connection=redis_conn)
        logging.info(f"[{worker_id}] Starting worker for queue '{queue_name}' (PID: {multiprocessing.current_process().pid})")

        # CRITICAL: Clear abandoned jobs from previous crashes before starting work
        # This prevents workers from hanging on stale jobs that will never complete
        try:
            from rq.registry import StartedJobRegistry
            started_registry = StartedJobRegistry(queue_name, connection=redis_conn)
            abandoned_count = len(started_registry)
            if abandoned_count > 0:
                logging.warning(f"[{worker_id}] Found {abandoned_count} abandoned jobs in StartedJobRegistry, cleaning up...")
                started_registry.cleanup()
                logging.info(f"[{worker_id}] Abandoned jobs cleared")
        except Exception as e:
            logging.error(f"[{worker_id}] Failed to clear abandoned jobs: {e}")

        try:
            # Run continuously (burst=False) for idle-busy workers
            # This blocks and waits for jobs, processing one at a time
            worker.work(burst=False)
        except Exception as e:
            logging.error(f"[{worker_id}] Worker error: {e}")
        finally:
            worker_base.mark_off()

    def consume_multiprocess(self, worker_id: str, worker_type: str, shared_state):
        """Spawn a worker process with state management

        Args:
            worker_id: Unique identifier for this worker
            worker_type: Type of worker ("idle-busy" or "busy-off")
            shared_state: Shared dictionary for state communication
        """
        process = multiprocessing.Process(
            target=self._worker_process,
            args=(self._config, self._queue_name, worker_id, worker_type, shared_state),
            daemon=False,
            name=worker_id
        )
        process.start()
        return process


def enqueue_episodic_memory(topic_data: dict):
    """
    Enqueue episodic memory generation job.

    Args:
        topic_data: Dict with 'topic' key. Worker will load exchanges from conversation file.
    """
    config = ConfigService.connections().get("redis", {})
    queue_name = config.get("topics", {}).get("episodic_memory", "episodic-memory-queue")

    # Get timeout from config
    queue_configs = config.get("queues", {})
    queue_config = queue_configs.get("episodic_memory_queue", {})
    timeout = queue_config.get("timeout", 600)

    queue = Queue(queue_name, connection=RedisClientService.create_connection(decode_responses=False), default_timeout=timeout)

    job_data = {
        'topic': topic_data['topic']
    }
    if topic_data.get('thread_id'):
        job_data['thread_id'] = topic_data['thread_id']

    queue.enqueue('workers.episodic_memory_worker.episodic_memory_worker', job_data)
    logging.info(f"Enqueued episodic memory job for topic '{topic_data['topic']}'")