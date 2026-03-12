"""
Prompt Queue — Lightweight thread dispatcher.

Spawns background threads for job execution. Serialization per queue name
ensures one job at a time per queue.
"""

import logging
import threading

logger = logging.getLogger(__name__)


class PromptQueue:
    """Lightweight thread dispatcher for serialized background job execution."""

    # Class-level registry: shared locks per queue name for serialization
    _locks: dict = {}
    _lock_guard = threading.Lock()

    def __init__(self, queue_name: str = None, worker_func=None):
        """Initialise the queue and acquire (or create) its serialization lock.

        A class-level lock registry ensures that all ``PromptQueue`` instances
        sharing the same *queue_name* contend on a single
        :class:`threading.Lock`, so only one job per logical queue runs at a
        time across the process.

        Args:
            queue_name: Logical queue identifier used for lock sharing and log
                messages.  Defaults to ``"prompt-queue"`` when *None*.
            worker_func: Callable invoked by background threads dispatched via
                :meth:`enqueue`.  May be ``None`` if the queue is used purely
                for lock sharing; calling :meth:`enqueue` without a worker
                raises :class:`ValueError`.
        """
        self._queue_name = queue_name or "prompt-queue"
        self._worker_func = worker_func

        # Ensure one lock per queue name (so all PromptQueue instances
        # for "prompt-queue" share the same serialization lock)
        with self._lock_guard:
            if self._queue_name not in self._locks:
                self._locks[self._queue_name] = threading.Lock()
        self._lock = self._locks[self._queue_name]

    @property
    def queue_name(self) -> str:
        """The logical name of this queue, used for lock sharing and logging.

        Returns:
            The queue name string passed at construction, or ``"prompt-queue"``
            if none was provided.
        """
        return self._queue_name

    def enqueue(self, *args, **kwargs):
        """Run worker_func in a background thread, serialized per queue name."""
        if not self._worker_func:
            raise ValueError("No worker function configured for this queue")

        def _run():
            with self._lock:  # ensures one job at a time per queue
                try:
                    self._worker_func(*args, **kwargs)
                except Exception:
                    logger.exception(f"[{self._queue_name}] Job failed")

        t = threading.Thread(target=_run, daemon=True, name=f"{self._queue_name}-job")
        t.start()
        return t  # callers that need the job reference get the thread

    def consume(self, burst=False):
        """No-op — threads are dispatched on enqueue(), no separate consumer needed."""
        pass

    def consume_multiprocess(self, worker_id: str, worker_type: str, shared_state):
        """No-op — PromptQueue dispatches threads on enqueue(). Returns None."""
        return None


def enqueue_episodic_memory(topic_data: dict):
    """
    Enqueue episodic memory generation job.

    Args:
        topic_data: Dict with 'topic' key.
    """
    from workers.episodic_memory_worker import episodic_memory_worker

    job_data = {
        'topic': topic_data['topic']
    }
    if topic_data.get('thread_id'):
        job_data['thread_id'] = topic_data['thread_id']
    if topic_data.get('retry_count'):
        job_data['retry_count'] = topic_data['retry_count']

    queue = PromptQueue(queue_name="episodic-memory-queue", worker_func=episodic_memory_worker)
    queue.enqueue(job_data)
    logger.info(f"Enqueued episodic memory job for topic '{topic_data['topic']}'")
