"""Document Processing Queue — concurrent background document processing.

Replaces the old serial PromptQueue approach with a proper queue.Queue + worker pool.
- FIFO ordering (queue.Queue)
- Configurable concurrency (WORKER_COUNT workers process in parallel)
- Deduplication (won't re-enqueue a doc_id already queued or being processed)
- Hard timeout per document (PROCESSING_TIMEOUT seconds)
"""

import logging
import queue
import threading

logger = logging.getLogger(__name__)

WORKER_COUNT = 3
PROCESSING_TIMEOUT = 600  # 10 minutes per document

_queue = queue.Queue(maxsize=500)
_active = set()           # doc_ids currently queued or being processed
_active_lock = threading.Lock()
_workers_started = False
_start_lock = threading.Lock()


def enqueue_document_processing(doc_id: str):
    """Add a document to the background processing queue, deduplicating in-flight IDs.

    Args:
        doc_id: Document ID to enqueue.  Silently no-ops if the same ID is
            already queued or currently being processed.
    """
    with _active_lock:
        if doc_id in _active:
            logger.debug(f"[DOC QUEUE] {doc_id} already queued/processing, skipping")
            return
        _active.add(doc_id)

    _ensure_workers()

    try:
        _queue.put_nowait(doc_id)
        logger.info(f"[DOC QUEUE] Enqueued {doc_id} (queue depth: ~{_queue.qsize()})")
    except queue.Full:
        with _active_lock:
            _active.discard(doc_id)
        logger.warning(f"[DOC QUEUE] Queue full, cannot enqueue {doc_id}")


def _ensure_workers():
    """Lazily start worker threads on first enqueue."""
    global _workers_started
    with _start_lock:
        if _workers_started:
            return
        _workers_started = True
        for i in range(WORKER_COUNT):
            t = threading.Thread(
                target=_worker_loop,
                daemon=True,
                name=f"doc-worker-{i}",
            )
            t.start()
        logger.info(f"[DOC QUEUE] Started {WORKER_COUNT} document processing workers")


def _worker_loop():
    """Worker loop: pull doc_ids from queue and process them."""
    while True:
        doc_id = _queue.get()
        try:
            _process_with_timeout(doc_id)
        except Exception as e:
            logger.error(f"[DOC WORKER] Unhandled error for {doc_id}: {e}")
        finally:
            with _active_lock:
                _active.discard(doc_id)
            _queue.task_done()


def _process_with_timeout(doc_id: str):
    """Process a single document inside a daemon thread with a hard timeout.

    Spawns a daemon thread that calls
    :meth:`~services.document_processing_service.DocumentProcessingService.process_document`.
    Blocks until the thread completes or ``PROCESSING_TIMEOUT`` seconds elapse,
    then marks the document as failed if the thread is still alive.

    Args:
        doc_id: Document ID to process.
    """
    result = {}

    def _run():
        try:
            from services.document_processing_service import DocumentProcessingService
            service = DocumentProcessingService()
            result['ok'] = service.process_document(doc_id)
        except Exception as e:
            logger.error(f"[DOC WORKER] Error processing {doc_id}: {e}", exc_info=True)
            result['error'] = str(e)[:500]

    thread = threading.Thread(target=_run, daemon=True, name=f"doc-proc-{doc_id[:8]}")
    thread.start()
    thread.join(timeout=PROCESSING_TIMEOUT)

    if thread.is_alive():
        logger.error(f"[DOC WORKER] Timeout for {doc_id} after {PROCESSING_TIMEOUT}s")
        _mark_failed(doc_id, f'Processing timed out after {PROCESSING_TIMEOUT}s')
        return

    if result.get('error'):
        _mark_failed(doc_id, result['error'])
    elif result.get('ok'):
        logger.info(f"[DOC WORKER] {doc_id} processed successfully")
    else:
        logger.warning(f"[DOC WORKER] {doc_id} processing returned False")


def _mark_failed(doc_id: str, error: str):
    """Mark a document as failed in the database, best-effort.

    Args:
        doc_id: Document ID to mark as failed.
        error: Error message string to store (truncated by the caller).
    """
    try:
        from services.document_service import DocumentService
        from services.database_service import get_shared_db_service
        DocumentService(get_shared_db_service()).update_status(doc_id, 'failed', error)
    except Exception:
        pass
