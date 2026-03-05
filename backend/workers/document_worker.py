"""
Document Worker — background thread worker for document processing jobs.

Jobs are dispatched via PromptQueue (daemon threads), not RQ/subprocess.
Timeout is enforced with threading.Thread.join() — signal.alarm() cannot
be used in non-main threads.
"""

import logging
import threading

logger = logging.getLogger(__name__)

# Hard timeout for document processing (10 minutes)
PROCESSING_TIMEOUT_SECONDS = 600


def process_document_job(job_data: dict) -> str:
    """
    Process a single document processing job.

    Args:
        job_data: dict with 'doc_id' key

    Returns:
        Status string
    """
    doc_id = job_data.get('doc_id')
    if not doc_id:
        logger.error("[DOC WORKER] Job missing doc_id")
        return "error: missing doc_id"

    result = {}

    def _process():
        try:
            # Force-reload processing modules so live code changes take effect
            # without requiring a full restart.
            import importlib
            import services.document_service as _ds_mod
            import services.document_processing_service as _dps_mod
            importlib.reload(_ds_mod)
            importlib.reload(_dps_mod)
            from services.document_processing_service import DocumentProcessingService

            logger.info(f"[DOC WORKER] Processing document {doc_id}")
            service = DocumentProcessingService()
            success = service.process_document(doc_id)
            result['status'] = 'success' if success else 'failed'
        except Exception as e:
            logger.error(f"[DOC WORKER] Error processing {doc_id}: {e}", exc_info=True)
            result['error'] = str(e)[:500]

    thread = threading.Thread(target=_process, daemon=True, name=f"doc-proc-{doc_id[:8]}")
    thread.start()
    thread.join(timeout=PROCESSING_TIMEOUT_SECONDS)

    if thread.is_alive():
        # Thread still running after timeout — mark failed and release the queue lock
        logger.error(f"[DOC WORKER] Timeout processing {doc_id} after {PROCESSING_TIMEOUT_SECONDS}s")
        try:
            from services.document_service import DocumentService
            from services.database_service import get_shared_db_service
            DocumentService(get_shared_db_service()).update_status(
                doc_id, 'failed', f'Processing timed out after {PROCESSING_TIMEOUT_SECONDS}s'
            )
        except Exception:
            pass
        return f"timeout: {doc_id}"

    if 'error' in result:
        try:
            from services.document_service import DocumentService
            from services.database_service import get_shared_db_service
            DocumentService(get_shared_db_service()).update_status(
                doc_id, 'failed', result['error']
            )
        except Exception:
            pass
        return f"error: {doc_id}: {result['error']}"

    if result.get('status') == 'success':
        logger.info(f"[DOC WORKER] Document {doc_id} processed successfully")
        return f"success: {doc_id}"
    else:
        logger.warning(f"[DOC WORKER] Document {doc_id} processing returned False")
        return f"failed: {doc_id}"


def document_purge_worker(shared_state):
    """
    Background service that purges expired documents every 6 hours.

    Runs as a registered service in run.py.
    """
    import time

    CYCLE_SECONDS = 6 * 60 * 60  # 6 hours
    INITIAL_DELAY = 300  # 5 minutes after startup

    logger.info("[DOC PURGE] Starting document purge worker")
    time.sleep(INITIAL_DELAY)

    while True:
        try:
            from services.document_service import DocumentService
            from services.database_service import get_shared_db_service

            service = DocumentService(get_shared_db_service())
            count = service.purge_expired()
            if count > 0:
                logger.info(f"[DOC PURGE] Purged {count} expired documents")

        except Exception as e:
            logger.error(f"[DOC PURGE] Purge cycle failed: {e}")

        time.sleep(CYCLE_SECONDS)
