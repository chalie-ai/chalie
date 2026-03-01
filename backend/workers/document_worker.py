"""
Document Worker — RQ-based worker for document processing jobs.

Listens for document processing jobs enqueued by the REST API upload endpoint.
Pattern: tool_worker.py (enqueued via PromptQueue → processed by RQ).
"""

import logging
import signal

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

    # Set timeout
    def _timeout_handler(signum, frame):
        raise TimeoutError(f"Document processing timed out after {PROCESSING_TIMEOUT_SECONDS}s")

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(PROCESSING_TIMEOUT_SECONDS)

    try:
        # Force-reload processing modules so live code changes take effect
        # without requiring a full consumer restart.  SimpleWorker doesn't
        # fork, so sys.modules caches the first import forever.
        import importlib
        import services.document_service as _ds_mod
        import services.document_processing_service as _dps_mod
        importlib.reload(_ds_mod)
        importlib.reload(_dps_mod)
        from services.document_processing_service import DocumentProcessingService

        logger.info(f"[DOC WORKER] Processing document {doc_id}")
        service = DocumentProcessingService()
        success = service.process_document(doc_id)

        if success:
            logger.info(f"[DOC WORKER] Document {doc_id} processed successfully")
            return f"success: {doc_id}"
        else:
            logger.warning(f"[DOC WORKER] Document {doc_id} processing returned False")
            return f"failed: {doc_id}"

    except TimeoutError as e:
        logger.error(f"[DOC WORKER] Timeout processing {doc_id}: {e}")
        try:
            from services.document_service import DocumentService
            from services.database_service import get_shared_db_service
            DocumentService(get_shared_db_service()).update_status(
                doc_id, 'failed', f'Processing timed out after {PROCESSING_TIMEOUT_SECONDS}s'
            )
        except Exception:
            pass
        return f"timeout: {doc_id}"

    except Exception as e:
        logger.error(f"[DOC WORKER] Error processing {doc_id}: {e}", exc_info=True)
        try:
            from services.document_service import DocumentService
            from services.database_service import get_shared_db_service
            DocumentService(get_shared_db_service()).update_status(
                doc_id, 'failed', str(e)[:500]
            )
        except Exception:
            pass
        return f"error: {doc_id}: {e}"

    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def document_purge_worker(shared_state):
    """
    Background service that purges expired documents every 6 hours.

    Runs as a registered service in consumer.py.
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
