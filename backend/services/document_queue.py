"""Shared document processing queue helper."""

import logging

logger = logging.getLogger(__name__)


def enqueue_document_processing(doc_id: str):
    """Enqueue a document for background processing via PromptQueue."""
    try:
        from services import PromptQueue
        from workers.document_worker import process_document_job
        queue = PromptQueue(queue_name="document-queue", worker_func=process_document_job)
        queue.enqueue({'doc_id': doc_id})
        logger.info(f"[DOC QUEUE] Enqueued processing for {doc_id}")
    except Exception as e:
        logger.error(f"[DOC QUEUE] Failed to enqueue processing: {e}")
