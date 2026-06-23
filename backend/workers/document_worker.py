"""
Document Worker — background purge worker for expired documents.
"""

import logging

logger = logging.getLogger(__name__)


def document_purge_worker() -> None:
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
