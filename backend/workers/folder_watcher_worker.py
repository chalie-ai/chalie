"""
Folder Watcher Worker — Background daemon thread that scans watched folders for changes.

Runs every CHECK_INTERVAL seconds, checking which folders are due for a scan
(based on their individual scan_interval) or have a manual scan requested.

Registered in run.py as "folder-watcher-service".
"""

import logging
import time

logger = logging.getLogger(__name__)

INITIAL_DELAY = 30    # 30 seconds after startup
CHECK_INTERVAL = 30   # Check for due scans every 30s


def folder_watcher_worker(shared_state=None):
    """Entry point for the folder watcher daemon thread."""
    logger.info("[FOLDER WATCHER] Starting (initial delay %ds)", INITIAL_DELAY)
    time.sleep(INITIAL_DELAY)

    while True:
        try:
            from services.database_service import get_shared_db_service
            from services.folder_watcher_service import FolderWatcherService

            service = FolderWatcherService(get_shared_db_service())
            folders = service.get_enabled_folders()

            for folder in folders:
                try:
                    if service.is_scan_due(folder) or service.is_scan_requested(folder['id']):
                        result = service.scan_folder(folder)
                        label = folder.get('label') or folder.get('folder_path', '?')
                        total = result['new'] + result['updated'] + result['deleted'] + result['renamed']
                        if total > 0:
                            logger.info(
                                "[FOLDER WATCHER] %s: +%d new, ~%d updated, -%d deleted, ≈%d renamed",
                                label, result['new'], result['updated'],
                                result['deleted'], result['renamed'],
                            )
                        if result.get('errors'):
                            logger.warning(
                                "[FOLDER WATCHER] %s: %d errors during scan",
                                label, len(result['errors']),
                            )
                except Exception as e:
                    logger.error(
                        "[FOLDER WATCHER] Scan failed for %s: %s",
                        folder.get('id', '?'), e,
                    )

        except Exception as e:
            logger.error("[FOLDER WATCHER] Cycle error: %s", e)

        time.sleep(CHECK_INTERVAL)
