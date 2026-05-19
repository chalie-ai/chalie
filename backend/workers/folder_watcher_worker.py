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


def _scan_folder_if_due(service, folder: dict) -> None:
    """Scan a single folder when it is due or has been manually requested."""
    if not (service.is_scan_due(folder) or service.is_scan_requested(folder['id'])):
        return
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


def _run_scan_cycle() -> None:
    """Load all enabled folders and scan each one that is due."""
    from services.database_service import get_shared_db_service
    from services.folder_watcher_service import FolderWatcherService

    service = FolderWatcherService(get_shared_db_service())
    for folder in service.get_enabled_folders():
        try:
            _scan_folder_if_due(service, folder)
        except Exception as e:
            logger.error(
                "[FOLDER WATCHER] Scan failed for %s: %s",
                folder.get('id', '?'), e,
            )


def folder_watcher_worker():
    """Entry point for the folder watcher daemon thread."""
    logger.info("[FOLDER WATCHER] Starting (initial delay %ds)", INITIAL_DELAY)
    time.sleep(INITIAL_DELAY)

    while True:
        try:
            _run_scan_cycle()
        except Exception as e:
            logger.error("[FOLDER WATCHER] Cycle error: %s", e)

        time.sleep(CHECK_INTERVAL)
