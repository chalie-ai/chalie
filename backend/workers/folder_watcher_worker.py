"""
Folder Watcher Worker - background daemon thread. Registered in run.py as
"folder-watcher-service".
"""

import logging
import time
from typing import cast

from services.folder_watcher_service import FolderWatcherService

logger = logging.getLogger(__name__)

INITIAL_DELAY = 30    # 30 seconds after startup
CHECK_INTERVAL = 30   # Check for due scans every 30s


def _scan_folder_if_due(service: FolderWatcherService, folder: dict[str, object]) -> None:
    """Scan folder if its interval is due or a manual scan was requested."""
    if not (service.is_scan_due(folder) or service.is_scan_requested(cast(str, folder['id']))):
        return
    result = service.scan_folder(folder)
    label = folder.get('label') or folder.get('folder_path', '?')
    total = cast("int", result['new']) + cast("int", result['updated']) + cast("int", result['deleted']) + cast("int", result['renamed'])
    if total > 0:
        logger.info(
            "[FOLDER WATCHER] %s: +%d new, ~%d updated, -%d deleted, ≈%d renamed",
            label, result['new'], result['updated'],
            result['deleted'], result['renamed'],
        )
    if result.get('errors'):
        logger.warning(
            "[FOLDER WATCHER] %s: %d errors during scan",
            label, len(cast(list[object], result['errors'])),
        )


def _run_scan_cycle() -> None:
    """One scan cycle across all enabled watched folders."""
    service = FolderWatcherService()
    for folder in service.get_enabled_folders():
        try:
            _scan_folder_if_due(service, folder)
        except Exception as e:
            logger.error(
                "[FOLDER WATCHER] Scan failed for %s: %s",
                folder.get('id', '?'), e,
            )


def folder_watcher_worker() -> None:
    """Daemon thread entry point."""
    logger.info("[FOLDER WATCHER] Starting (initial delay %ds)", INITIAL_DELAY)
    time.sleep(INITIAL_DELAY)

    while True:
        try:
            _run_scan_cycle()
        except Exception as e:
            logger.error("[FOLDER WATCHER] Cycle error: %s", e)

        time.sleep(CHECK_INTERVAL)
