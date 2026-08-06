"""
File Index Worker - background daemon thread that watches the filesystem
with watchdog and keeps FileIndexService in sync. Registered in run.py as
"file-index-service".
"""

import logging
import os
import time

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver

from services.file_index_service import FileIndexService

logger = logging.getLogger(__name__)

INITIAL_DELAY = 30         # 30 seconds after startup
RECONCILE_INTERVAL = 3600  # Re-run reconcile() every hour as the truth path


class _FileIndexHandler(FileSystemEventHandler):
    """Dispatch watchdog events to FileIndexService methods."""

    def __init__(self, service: FileIndexService) -> None:
        self._service = service

    def on_created(self, event: FileSystemEvent) -> None:
        """A new file was created."""
        if event.is_directory:
            return
        path = os.fsdecode(event.src_path)
        try:
            if self._service.should_index(path):
                self._service.index_file(path)
        except Exception as e:
            logger.error("[FILE INDEX] on_created %s: %s", path, e)

    def on_modified(self, event: FileSystemEvent) -> None:
        """A file was modified."""
        if event.is_directory:
            return
        path = os.fsdecode(event.src_path)
        try:
            if self._service.should_index(path):
                self._service.index_file(path)
        except Exception as e:
            logger.error("[FILE INDEX] on_modified %s: %s", path, e)

    def on_deleted(self, event: FileSystemEvent) -> None:
        """A file was deleted."""
        if event.is_directory:
            return
        path = os.fsdecode(event.src_path)
        try:
            self._service.remove_file(path)
        except Exception as e:
            logger.error("[FILE INDEX] on_deleted %s: %s", path, e)

    def on_moved(self, event: FileSystemEvent) -> None:
        """A file was moved or renamed."""
        if event.is_directory:
            return
        src = os.fsdecode(event.src_path)
        dest = os.fsdecode(event.dest_path)
        try:
            if self._service.should_index(dest):
                self._service.move_file(src, dest)
            else:
                self._service.remove_file(src)
        except Exception as e:
            logger.error("[FILE INDEX] on_moved %s -> %s: %s", src, dest, e)


def _start_observer(service: FileIndexService) -> BaseObserver | None:
    """Start a recursive watchdog observer on the scan root, or None on failure.

    Failure (e.g. inotify watch limits on Linux) is logged loudly, once; the
    hourly reconcile loop still converges the index without live events.
    """
    try:
        observer = Observer()
        observer.schedule(
            _FileIndexHandler(service), service.scan_root, recursive=True,
        )
        observer.start()
    except Exception as e:
        logger.error(
            "[FILE INDEX] Observer failed to start (%s) — continuing with the "
            "hourly reconcile loop only; the index will still converge.", e,
        )
        return None
    logger.info("[FILE INDEX] Watchdog observer started on %s", service.scan_root)
    return observer


class FileIndexWorker:
    """Daemon thread entry point for the file index watchdog service."""

    @classmethod
    def run(cls) -> None:
        logger.info("[FILE INDEX] Starting (initial delay %ds)", INITIAL_DELAY)
        time.sleep(INITIAL_DELAY)

        service = FileIndexService()
        observer = _start_observer(service)

        try:
            while True:
                service.reconcile()
                time.sleep(RECONCILE_INTERVAL)
        finally:
            if observer is not None and observer.is_alive():
                observer.stop()
                observer.join(timeout=5)
