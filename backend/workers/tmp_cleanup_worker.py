"""
Tmp Cleanup Worker — sweep stale ``chalie_*`` temp files older than 24 h.

Registered in run.py as a named WorkerManager service. Runs on startup
and then every hour, removing any ``chalie_*`` path (file or directory) under
the OS temp dir that was last modified more than 24 h ago.

Files accumulate when a user uploads a file via POST /upload but never
sends the chat message. Without this sweep those files stay in the temp dir
indefinitely. The 24 h window is generous: chat sessions last at most a
few hours, so a 24 h TTL never removes a file that is still in use.

The temp directory and prefix come from ``services.tmp_storage`` so the write
(api/upload.py), read (services/message_processor.py) and sweep sites stay in
lockstep.
"""

import logging
import os
import time

from services.tmp_storage import TMP_DIR as _TMP_DIR
from services.tmp_storage import TMP_PREFIX as _PREFIX

logger = logging.getLogger(__name__)

_MAX_AGE_SECONDS = 24 * 3600   # 24 h
_SWEEP_INTERVAL_SECONDS = 3600  # run every hour


def _sweep_once() -> int:
    """Remove stale chalie_* temp paths and return the count deleted."""
    cutoff = time.time() - _MAX_AGE_SECONDS
    deleted = 0
    try:
        for entry in os.scandir(_TMP_DIR):
            if not entry.name.startswith(_PREFIX):
                continue
            try:
                stat = entry.stat(follow_symlinks=False)
                if stat.st_mtime < cutoff:
                    if entry.is_dir(follow_symlinks=False):
                        import shutil
                        shutil.rmtree(entry.path, ignore_errors=True)
                    else:
                        os.unlink(entry.path)
                    deleted += 1
                    logger.debug('[TMP CLEANUP] Removed stale tmp path: %s', entry.path)
            except OSError as exc:
                logger.debug('[TMP CLEANUP] Could not remove %s: %s', entry.path, exc)
    except OSError as exc:
        logger.warning('[TMP CLEANUP] Could not scan %s: %s', _TMP_DIR, exc)
    return deleted


def tmp_cleanup_worker(stop_event=None) -> None:
    """Sweep loop — runs on startup then once per hour until stop_event fires."""
    logger.info('[TMP CLEANUP] Worker started (interval=%ds, max_age=%ds)',
                _SWEEP_INTERVAL_SECONDS, _MAX_AGE_SECONDS)
    while True:
        count = _sweep_once()
        if count:
            logger.info('[TMP CLEANUP] Swept %d stale tmp file(s)', count)
        # Wait for the next sweep interval or until stop is requested.
        if stop_event is not None:
            if stop_event.wait(timeout=_SWEEP_INTERVAL_SECONDS):
                break
        else:
            time.sleep(_SWEEP_INTERVAL_SECONDS)
