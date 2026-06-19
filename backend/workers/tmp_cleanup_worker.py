import logging
import os
import time

from services.tmp_storage import TMP_DIR as _TMP_DIR
from services.tmp_storage import TMP_PREFIX as _PREFIX

logger = logging.getLogger(__name__)

_MAX_AGE_SECONDS = 24 * 3600   # 24 h
_SWEEP_INTERVAL_SECONDS = 3600  # run every hour


def _sweep_once() -> int:
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
