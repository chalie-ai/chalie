"""Minute-aligned cron runner — the WorkerManager entry point.

Mirrors the minute-alignment loop from ``scheduler_service.py``: sleep until
the next wall-clock minute, then iterate ``cron.JOBS``. Each job's
``should_run()`` and dispatch are wrapped in try/except so one bad job never
kills the tick or the thread. Jobs that need to run acquire their own
non-reentrant ``_lock`` (set by ``ScheduledJob.__init__``); if already held
the tick skips that job for the minute.
"""

import logging
import threading
import time

from cron.base import ScheduledJob
from services.time_utils import utc_now
from utils.logger import Logger

logger = logging.getLogger(__name__)

LOG_PREFIX = "[CRON]"


def cron_runner() -> None:
    """WorkerManager entry point. Minute-aligned poll loop."""
    Logger.start()
    logger.info(f"{LOG_PREFIX} Service started (minute-aligned poll)")

    while True:
        try:
            now = utc_now()
            sleep_secs = 60 - now.second - now.microsecond / 1e6
            time.sleep(sleep_secs)
            _poll_and_dispatch()
        except KeyboardInterrupt:
            logger.info(f"{LOG_PREFIX} Shutting down")
            break
        except Exception as e:
            logger.exception(f"{LOG_PREFIX} Poll cycle error: {e}")


def _poll_and_dispatch() -> None:
    """Iterate ``cron.JOBS`` and dispatch each runnable job on its own thread."""
    from cron import JOBS  # noqa: PLC0415

    for job in JOBS:
        try:
            should_run = job.should_run()
        except Exception as e:
            logger.exception(f"{LOG_PREFIX} should_run failed for {job.name}: {e}")
            continue

        if not should_run:
            continue

        acquired = job._lock.acquire(blocking=False)
        if not acquired:
            logger.info(f"{LOG_PREFIX} {job.name} still running — skipping")
            continue

        # The lock stays held for the job's whole lifetime; ``_run_job``
        # releases it in its ``finally``. Only release here if the thread
        # never starts (so a failed launch can't leave the job wedged).
        try:
            threading.Thread(
                target=_run_job,
                args=(job,),
                daemon=True,
                name=f"cron-{job.name}",
            ).start()
        except Exception as e:
            logger.exception(f"{LOG_PREFIX} thread start failed for {job.name}: {e}")
            job._lock.release()


def _run_job(job: ScheduledJob) -> None:
    """Execute a job on its own thread, releasing the lock in ``finally``."""
    try:
        job.execute()
    except Exception as e:
        logger.exception(f"{LOG_PREFIX} {job.name} failed: {e}")
    finally:
        job._lock.release()
