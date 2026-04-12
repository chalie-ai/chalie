"""
Moment Worker — reads recent transcript and writes kind='moment' rows to data_graph.

Runs every 6h. Picks up assistant turns from the last 6h window and stores
each as a moment row via DataGraphService. Idempotent: key = moment_{transcript_id}.
"""

import logging
import time
import uuid

from services.time_utils import utc_now

logger = logging.getLogger(__name__)

LOG_PREFIX = "[MOMENT WORKER]"

POLL_INTERVAL = 21600       # 6 hours
LOOKBACK_HOURS = 6


def moment_context_worker(shared_state=None):
    """Entry point for run.py. Runs one immediate pass, then polls every 6h."""
    logger.info(f"{LOG_PREFIX} Service started (poll interval: {POLL_INTERVAL}s)")

    try:
        _poll_and_write()
    except Exception as e:
        logger.error(f"{LOG_PREFIX} Initial poll error: {e}")

    next_tick = time.monotonic() + POLL_INTERVAL
    while True:
        try:
            sleep_secs = max(0, next_tick - time.monotonic())
            time.sleep(sleep_secs)
            next_tick += POLL_INTERVAL
            _poll_and_write()
        except KeyboardInterrupt:
            logger.info(f"{LOG_PREFIX} Shutting down")
            break
        except Exception as e:
            logger.error(f"{LOG_PREFIX} Poll cycle error: {e}")
            next_tick = time.monotonic() + POLL_INTERVAL


def _poll_and_write():
    """Read recent assistant transcript turns and write kind='moment' rows."""
    from datetime import timedelta
    from services.database_service import get_shared_db_service
    from services.data_graph_service import get_data_graph_service, KIND_MOMENT

    db = get_shared_db_service()
    dg = get_data_graph_service()

    cutoff_iso = (utc_now() - timedelta(hours=LOOKBACK_HOURS)).isoformat()

    try:
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, content
                FROM transcript
                WHERE channel = 'user'
                  AND role = 'assistant'
                  AND created_at >= ?
                ORDER BY id ASC
            """, (cutoff_iso,))
            rows = cursor.fetchall()
            cursor.close()

        if not rows:
            return

        written = 0
        for row in rows:
            transcript_id, content = row[0], row[1]
            if not content:
                continue
            key = f"moment_{transcript_id}"
            result = dg.store(
                kind=KIND_MOMENT,
                key=key,
                value=content,
                source='moment_worker',
            )
            if result:
                written += 1

        if written:
            logger.info(f"{LOG_PREFIX} Wrote {written} moment row(s)")

    except Exception as e:
        logger.error(f"{LOG_PREFIX} _poll_and_write failed: {e}")
