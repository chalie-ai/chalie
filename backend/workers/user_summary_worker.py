"""
User Summary Worker — 30-minute cadence daemon for user-profile re-synthesis.

Sleeps 1800 seconds between ticks.  Each tick calls ``_should_synthesise()``
and, when True, runs ``UserSummaryProcessor().send()`` so that
``UserSummaryProcessor.postTurn()`` writes the updated synopsis rows to
data_graph.

NO immediate-on-boot fire.  The lazy fallback in
``UserMessageProcessor.getUserDefinition()`` handles cold-start synthesis.

``_should_synthesise()`` compares the most-recently-confirmed ``user_specific``
row timestamp against the ``user_summary`` row timestamp (both from
``data_graph.last_confirmed_at``).  Filtering strictly by
``kind='user_specific'`` in the MAX query prevents an infinite re-synthesis
loop where the summary row itself would always appear newer than itself.
"""

import logging
import time

logger = logging.getLogger(__name__)

_TICK_INTERVAL = 1800  # 30 minutes


def _should_synthesise() -> bool:
    """Return True when re-synthesis is warranted.

    Logic:
    - ``latest_trait_ts`` = MAX(last_confirmed_at) WHERE kind='user_specific' AND active=1
    - ``current_summary`` = row WHERE kind='system' AND key='user_summary' AND active=1

    Cases:
    - No traits at all → False (nothing to synthesise).
    - Traits exist, no summary → True (first synthesis needed).
    - Both exist → True only when latest trait is newer than the summary row.

    The ``kind='user_specific'`` filter on the MAX query is CRITICAL: querying
    all kinds would include the ``user_summary`` row itself, which is always
    confirmed more recently after synthesis, creating a permanent re-synthesis
    loop.
    """
    try:
        from services.database_service import get_shared_db_service

        db = get_shared_db_service()
        with db.connection() as conn:
            row = conn.execute(
                """
                SELECT MAX(last_confirmed_at)
                FROM data_graph
                WHERE kind = 'user_specific'
                  AND active = 1
                  AND deleted_at IS NULL
                """
            ).fetchone()
            latest_trait_ts = row[0] if row else None

            if latest_trait_ts is None:
                return False

            summary_row = conn.execute(
                """
                SELECT last_confirmed_at
                FROM data_graph
                WHERE kind = 'system'
                  AND key = 'user_summary'
                  AND active = 1
                  AND deleted_at IS NULL
                LIMIT 1
                """
            ).fetchone()

        if summary_row is None:
            return True

        from services.time_utils import parse_utc

        try:
            trait_dt = parse_utc(latest_trait_ts)
            summary_dt = parse_utc(summary_row[0])
        except Exception as exc:
            # Un-parseable timestamp → schema drift or corrupted row. Loud log
            # so the worker doesn't silently starve re-synthesis forever.
            logger.error(
                "[USER SUMMARY WORKER] parse_utc failed on trait_ts=%r summary_ts=%r: %s",
                latest_trait_ts,
                summary_row[0],
                exc,
            )
            return False

        return trait_dt > summary_dt

    except Exception as exc:
        logger.warning("[USER SUMMARY WORKER] _should_synthesise failed: %s", exc)
        return False


def run(shared_state=None):
    """Entry point registered with run.py.

    Runs indefinitely.  Sleeps ``_TICK_INTERVAL`` seconds between ticks so
    the first fire is intentionally delayed — cold-start synthesis is
    handled by the lazy fallback in UserMessageProcessor.getUserDefinition().
    """
    logger.info("[USER SUMMARY WORKER] Started (tick_interval=%ds)", _TICK_INTERVAL)

    while True:
        time.sleep(_TICK_INTERVAL)

        try:
            if not _should_synthesise():
                logger.debug("[USER SUMMARY WORKER] tick: no synthesis needed")
                continue

            logger.info("[USER SUMMARY WORKER] tick: synthesising user profile")
            from services.user_summary_processor import UserSummaryProcessor

            UserSummaryProcessor().send()
            logger.info("[USER SUMMARY WORKER] tick: synthesis complete")

        except Exception as exc:
            logger.warning("[USER SUMMARY WORKER] tick failed: %s", exc)
            # Continue — next tick will reconsider.
