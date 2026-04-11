"""
Moment Context Enrichment Worker.

Runs every 6h. Finds kind='moment' rows older than 4h with no context rows.
Stores N-5/N+5 transcript turns as moment_context knowledge entries.
"""

import json
import logging
import time

from services.time_utils import utc_now

logger = logging.getLogger(__name__)

LOG_PREFIX = "[MOMENT CONTEXT]"

POLL_INTERVAL = 21600       # 6 hours
MOMENT_AGE_HOURS = 4
CAP_PER_PASS = 50
CTX_BEFORE = 5              # transcript rows pulled from *before* the pinned id
CTX_AFTER = 5               # transcript rows pulled from *after*  the pinned id


def moment_context_worker(shared_state=None):
    """Entry point for run.py. Runs one immediate pass, then polls every 6h."""
    logger.info(f"{LOG_PREFIX} Service started (poll interval: {POLL_INTERVAL}s)")

    # Immediate first pass so any pins made before the worker booted get
    # enriched without waiting for the full 6h interval.
    try:
        _poll_and_enrich()
    except Exception as e:
        logger.error(f"{LOG_PREFIX} Initial poll error: {e}")

    next_tick = time.monotonic() + POLL_INTERVAL
    while True:
        try:
            sleep_secs = max(0, next_tick - time.monotonic())
            time.sleep(sleep_secs)
            next_tick += POLL_INTERVAL
            _poll_and_enrich()
        except KeyboardInterrupt:
            logger.info(f"{LOG_PREFIX} Shutting down")
            break
        except Exception as e:
            logger.error(f"{LOG_PREFIX} Poll cycle error: {e}")
            next_tick = time.monotonic() + POLL_INTERVAL


def _poll_and_enrich():
    """
    Find un-enriched moments and add context rows.

    Idempotency lives in SQL — the NOT EXISTS clause guarantees we only pull
    rows that don't already have any moment_context children, regardless of
    how many enriched moments exist above the cap.
    """
    from datetime import timedelta
    from services.database_service import get_shared_db_service
    from services.knowledge_service import KnowledgeService

    db = get_shared_db_service()
    ks = KnowledgeService(db)

    cutoff_iso = (utc_now() - timedelta(hours=MOMENT_AGE_HOURS)).isoformat()

    try:
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT k.id, k.entity, k.key, k.value, k.data, k.created_at
                FROM knowledge k
                WHERE k.kind = 'moment'
                  AND k.deleted_at IS NULL
                  AND k.created_at < ?
                  AND NOT EXISTS (
                      SELECT 1 FROM knowledge c
                      WHERE c.entity = k.key
                        AND c.kind = 'moment_context'
                        AND c.deleted_at IS NULL
                  )
                ORDER BY k.created_at ASC
                LIMIT ?
            """, (cutoff_iso, CAP_PER_PASS))
            rows = cursor.fetchall()
            cursor.close()

        if not rows:
            return

        logger.info(f"{LOG_PREFIX} Processing {len(rows)} moment(s)")
        for row in rows:
            try:
                _enrich_moment(db, ks, row)
            except Exception as e:
                logger.error(f"{LOG_PREFIX} Failed to enrich moment key={row[2]}: {e}")

    except Exception as e:
        logger.error(f"{LOG_PREFIX} _poll_and_enrich failed: {e}")


def _enrich_moment(db, ks, moment_row):
    """Add context turns to a single moment if not already enriched."""
    # moment_row columns: id, entity, key, value, data, created_at
    _row_id, entity, key, _value, data_raw, _created_at = moment_row

    data = {}
    if data_raw:
        try:
            data = json.loads(data_raw)
        except (json.JSONDecodeError, TypeError):
            pass

    transcript_id = data.get('transcript_id')
    if not transcript_id:
        logger.debug(f"{LOG_PREFIX} No transcript_id on moment key={key}, skipping")
        return

    # Race guard: by the time the worker picks up this row the parent may
    # already have been forgotten. Re-check before writing orphaned rows.
    parent_entity = entity or 'user'
    if ks.get(parent_entity, key) is None:
        logger.debug(f"{LOG_PREFIX} Parent moment gone mid-enrichment (key={key}), skipping")
        return

    if _is_enriched(db, key):
        return

    turns = _fetch_context_turns(db, transcript_id)
    if not turns:
        return

    for n, turn in enumerate(turns):
        ctx_key = f"{key}_ctx_{n}"
        ctx_value = f"[{turn['role']}] {turn['content']}"
        ctx_data = {
            'parent_key': key,
            'transcript_id': turn['id'],
            'role': turn['role'],
            'ctx_index': n,
        }
        ks.store(
            kind='moment_context',
            entity=key,
            key=ctx_key,
            value=ctx_value,
            data=ctx_data,
            decay_class='permanent',
            confidence=1.0,
            source='moment_enrichment',
        )

    logger.info(f"{LOG_PREFIX} Enriched moment key={key} with {len(turns)} context turns")


def _fetch_context_turns(db, transcript_id: int, limit_before: int = CTX_BEFORE,
                         limit_after: int = CTX_AFTER) -> list:
    """
    Return up to `limit_before + limit_after` nearest transcript turns.

    Two indexed range scans (< and >) are joined via UNION ALL — O(log n)
    per side rather than the full-table scan triggered by
    `ORDER BY ABS(id - ?)`. Ordered from earliest to latest for stable
    ctx_index assignment downstream.
    """
    try:
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, role, content FROM (
                    SELECT id, role, content
                    FROM transcript
                    WHERE channel = 'user' AND id < ?
                    ORDER BY id DESC
                    LIMIT ?
                )
                UNION ALL
                SELECT id, role, content FROM (
                    SELECT id, role, content
                    FROM transcript
                    WHERE channel = 'user' AND id > ?
                    ORDER BY id ASC
                    LIMIT ?
                )
                ORDER BY id ASC
                """,
                (transcript_id, limit_before, transcript_id, limit_after),
            )
            rows = cursor.fetchall()
            cursor.close()
        return [{'id': r[0], 'role': r[1], 'content': r[2]} for r in rows]
    except Exception as e:
        logger.warning(f"{LOG_PREFIX} _fetch_context_turns failed: {e}")
        return []


def _is_enriched(db, parent_key: str) -> bool:
    """True if context rows already exist for this parent moment key."""
    try:
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 1 FROM knowledge
                WHERE entity = ?
                  AND kind = 'moment_context'
                  AND deleted_at IS NULL
                LIMIT 1
            """, (parent_key,))
            row = cursor.fetchone()
            cursor.close()
        return row is not None
    except Exception as e:
        logger.warning(f"{LOG_PREFIX} _is_enriched check failed: {e}")
        return False
