"""
Moment Enrichment Service — Background worker that enriches pinned moments.

Polls every 5 minutes for moments with status='enriching'. For each:
1. Collects gists from interaction_log in ±4hr window around pinned_at
2. Merges into moment's gists array (Jaccard dedup)
3. Generates LLM summary when >= 2 gists collected
4. Seals the moment when pinned_at + 4hrs has passed

Entry point: moment_enrichment_worker(shared_state=None) registered in consumer.py.
"""

import logging
import time
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

LOG_PREFIX = "[MOMENT ENRICHMENT]"
_POLL_INTERVAL = 300  # 5 minutes


def moment_enrichment_worker(shared_state=None):
    """Module-level entry point for consumer.py."""
    logging.basicConfig(level=logging.INFO)
    logger.info(f"{LOG_PREFIX} Service started (poll interval: {_POLL_INTERVAL}s)")

    next_tick = time.monotonic() + _POLL_INTERVAL
    while True:
        try:
            now = time.monotonic()
            sleep_secs = max(0, next_tick - now)
            time.sleep(sleep_secs)
            next_tick += _POLL_INTERVAL
            _poll_and_enrich()
        except KeyboardInterrupt:
            logger.info(f"{LOG_PREFIX} Shutting down")
            break
        except Exception as e:
            logger.error(f"{LOG_PREFIX} Poll cycle error: {e}")
            next_tick = time.monotonic() + _POLL_INTERVAL


def _poll_and_enrich():
    """Poll for enriching moments and process each."""
    try:
        from services.database_service import get_shared_db_service
        from services.moment_service import MomentService

        db = get_shared_db_service()
        service = MomentService(db)
        moments = service.get_enriching_moments()

        if not moments:
            return

        logger.info(f"{LOG_PREFIX} Processing {len(moments)} enriching moment(s)")

        now = datetime.now(timezone.utc)

        for moment in moments:
            try:
                _enrich_single_moment(service, db, moment, now)
            except Exception as e:
                logger.error(f"{LOG_PREFIX} Failed to enrich moment {moment['id']}: {e}")

    except Exception as e:
        logger.error(f"{LOG_PREFIX} _poll_and_enrich failed: {e}")


def _enrich_single_moment(service, db, moment, now):
    """Enrich a single moment with gists from interaction_log."""
    moment_id = moment["id"]
    topic = moment.get("topic")
    pinned_at = moment.get("pinned_at")

    if not pinned_at:
        logger.warning(f"{LOG_PREFIX} Moment {moment_id} has no pinned_at, skipping")
        return

    # Ensure timezone-aware
    if pinned_at.tzinfo is None:
        pinned_at = pinned_at.replace(tzinfo=timezone.utc)

    window_start = pinned_at - timedelta(hours=4)
    window_end = min(now, pinned_at + timedelta(hours=4))

    # Collect interaction entries from the time window
    new_gists = _collect_gists_from_interactions(db, topic, window_start, window_end)

    # Merge gists
    gists_changed = False
    if new_gists:
        gists_changed = service.enrich_moment(moment_id, new_gists)

    # Regenerate summary if gists changed and >= 2 gists
    if gists_changed:
        updated_moment = service.get_moment(moment_id)
        if updated_moment and len(updated_moment.get("gists") or []) >= 2:
            service.generate_summary(moment_id)

    # Seal if past enrichment window
    if now > pinned_at + timedelta(hours=4):
        # Generate final summary if not yet done and has gists
        current_moment = service.get_moment(moment_id)
        if current_moment:
            gists = current_moment.get("gists") or []
            if len(gists) >= 2 and not current_moment.get("summary"):
                service.generate_summary(moment_id)

            service.seal_moment(moment_id)
            logger.info(f"{LOG_PREFIX} Sealed moment {moment_id} (past 4hr window)")


def _collect_gists_from_interactions(db, topic, window_start, window_end):
    """Query interaction_log for entries in the time window and extract gist-like summaries."""
    gists = []

    try:
        with db.connection() as conn:
            cursor = conn.cursor()

            # Build query — filter by topic if available, otherwise use time window only
            if topic:
                cursor.execute("""
                    SELECT payload, event_type, created_at
                    FROM interaction_log
                    WHERE topic = %s
                      AND event_type IN ('system_response', 'user_input')
                      AND created_at BETWEEN %s AND %s
                    ORDER BY created_at ASC
                    LIMIT 20
                """, (topic, window_start, window_end))
            else:
                cursor.execute("""
                    SELECT payload, event_type, created_at
                    FROM interaction_log
                    WHERE event_type IN ('system_response', 'user_input')
                      AND created_at BETWEEN %s AND %s
                    ORDER BY created_at ASC
                    LIMIT 20
                """, (window_start, window_end))

            rows = cursor.fetchall()
            cursor.close()

        # Extract text from payloads and create short summaries
        for payload, event_type, created_at in rows:
            text = _extract_text_from_payload(payload)
            if text and len(text) > 10:
                # Truncate long entries to gist-like summaries
                gist = text[:200].strip()
                if len(text) > 200:
                    gist += "..."
                gists.append(gist)

    except Exception as e:
        logger.warning(f"{LOG_PREFIX} _collect_gists_from_interactions failed: {e}")

    return gists


def _extract_text_from_payload(payload):
    """Extract text content from interaction_log payload JSONB."""
    if not payload:
        return ""

    if isinstance(payload, str):
        import json
        try:
            payload = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            return payload

    if isinstance(payload, dict):
        return payload.get("message", "") or payload.get("text", "") or ""

    return ""
