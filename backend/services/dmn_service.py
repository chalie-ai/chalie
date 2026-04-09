"""
DMN Service — Default Mode Network (timer-based proactive intelligence).

Fires proactive LLM calls after the user goes idle:
  - 60min idle → recent context (last 50 episodes)
  - 6h cadence → salience context (high-confidence concepts + active goals + pending tasks)
"""

import os
import logging
import time
import uuid

from services.time_utils import utc_now, get_user_tz
from services.database_service import get_shared_db_service
from services.memory_client import MemoryClientService
from services.interaction_log_service import InteractionLogService

logger = logging.getLogger(__name__)

FIRST_IDLE_S = int(os.environ.get('CHALIE_DMN_FIRST_IDLE_S', 3600))
REPEAT_S = int(os.environ.get('CHALIE_DMN_REPEAT_S', 21600))
MAX_PROACTIVE_PER_DAY = 4
QUIET_START = 23  # 11 PM local
QUIET_END = 8     # 8 AM local

_DELIVERY_ZSET = 'dmn:deliveries'
_24H_S = 86400.0


class DMNService:
    """Default Mode Network — timer-based proactive intelligence."""

    def __init__(self):
        """Initialise the service. Called once at worker start."""
        self._last_turn_ts = utc_now()
        self._recent_fired = False
        self._last_salience_ts = None
        self._store = MemoryClientService.create_connection()
        self._db = get_shared_db_service()
        self._log = InteractionLogService(self._db)

    def on_turn(self):
        """Reset the idle timer. Call on every user or assistant message."""
        self._last_turn_ts = utc_now()
        self._recent_fired = False

    def check(self):
        """Evaluate DMN conditions. Called every 60 s by the worker loop."""
        now = utc_now()
        idle_s = (now - self._last_turn_ts).total_seconds()

        if not self._recent_fired and idle_s >= FIRST_IDLE_S:
            self._fire('recent')
            self._recent_fired = True
            self._last_salience_ts = now
            return

        if self._recent_fired and self._last_salience_ts is not None:
            since_last = (now - self._last_salience_ts).total_seconds()
            if since_last >= REPEAT_S:
                self._fire('salience')
                self._last_salience_ts = now

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _fire(self, mode: str):
        """Execute one DMN cycle for *mode* ('recent' or 'salience')."""
        if self._in_quiet_hours():
            logger.info("[DMN] Skipping %s — quiet hours", mode)
            return
        if self._rate_limit_exceeded():
            logger.info("[DMN] Skipping %s — rate limit exceeded", mode)
            return

        logger.info("[DMN] Firing %s cycle", mode)

        context = self._gather_context(mode)
        if not context:
            logger.info("[DMN] No context for %s, skipping", mode)
            self._log_cycle(mode, produced_output=False)
            return

        produced_output = self._proactive_generate(mode, context)
        self._log_cycle(mode, produced_output=produced_output)

        if produced_output:
            self._record_delivery()

    def _in_quiet_hours(self) -> bool:
        """Return True if the current local hour falls in 23:00–08:00."""
        try:
            tz = get_user_tz()
            local_hour = utc_now().astimezone(tz).hour
            # Wraps midnight: quiet if hour >= 23 OR hour < 8
            return local_hour >= QUIET_START or local_hour < QUIET_END
        except Exception as exc:
            logger.warning("[DMN] Quiet-hours check failed: %s", exc)
            return True

    def _rate_limit_exceeded(self) -> bool:
        """Return True if MAX_PROACTIVE_PER_DAY deliveries occurred in the last 24 h."""
        try:
            cutoff = utc_now().timestamp() - _24H_S
            # Prune stale entries on every check (not just on delivery)
            self._store.zremrangebyscore(_DELIVERY_ZSET, float('-inf'), cutoff)
            count = len(self._store.zrangebyscore(_DELIVERY_ZSET, cutoff, float('inf')))
            return count >= MAX_PROACTIVE_PER_DAY
        except Exception as exc:
            logger.warning("[DMN] Rate-limit check failed: %s", exc)
            return False

    def _record_delivery(self):
        """Record a proactive delivery in the rolling 24 h window."""
        try:
            ts = utc_now().timestamp()
            self._store.zadd(_DELIVERY_ZSET, {str(uuid.uuid4()): ts})
            # Prune entries older than 24 h
            cutoff = ts - _24H_S
            self._store.zremrangebyscore(_DELIVERY_ZSET, float('-inf'), cutoff)
            self._store.expire(_DELIVERY_ZSET, 86400)
        except Exception as exc:
            logger.warning("[DMN] Failed to record delivery: %s", exc)

    def _gather_context(self, mode: str) -> str:
        """Return formatted context string for *mode*, or empty string on failure."""
        try:
            if mode == 'recent':
                return self._gather_recent_context()
            return self._gather_salience_context()
        except Exception as exc:
            logger.error("[DMN] Context gathering failed for %s: %s", mode, exc, exc_info=True)
            return ''

    def _gather_recent_context(self) -> str:
        """Last 50 episodes ordered newest-first."""
        with self._db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT gist, action, outcome, salience, emotional_valence, "
                "entities, open_loops, created_at "
                "FROM episodes "
                "WHERE deleted_at IS NULL "
                "ORDER BY created_at DESC "
                "LIMIT 50"
            )
            rows = cursor.fetchall()

        if not rows:
            return ''

        return self._format_episodes(rows)

    def _gather_salience_context(self) -> str:
        """High-confidence concepts + salient episodes + active goals."""
        sections = []

        with self._db.connection() as conn:
            cursor = conn.cursor()

            # Top concepts by confidence
            cursor.execute(
                "SELECT key, value FROM knowledge "
                "WHERE kind = 'concept' AND deleted_at IS NULL "
                "ORDER BY confidence DESC LIMIT 30"
            )
            concepts = cursor.fetchall()
            if concepts:
                lines = [f"- {key}: {value}" for key, value in concepts if key or value]
                if lines:
                    sections.append("High-priority memories:\n" + '\n'.join(lines))

            # High-retrieval-weight episodes
            cursor.execute(
                "SELECT gist, action, outcome, salience, emotional_valence, "
                "entities, open_loops, created_at "
                "FROM episodes "
                "WHERE deleted_at IS NULL AND retrieval_weight >= 0.7 "
                "ORDER BY retrieval_weight DESC "
                "LIMIT 30"
            )
            ep_rows = cursor.fetchall()
            if ep_rows:
                sections.append("Notable episodes:\n" + self._format_episodes(ep_rows))

            # Active goals with confidence
            cursor.execute(
                "SELECT description, confidence FROM goals WHERE status = 'active' LIMIT 20"
            )
            goals = cursor.fetchall()
            if goals:
                lines = []
                for description, confidence in goals:
                    if confidence is not None:
                        lines.append(f"- {description or ''} (confidence: {int(confidence * 100)}%)")
                    else:
                        lines.append(f"- {description or ''}")
                sections.append("Active goals:\n" + '\n'.join(lines))

        return '\n\n'.join(sections)

    def _active_topic(self) -> str:
        """Return the most recently active user-facing channel, or 'general'."""
        try:
            with self._db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT channel FROM transcript "
                    "WHERE channel NOT LIKE 'goal_pursuit:%' "
                    "  AND channel NOT LIKE 'scheduled:%' "
                    "  AND channel NOT LIKE 'cron_tool:%' "
                    "ORDER BY created_at DESC LIMIT 1"
                )
                row = cursor.fetchone()
            return row[0] if row else 'general'
        except Exception:
            return 'general'

    def _format_episodes(self, rows) -> str:
        """Format episode rows into a readable numbered list.

        Each row must be: (gist, action, outcome, salience, emotional_valence,
        entities, open_loops, created_at)
        """
        import json

        lines = []
        for i, (gist, action, outcome, salience, emotional_valence,
                entities, open_loops, created_at) in enumerate(rows, 1):
            # Header: index, timestamp, salience, optional valence
            ts = (created_at or '')[:16].replace('T', ' ')
            header = f"{i}. [{ts}] salience:{salience}"
            if emotional_valence is not None:
                try:
                    header += f" valence:{float(emotional_valence):.1f}"
                except (ValueError, TypeError):
                    pass
            lines.append(header)

            # Gist (always present per schema NOT NULL)
            if gist:
                lines.append(f"   {gist}")

            if action:
                lines.append(f"   Action: {action}")
            if outcome:
                lines.append(f"   Outcome: {outcome}")

            # Entities (JSONB list)
            try:
                entity_list = json.loads(entities) if entities else []
                if entity_list:
                    lines.append(f"   Entities: {', '.join(str(e) for e in entity_list)}")
            except (ValueError, TypeError):
                pass

            # Open loops (JSONB list)
            try:
                loop_list = json.loads(open_loops) if open_loops else []
                if loop_list:
                    lines.append(f"   Open loops: {', '.join(str(l) for l in loop_list)}")
            except (ValueError, TypeError):
                pass

        return '\n'.join(lines)

    def _proactive_generate(self, mode: str, context: str) -> bool:
        """Run a proactive LLM call through DMNMessageProcessor.

        Calls synchronously (this runs in the DMN worker thread).
        Returns True if the LLM produced actionable output, False otherwise.
        """
        from services.dmn_message_processor import DMNMessageProcessor
        from services.output_service import OutputService

        try:
            channel = self._active_topic()
            result = DMNMessageProcessor().process(context, mode, channel)

            response = (result.get('response') or '').strip()
            if not response or 'DMN_NO_ACTION' in response:
                logger.info("[DMN] %s — no action", mode)
                return False

            OutputService().enqueue_proactive(
                topic=channel,
                response=response,
                source='dmn',
            )
            logger.info("[DMN] %s — delivered %d chars", mode, len(response))
            return True
        except Exception as exc:
            logger.error("[DMN] %s proactive generate failed: %s", mode, exc, exc_info=True)
            return False

    def _log_cycle(self, mode: str, produced_output: bool):
        """Write a dmn_fired event to interaction_log."""
        self._log.log_event(
            event_type='dmn_fired',
            payload={'mode': mode, 'produced_output': produced_output},
            source='dmn',
        )


# ── Module-level singleton ─────────────────────────────────────────────────────

_dmn_instance: DMNService | None = None


def get_dmn_service() -> DMNService:
    """Return the process-wide DMNService singleton."""
    global _dmn_instance
    if _dmn_instance is None:
        _dmn_instance = DMNService()
    return _dmn_instance


def dmn_worker(shared_state: dict = None):
    """Background worker — registered as a daemon thread in run.py."""
    logger.info("[DMN] Worker starting")
    time.sleep(30)  # Allow other services to boot first

    dmn = get_dmn_service()

    while True:
        try:
            dmn.check()
        except KeyboardInterrupt:
            logger.info("[DMN] Worker stopped")
            break
        except Exception as exc:
            logger.error("[DMN] Check failed: %s", exc, exc_info=True)
        time.sleep(60)
