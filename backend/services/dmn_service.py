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
            return False

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
                "SELECT gist, action FROM episodes ORDER BY created_at DESC LIMIT 50"
            )
            rows = cursor.fetchall()

        if not rows:
            return ''

        lines = []
        for i, (gist, action) in enumerate(rows, 1):
            entry = gist or ''
            if action:
                entry = f"{entry} [{action}]" if entry else action
            lines.append(f"{i}. {entry}")
        return '\n'.join(lines)

    def _gather_salience_context(self) -> str:
        """High-confidence concepts + active goals + pending tasks."""
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

            # Active goals
            cursor.execute(
                "SELECT description FROM goals WHERE status = 'active' LIMIT 20"
            )
            goals = cursor.fetchall()
            if goals:
                lines = [f"- {description or ''}" for (description,) in goals]
                sections.append("Active goals:\n" + '\n'.join(lines))

            # Accepted (queued) background tasks
            cursor.execute(
                "SELECT goal, status FROM persistent_tasks WHERE status = 'accepted' LIMIT 10"
            )
            tasks = cursor.fetchall()
            if tasks:
                lines = [f"- {goal or ''} ({status})" for goal, status in tasks]
                sections.append("Pending tasks:\n" + '\n'.join(lines))

        return '\n\n'.join(sections)

    def _active_topic(self) -> str:
        """Return the most recently active topic, or 'general'."""
        try:
            with self._db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT topic FROM topic_transcript "
                    "ORDER BY created_at DESC LIMIT 1"
                )
                row = cursor.fetchone()
            return row[0] if row else 'general'
        except Exception:
            return 'general'

    def _build_prompt(self, mode: str, context: str) -> str:
        """Build the proactive prompt for the given mode."""
        if mode == 'recent':
            return (
                "You are reviewing recent interactions. Based on the following episodes, "
                "is there anything proactive you could do for the user? "
                "If nothing useful, respond with exactly 'DMN_NO_ACTION'. "
                "Otherwise, go ahead — respond to the user, set a schedule, create a goal, "
                "whatever is most useful.\n\nRecent episodes:\n" + context
            )
        return (
            "You are reviewing high-priority memories and active goals. "
            "Based on the following context, is there anything proactive you could do for the user? "
            "If nothing useful, respond with exactly 'DMN_NO_ACTION'. "
            "Otherwise, go ahead — respond to the user, set a schedule, create a goal, "
            "whatever is most useful.\n\n" + context
        )

    def _proactive_generate(self, mode: str, context: str) -> bool:
        """Run a proactive LLM call through unified_generate with tools.

        Calls synchronously (this runs in the DMN worker thread).
        Returns True if the LLM produced actionable output, False otherwise.
        """
        topic = self._active_topic()
        prompt = self._build_prompt(mode, context)

        try:
            from workers.digest_singletons import load_configs, get_thread_conv_service
            from workers.digest_worker import unified_generate
            from services.output_service import OutputService

            configs = load_configs()
            cortex_config = configs['cortex']['config']
            cortex_prompt_map = configs['cortex']['prompt_map']

            metadata = {
                'uuid': str(uuid.uuid4()),
                'source': 'dmn',
                'destination': 'web',
                'dmn_mode': mode,
            }

            response_data, _routing = unified_generate(
                topic=topic,
                text=prompt,
                classification={},
                thread_conv_service=get_thread_conv_service(),
                cortex_config=cortex_config,
                cortex_prompt_map=cortex_prompt_map,
                signals={},
                metadata=metadata,
                proactive=True,
            )

            response = (response_data.get('response') or '').strip()
            if not response or 'DMN_NO_ACTION' in response:
                logger.info("[DMN] %s — LLM chose no action", mode)
                return False

            # Deliver to user
            OutputService().enqueue_text(
                topic=topic,
                response=response,
                mode='DMN',
                confidence=response_data.get('confidence', 0.7),
                generation_time=response_data.get('generation_time', 0.0),
                original_metadata=metadata,
            )
            logger.info("[DMN] %s — delivered %d chars to topic '%s'", mode, len(response), topic)
            return True

        except Exception as exc:
            logger.error("[DMN] Proactive generation failed for %s: %s", mode, exc, exc_info=True)
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
