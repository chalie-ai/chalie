"""
NurtureAction — Gentle presence signals during early relationship phases.

Fills the dead zone between spark-welcome (first_contact) and COMMUNICATE/SUGGEST
(connected phase). Sends ~1 message per day during surface and exploratory phases.

Priority: 7 (below SUGGEST=8, above SEED_THREAD=6)

Gates:
  1. Phase gate: Must be surface or exploratory
  2. Timing gate: Min idle (6h surface, 2h exploratory), quiet hours, daily cooldown
  3. Backoff gate: Max 3 unanswered before pause
  4. Content gate: Must have at least 1 episode

Self-disables: Stops when phase reaches connected (COMMUNICATE + SUGGEST take over).
"""

import logging
import time
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

from services.redis_client import RedisClientService
from .base import AutonomousAction, ActionResult, ThoughtContext

logger = logging.getLogger(__name__)

LOG_PREFIX = "[NURTURE]"

_NS = "spark_nurture"


def _key(user_id: str, suffix: str) -> str:
    return f"{_NS}:{user_id}:{suffix}"


class NurtureAction(AutonomousAction):
    """
    Sends gentle, phase-appropriate presence signals during surface and exploratory
    phases. Uses the drift cycle as a timing opportunity but generates its own
    content via a dedicated prompt.
    """

    def __init__(self, config: dict = None):
        super().__init__(name='NURTURE', enabled=True, priority=7)

        config = config or {}
        self.redis = RedisClientService.create_connection()
        self.user_id = config.get('user_id', 'default')

        # Phase-specific minimum idle (seconds)
        self.min_idle_surface = config.get('min_idle_surface', 21600)        # 6h
        self.min_idle_exploratory = config.get('min_idle_exploratory', 7200)  # 2h

        # Daily cooldown (multiplied by backoff)
        self.daily_cooldown_seconds = config.get('daily_cooldown_seconds', 86400)  # 24h

        # Backoff
        self.max_unanswered = config.get('max_unanswered', 3)
        self.backoff_multiplier_cap = config.get('backoff_multiplier_cap', 4)

        # Quiet hours (same convention as COMMUNICATE)
        self.quiet_hours_start = config.get('quiet_hours_start', 23)
        self.quiet_hours_end = config.get('quiet_hours_end', 8)

        # LLM timeout
        self.llm_timeout = config.get('llm_timeout', 8.0)

        # Score: intentionally modest so COMMUNICATE wins when eligible
        self.base_score = config.get('base_score', 0.35)

    # ── Gate checks ───────────────────────────────────────────────

    def _phase_gate(self) -> Tuple[bool, str]:
        """Phase must be surface or exploratory. Self-disables at connected+."""
        try:
            from services.spark_state_service import SparkStateService
            phase = SparkStateService(user_id=self.user_id).get_phase()
            if phase in ('surface', 'exploratory'):
                return (True, phase)
            return (False, phase)
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Phase gate error: {e}")
            return (False, 'unknown')

    def _timing_gate(self, phase: str) -> Tuple[bool, Dict]:
        """Check idle time, quiet hours, and daily cooldown."""
        now = time.time()
        details: Dict[str, Any] = {}

        # 1. Quiet hours
        current_hour = datetime.now().hour
        if self._is_quiet_hours(current_hour):
            details['rejected'] = 'quiet_hours'
            return (False, details)

        # 2. Must have prior interaction history (reuse COMMUNICATE's key)
        last_interaction = self.redis.get(f"proactive:{self.user_id}:last_interaction_ts")
        if not last_interaction:
            details['rejected'] = 'no_interaction_history'
            return (False, details)

        idle_seconds = now - float(last_interaction)
        details['idle_seconds'] = idle_seconds

        # 3. Phase-specific minimum idle
        min_idle = self.min_idle_surface if phase == 'surface' else self.min_idle_exploratory
        details['min_idle'] = min_idle
        if idle_seconds < min_idle:
            details['rejected'] = 'too_soon'
            return (False, details)

        # 4. Daily cooldown (multiplied by backoff)
        backoff = int(self.redis.get(_key(self.user_id, 'backoff_multiplier')) or 1)
        effective_cooldown = self.daily_cooldown_seconds * backoff
        details['backoff'] = backoff
        details['effective_cooldown'] = effective_cooldown

        last_sent = self.redis.get(_key(self.user_id, 'last_sent_ts'))
        if last_sent and (now - float(last_sent)) < effective_cooldown:
            details['rejected'] = 'daily_cooldown'
            return (False, details)

        return (True, details)

    def _backoff_gate(self) -> Tuple[bool, Dict]:
        """Check unanswered count and pause state."""
        details: Dict[str, Any] = {}

        # Paused?
        if self.redis.get(_key(self.user_id, 'paused')) == '1':
            details['rejected'] = 'paused'
            return (False, details)

        # Too many unanswered?
        unanswered = int(self.redis.get(_key(self.user_id, 'unanswered_count')) or 0)
        details['unanswered'] = unanswered
        if unanswered >= self.max_unanswered:
            self.redis.set(_key(self.user_id, 'paused'), '1')
            logger.info(f"{LOG_PREFIX} Paused — {unanswered} unanswered nurtures")
            details['rejected'] = 'max_unanswered'
            return (False, details)

        return (True, details)

    def _content_gate(self) -> Tuple[bool, Dict]:
        """Must have at least 1 episode to ground the message."""
        try:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT COUNT(*) FROM episodes WHERE deleted_at IS NULL"
                )
                count = cursor.fetchone()[0]
                cursor.close()
            if count >= 1:
                return (True, {'episodes': count})
            return (False, {'rejected': 'no_episodes'})
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Content gate error: {e}")
            return (False, {'rejected': 'db_error'})

    def _is_quiet_hours(self, hour: int) -> bool:
        start = self.quiet_hours_start
        end = self.quiet_hours_end
        if start > end:
            return hour >= start or hour < end
        else:
            return start <= hour < end

    # ── Main interface ────────────────────────────────────────────

    def should_execute(self, thought: ThoughtContext) -> tuple:
        """Evaluate all gates. Returns (score, eligible)."""

        # Gate 1: Phase
        phase_passes, phase = self._phase_gate()
        if not phase_passes:
            return (0.0, False)

        # Gate 2: Timing
        timing_passes, timing_details = self._timing_gate(phase)
        if not timing_passes:
            return (0.0, False)

        # Gate 3: Backoff
        backoff_passes, backoff_details = self._backoff_gate()
        if not backoff_passes:
            return (0.0, False)

        # Gate 4: Content
        content_passes, content_details = self._content_gate()
        if not content_passes:
            return (0.0, False)

        # Store phase for execute()
        self._pending_phase = phase

        return (self.base_score, True)

    def execute(self, thought: ThoughtContext) -> ActionResult:
        """Generate and deliver a nurture message."""
        phase = getattr(self, '_pending_phase', 'surface')

        # 1. Gather context
        context = self._gather_context(thought, phase)

        # 2. Generate message via LLM
        message = self._generate_nurture_message(context, phase)
        if not message:
            self._track_llm_failure()
            return ActionResult(
                action_name='NURTURE', success=False,
                details={'reason': 'generation_failed'},
            )

        # 3. Deliver
        self._deliver_nurture(message)

        # 4. Update tracking
        now = time.time()
        self.redis.set(_key(self.user_id, 'last_sent_ts'), str(now))
        self.redis.incr(_key(self.user_id, 'unanswered_count'))
        self.redis.incr(_key(self.user_id, 'total_sent'))

        # Increase backoff (capped)
        current_backoff = int(
            self.redis.get(_key(self.user_id, 'backoff_multiplier')) or 1
        )
        new_backoff = min(current_backoff * 2, self.backoff_multiplier_cap)
        self.redis.set(_key(self.user_id, 'backoff_multiplier'), str(new_backoff))

        # 5. Log event
        self._log_nurture_event(phase, thought)

        logger.info(
            f"{LOG_PREFIX} Nurture sent (phase={phase}, backoff={new_backoff}x)"
        )

        return ActionResult(
            action_name='NURTURE',
            success=True,
            details={
                'phase': phase,
                'backoff': new_backoff,
            },
        )

    # ── Context gathering ─────────────────────────────────────────

    def _gather_context(self, thought: ThoughtContext, phase: str) -> dict:
        """Gather context for the LLM prompt."""
        context: Dict[str, Any] = {
            'phase': phase,
            'thought_content': thought.thought_content,
            'thought_type': thought.thought_type,
            'seed_topic': thought.seed_topic,
            'topics_discussed': [],
            'recent_episode_gist': '',
            'days_since_welcome': 0,
        }

        # Spark state: topics and timing
        try:
            from services.spark_state_service import SparkStateService
            state = SparkStateService(user_id=self.user_id).get_state()
            context['topics_discussed'] = state.get('topics_discussed', [])
            welcome_sent_at = state.get('welcome_sent_at')
            if welcome_sent_at:
                context['days_since_welcome'] = (time.time() - welcome_sent_at) / 86400
        except Exception:
            pass

        # Most recent episode gist
        try:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT gist, topic FROM episodes "
                    "WHERE deleted_at IS NULL "
                    "ORDER BY created_at DESC LIMIT 1"
                )
                row = cursor.fetchone()
                cursor.close()
                if row:
                    context['recent_episode_gist'] = row[0] or ''
        except Exception:
            pass

        return context

    # ── LLM generation ────────────────────────────────────────────

    def _generate_nurture_message(
        self, context: dict, phase: str,
    ) -> Optional[str]:
        """Generate nurture message via LLM. Returns None on failure."""
        import threading

        llm_result: list = [None]
        llm_done = threading.Event()

        def _llm_generate():
            try:
                from services.config_service import ConfigService
                from services.llm_service import create_llm_service

                try:
                    config = ConfigService.resolve_agent_config(
                        "frontal-cortex-acknowledge"
                    )
                except Exception:
                    config = ConfigService.resolve_agent_config("frontal-cortex")

                config = dict(config)
                config['format'] = ''

                prompt = ConfigService.get_agent_prompt("spark-nurture")

                # Inject variables
                topics_str = ', '.join(
                    context.get('topics_discussed', [])[:5]
                ) or 'none yet'

                prompt = prompt.replace('{{phase}}', phase)
                prompt = prompt.replace(
                    '{{thought_content}}', context.get('thought_content', '')
                )
                prompt = prompt.replace(
                    '{{seed_topic}}', context.get('seed_topic', '')
                )
                prompt = prompt.replace('{{topics_discussed}}', topics_str)
                prompt = prompt.replace(
                    '{{recent_episode_gist}}',
                    context.get('recent_episode_gist', '')[:300] or 'no memories yet',
                )
                prompt = prompt.replace(
                    '{{days_since_welcome}}',
                    str(round(context.get('days_since_welcome', 0), 1)),
                )

                llm = create_llm_service(config)
                response = llm.send_message(
                    prompt, "Generate a nurture message."
                ).text

                text = response.strip().strip('"').strip("'").strip()

                # Validate: reject too long, too short
                if not text or len(text) < 10 or len(text) > 300:
                    return

                # Strip exclamation marks — too eager for early relationship
                text = text.replace('!', '.')

                llm_result[0] = text
            except Exception as e:
                logger.warning(f"{LOG_PREFIX} LLM generation failed: {e}")
            finally:
                llm_done.set()

        thread = threading.Thread(target=_llm_generate, daemon=True)
        thread.start()

        llm_done.wait(timeout=self.llm_timeout)

        return llm_result[0]

    # ── Delivery ──────────────────────────────────────────────────

    def _deliver_nurture(self, text: str) -> None:
        """Deliver via OutputService (drift stream)."""
        from services.output_service import OutputService

        output_svc = OutputService()
        output_svc.enqueue_text(
            topic='spark_nurture',
            response=text,
            mode='RESPOND',
            confidence=0.7,
            generation_time=0.0,
            original_metadata={
                'source': 'spark_nurture',
            },
        )

    # ── Observability ─────────────────────────────────────────────

    def _track_llm_failure(self) -> None:
        """Track silent LLM failures for observability."""
        try:
            fail_key = _key(self.user_id, 'llm_fail_count')
            self.redis.incr(fail_key)
            self.redis.expire(fail_key, 604800)  # 7-day window
            self.redis.set(
                _key(self.user_id, 'last_llm_fail_ts'), str(time.time())
            )
        except Exception:
            pass

    def _log_nurture_event(self, phase: str, thought: ThoughtContext) -> None:
        """Log nurture event to interaction_log."""
        try:
            from services.database_service import get_shared_db_service
            from services.interaction_log_service import InteractionLogService

            db = get_shared_db_service()
            log_service = InteractionLogService(db)
            log_service.log_event(
                event_type='spark_nurture_sent',
                payload={
                    'phase': phase,
                    'seed_topic': thought.seed_topic,
                    'thought_type': thought.thought_type,
                },
                topic='spark_nurture',
                source='spark_nurture',
            )
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Nurture event logging failed: {e}")

    # ── User activity reset (called from digest worker) ───────────

    @staticmethod
    def record_user_activity(user_id: str = 'default') -> None:
        """
        Reset unanswered count, backoff, and pause when the user sends a message.

        Called from digest_worker alongside spark exchange tracking.
        """
        try:
            redis = RedisClientService.create_connection()
            redis.set(f"{_NS}:{user_id}:unanswered_count", '0')
            redis.set(f"{_NS}:{user_id}:backoff_multiplier", '1')
            redis.delete(f"{_NS}:{user_id}:paused")
        except Exception:
            pass
