"""
PlanAction — Proactive plan proposals from cognitive drift.

Fires when a recurring topic has sufficient activation energy and persistence
across drift cycles or conversations. Creates a persistent task with a step DAG.

Priority: 7 (same as NURTURE — router breaks ties by score)

Gates:
  1. Thought type: hypothesis or question
  2. Activation energy >= 0.7
  3. Signal persistence: topic appeared in >= 2 drift cycles OR >= 2 conversations
  4. Actionability: thought contains action verbs
  5. No similar active persistent task (Jaccard > 0.6)
  6. Active task count < MAX_ACTIVE_TASKS (5)
  7. 48h cooldown
"""

import logging
import time
from typing import Dict, Any, Optional, Tuple

from services.redis_client import RedisClientService
from .base import AutonomousAction, ActionResult, ThoughtContext

logger = logging.getLogger(__name__)
LOG_PREFIX = "[PLAN ACTION]"

# Redis keys
TOPIC_SIGNALS_KEY = "plan:topic_signals"
COOLDOWN_KEY = "plan:proposal_cooldown"
COOLDOWN_TTL = 172800  # 48 hours

# Defaults
DEFAULT_MIN_ACTIVATION = 0.7
DEFAULT_MIN_SIGNALS = 2
DEFAULT_MAX_ACTIVE_TASKS = 5
DEFAULT_ACTIONABLE_VERBS = [
    'research', 'find', 'compare', 'learn', 'build', 'prepare',
    'compile', 'analyze', 'investigate', 'explore',
]


class PlanAction(AutonomousAction):
    """
    Proposes plan-backed persistent tasks from recurring drift topics.

    Accumulates topic signals across drift cycles. On the first drift about
    a topic, records the signal. On subsequent drifts (meeting the persistence
    threshold), proposes a plan.
    """

    def __init__(self, config: dict = None):
        super().__init__(name='PLAN', enabled=True, priority=7)

        config = config or {}
        self.redis = RedisClientService.create_connection()

        self.min_activation = config.get('min_activation', DEFAULT_MIN_ACTIVATION)
        self.min_signals = config.get('min_signals', DEFAULT_MIN_SIGNALS)
        self.max_active_tasks = config.get('max_active_tasks', DEFAULT_MAX_ACTIVE_TASKS)
        self.cooldown_seconds = config.get('cooldown_seconds', COOLDOWN_TTL)
        self.actionable_verbs = config.get('actionable_verbs', DEFAULT_ACTIONABLE_VERBS)

    # ── Gate checks ───────────────────────────────────────────────

    def _thought_type_gate(self, thought: ThoughtContext) -> bool:
        """Gate 1: Only hypothesis or question thoughts trigger plans."""
        return thought.thought_type in ('hypothesis', 'question')

    def _activation_gate(self, thought: ThoughtContext) -> bool:
        """Gate 2: Activation energy must meet threshold."""
        return thought.activation_energy >= self.min_activation

    def _signal_persistence_gate(self, thought: ThoughtContext) -> Tuple[bool, int]:
        """
        Gate 3: Topic must have appeared in >= min_signals drift cycles
        OR be referenced in >= 2 conversations.

        Even when this gate fails, the signal is recorded for future cycles.
        """
        topic = thought.seed_topic
        if not topic or topic == 'general':
            return (False, 0)

        # Record this signal (always, even if gate fails)
        now = time.time()
        signal_key = f"{topic}:drift:{now}"
        self.redis.zadd(TOPIC_SIGNALS_KEY, {signal_key: now})

        # Clean signals older than 7 days
        week_ago = now - (7 * 86400)
        self.redis.zremrangebyscore(TOPIC_SIGNALS_KEY, '-inf', week_ago)

        # Count signals for this topic (from sorted set members starting with topic:)
        all_signals = self.redis.zrangebyscore(
            TOPIC_SIGNALS_KEY, week_ago, '+inf'
        )
        topic_count = sum(
            1 for s in all_signals
            if s.split(':')[0] == topic
        )

        if topic_count >= self.min_signals:
            return (True, topic_count)

        # Check conversation references as alternative
        conv_count = self._check_conversation_references(topic)
        total = topic_count + conv_count

        if total >= self.min_signals:
            return (True, total)

        logger.debug(
            f"{LOG_PREFIX} Signal persistence: {topic} has {topic_count} drift + "
            f"{conv_count} conversation signals (need {self.min_signals})"
        )
        return (False, total)

    def _actionability_gate(self, thought: ThoughtContext) -> bool:
        """Gate 4: Thought content must contain action verbs."""
        content_lower = thought.thought_content.lower()
        return any(verb in content_lower for verb in self.actionable_verbs)

    def _duplicate_gate(self, thought: ThoughtContext) -> bool:
        """Gate 5: No similar active persistent task."""
        try:
            from services.database_service import get_shared_db_service
            from services.persistent_task_service import PersistentTaskService

            db = get_shared_db_service()
            service = PersistentTaskService(db)
            account_id = self._get_account_id()
            duplicate = service.find_duplicate(account_id, thought.thought_content)
            return duplicate is None
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Duplicate gate error: {e}")
            return True  # Permissive on error

    def _active_count_gate(self) -> bool:
        """Gate 6: Active task count < max."""
        try:
            from services.database_service import get_shared_db_service
            from services.persistent_task_service import PersistentTaskService

            db = get_shared_db_service()
            service = PersistentTaskService(db)
            account_id = self._get_account_id()
            active = service.get_active_tasks(account_id)
            active_count = sum(
                1 for t in active
                if t['status'] in ('accepted', 'in_progress')
            )
            return active_count < self.max_active_tasks
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Active count gate error: {e}")
            return True

    def _cooldown_gate(self) -> bool:
        """Gate 7: 48h cooldown between plan proposals."""
        return self.redis.get(COOLDOWN_KEY) is None

    # ── Main interface ────────────────────────────────────────────

    def should_execute(self, thought: ThoughtContext) -> tuple:
        """Evaluate all 7 gates. Returns (score, eligible)."""

        # Gate 1: Thought type
        if not self._thought_type_gate(thought):
            return (0.0, False)

        # Gate 2: Activation energy
        if not self._activation_gate(thought):
            return (0.0, False)

        # Gate 3: Signal persistence (always records signal, even if gate fails)
        persistence_passes, signal_count = self._signal_persistence_gate(thought)
        if not persistence_passes:
            return (0.0, False)

        # Gate 4: Actionability
        if not self._actionability_gate(thought):
            return (0.0, False)

        # Gate 5: No duplicate task
        if not self._duplicate_gate(thought):
            return (0.0, False)

        # Gate 6: Active task count
        if not self._active_count_gate():
            return (0.0, False)

        # Gate 7: Cooldown
        if not self._cooldown_gate():
            return (0.0, False)

        score = thought.activation_energy * 0.7
        return (score, True)

    def execute(self, thought: ThoughtContext) -> ActionResult:
        """Create a plan-backed persistent task from the drift thought."""
        try:
            from services.plan_decomposition_service import PlanDecompositionService
            from services.database_service import get_shared_db_service
            from services.persistent_task_service import PersistentTaskService

            # Decompose
            decomposer = PlanDecompositionService()
            plan = decomposer.decompose(
                goal=thought.thought_content,
                scope=f"Triggered by recurring interest in: {thought.seed_topic}",
            )

            if not plan:
                return ActionResult(
                    action_name='PLAN', success=False,
                    details={'reason': 'decomposition_failed'},
                )

            # Create persistent task
            db = get_shared_db_service()
            task_service = PersistentTaskService(db)
            account_id = self._get_account_id()

            task = task_service.create_task(
                account_id=account_id,
                goal=thought.thought_content,
                scope=f"Triggered by recurring interest in: {thought.seed_topic}",
                priority=7,  # Lower priority than user-created tasks
            )
            task_id = task['id']

            # Store plan in progress
            progress = {'plan': plan, 'coverage_estimate': 0.0}
            task_service.checkpoint(task_id=task_id, progress=progress)

            cost_class = plan.get('cost_class', 'expensive')

            # Auto-start cheap plans, ask for confirmation on expensive ones
            if cost_class == 'cheap':
                task_service.transition(task_id, 'accepted')
                self._surface_auto_start(thought, plan)
            else:
                self._surface_confirmation(thought, plan)

            # Set cooldown
            self.redis.setex(COOLDOWN_KEY, self.cooldown_seconds, '1')

            # Log
            self._log_plan_event(thought, task_id, cost_class)

            logger.info(
                f"{LOG_PREFIX} Created plan task {task_id} for "
                f"'{thought.seed_topic}' ({cost_class}, "
                f"{len(plan['steps'])} steps)"
            )

            return ActionResult(
                action_name='PLAN',
                success=True,
                details={
                    'task_id': task_id,
                    'cost_class': cost_class,
                    'step_count': len(plan['steps']),
                    'topic': thought.seed_topic,
                },
            )

        except Exception as e:
            logger.error(f"{LOG_PREFIX} Execution failed: {e}", exc_info=True)
            return ActionResult(
                action_name='PLAN', success=False,
                details={'reason': str(e)},
            )

    # ── Helpers ────────────────────────────────────────────────────

    def _check_conversation_references(self, topic: str) -> int:
        """Count how many distinct conversations mention this topic (last 7 days)."""
        try:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT COUNT(DISTINCT thread_id) FROM interaction_log
                    WHERE topic = %s
                      AND created_at > NOW() - INTERVAL '7 days'
                      AND thread_id IS NOT NULL
                """, (topic,))
                count = cursor.fetchone()[0]
                cursor.close()
                return count
        except Exception:
            return 0

    @staticmethod
    def _get_account_id() -> int:
        """Get the current account ID."""
        try:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM master_account LIMIT 1")
                row = cursor.fetchone()
                return row[0] if row else 1
        except Exception:
            return 1

    def _surface_auto_start(self, thought: ThoughtContext, plan: dict):
        """Surface a brief notification for auto-started cheap plans."""
        try:
            from services.output_service import OutputService
            output = OutputService()
            message = (
                f"I'm looking into {thought.seed_topic} for you in the background. "
                f"({len(plan['steps'])} steps planned)"
            )
            output.enqueue_proactive(None, message, source='plan_action')
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Auto-start surfacing failed: {e}")

    def _surface_confirmation(self, thought: ThoughtContext, plan: dict):
        """Surface a confirmation request for expensive plans."""
        try:
            from services.output_service import OutputService
            output = OutputService()

            step_list = ', '.join(
                s['description'][:40] for s in plan['steps'][:4]
            )
            message = (
                f"I noticed you keep mentioning {thought.seed_topic}. "
                f"Want me to research this? Here's what I'd do: {step_list}"
            )
            if len(plan['steps']) > 4:
                message += f" (+{len(plan['steps']) - 4} more steps)"

            output.enqueue_proactive(None, message, source='plan_action')
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Confirmation surfacing failed: {e}")

    def _log_plan_event(self, thought: ThoughtContext, task_id: int,
                        cost_class: str):
        """Log plan creation to interaction_log."""
        try:
            from services.database_service import get_shared_db_service
            from services.interaction_log_service import InteractionLogService

            db = get_shared_db_service()
            log_service = InteractionLogService(db)
            log_service.log_event(
                event_type='plan_proposed',
                payload={
                    'task_id': task_id,
                    'topic': thought.seed_topic,
                    'cost_class': cost_class,
                    'thought_type': thought.thought_type,
                    'activation_energy': thought.activation_energy,
                },
                topic=thought.seed_topic,
                source='plan_action',
            )
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Plan event logging failed: {e}")
