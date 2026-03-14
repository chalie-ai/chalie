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
  7. 48h cooldown (adaptive: multiplied by backoff from cancellation learning)

Outcome learning:
  - Cancelled tasks increase backoff multiplier (1.5x per cancellation, cap 4.0)
  - Completed tasks decrease backoff multiplier (0.8x, floor 1.0)
  - Backoff decays linearly to 1.0 after 7 days of no cancellations
"""

import logging
import time
from typing import Dict, Any, Optional, Tuple

from services.memory_client import MemoryClientService
from .base import AutonomousAction, ActionResult, ThoughtContext

logger = logging.getLogger(__name__)
LOG_PREFIX = "[PLAN ACTION]"

# MemoryStore keys
TOPIC_SIGNALS_KEY = "plan:topic_signals"
COOLDOWN_KEY = "plan:proposal_cooldown"
COOLDOWN_TTL = 172800  # 48 hours

# Backoff learning keys
BACKOFF_MULTIPLIER_KEY = "plan_action:backoff_multiplier"
LAST_CANCELLATION_KEY = "plan_action:last_cancellation_at"

# Backoff constants
BACKOFF_INCREASE_FACTOR = 1.5
BACKOFF_DECREASE_FACTOR = 0.8
BACKOFF_MAX = 4.0
BACKOFF_MIN = 1.0
BACKOFF_DECAY_DAYS = 7

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
        self.store = MemoryClientService.create_connection()

        self.min_activation = config.get('min_activation', DEFAULT_MIN_ACTIVATION)
        self.min_signals = config.get('min_signals', DEFAULT_MIN_SIGNALS)
        self.max_active_tasks = config.get('max_active_tasks', DEFAULT_MAX_ACTIVE_TASKS)
        self.cooldown_seconds = config.get('cooldown_seconds', COOLDOWN_TTL)
        self.actionable_verbs = config.get('actionable_verbs', DEFAULT_ACTIONABLE_VERBS)

    # -- Gate checks -----------------------------------------------------------

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
        self.store.zadd(TOPIC_SIGNALS_KEY, {signal_key: now})

        # Clean signals older than 7 days
        week_ago = now - (7 * 86400)
        self.store.zremrangebyscore(TOPIC_SIGNALS_KEY, '-inf', week_ago)

        # Count signals for this topic (from sorted set members starting with topic:)
        all_signals = self.store.zrangebyscore(
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
        """Gate 5: No similar active persistent task (Jaccard + topic match)."""
        try:
            from services.database_service import get_shared_db_service
            from services.persistent_task_service import PersistentTaskService

            db = get_shared_db_service()
            service = PersistentTaskService(db)
            account_id = self._get_account_id()

            # Standard Jaccard check on goal text
            duplicate = service.find_duplicate(account_id, thought.thought_content)
            if duplicate:
                return False

            # Secondary check: same seed topic in scope of any active task.
            # Jaccard misses paraphrased goals (e.g., "research AGI papers"
            # vs "deep dive into AGI development" score only 0.19).
            topic = thought.seed_topic
            if topic and topic != 'general':
                active = service.get_active_tasks(account_id)
                for task in active:
                    scope = task.get('scope', '') or ''
                    if topic.lower() in scope.lower():
                        logger.info(
                            f"{LOG_PREFIX} Topic-duplicate: '{topic}' matches "
                            f"task {task['id']} scope"
                        )
                        return False

            return True
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

    def _cooldown_gate(self) -> Tuple[bool, float]:
        """Gate 7: Adaptive cooldown between plan proposals.

        Base cooldown is 48h, multiplied by the backoff multiplier learned
        from cancellation/completion outcomes. Returns (passes, effective_cooldown_hours).
        """
        backoff = self._get_effective_backoff()
        # The cooldown key is set with TTL = base cooldown seconds.
        # We check if it exists AND if enough time has passed with backoff applied.
        cooldown_val = self.store.get(COOLDOWN_KEY)
        if cooldown_val is None:
            return (True, self.cooldown_seconds * backoff / 3600)

        # Cooldown key exists — check if backoff extends it beyond the TTL.
        # The key was set with base TTL. If backoff > 1.0, the effective cooldown
        # is longer than the TTL, so we check the timestamp stored in the value.
        try:
            set_at = float(cooldown_val)
            elapsed = time.time() - set_at
            effective_cooldown = self.cooldown_seconds * backoff
            if elapsed >= effective_cooldown:
                # Backoff-extended cooldown has passed
                self.store.delete(COOLDOWN_KEY)
                return (True, effective_cooldown / 3600)
        except (ValueError, TypeError):
            pass

        effective_cooldown = self.cooldown_seconds * backoff
        return (False, effective_cooldown / 3600)

    def _get_effective_backoff(self) -> float:
        """Get the current backoff multiplier, applying time-based decay.

        The multiplier decays linearly from its stored value back to 1.0
        over BACKOFF_DECAY_DAYS since the last cancellation.
        """
        raw = self.store.get(BACKOFF_MULTIPLIER_KEY)
        if raw is None:
            return BACKOFF_MIN

        stored_backoff = float(raw)
        if stored_backoff <= BACKOFF_MIN:
            return BACKOFF_MIN

        # Apply time-based decay
        last_cancel_raw = self.store.get(LAST_CANCELLATION_KEY)
        if last_cancel_raw is None:
            return stored_backoff

        try:
            last_cancel_ts = float(last_cancel_raw)
        except (ValueError, TypeError):
            return stored_backoff

        elapsed_days = (time.time() - last_cancel_ts) / 86400
        if elapsed_days >= BACKOFF_DECAY_DAYS:
            # Full decay — reset
            self.store.delete(BACKOFF_MULTIPLIER_KEY)
            self.store.delete(LAST_CANCELLATION_KEY)
            return BACKOFF_MIN

        # Linear decay: backoff approaches 1.0 over BACKOFF_DECAY_DAYS
        decay_progress = elapsed_days / BACKOFF_DECAY_DAYS
        effective = stored_backoff - (stored_backoff - BACKOFF_MIN) * decay_progress
        return max(BACKOFF_MIN, effective)

    # -- Main interface --------------------------------------------------------

    def should_execute(self, thought: ThoughtContext) -> tuple:
        """Evaluate all 7 gates. Returns (score, eligible)."""
        self.last_gate_result = None

        # Gate 1: Thought type
        if not self._thought_type_gate(thought):
            self.last_gate_result = {'gate': 'thought_type', 'reason': f"type '{thought.thought_type}' not in (hypothesis, question)"}
            return (0.0, False)

        # Gate 2: Activation energy
        if not self._activation_gate(thought):
            self.last_gate_result = {'gate': 'activation_energy', 'reason': f"energy {thought.activation_energy:.2f} < {self.min_activation}"}
            return (0.0, False)

        # Gate 3: Signal persistence (always records signal, even if gate fails)
        persistence_passes, signal_count = self._signal_persistence_gate(thought)
        if not persistence_passes:
            self.last_gate_result = {'gate': 'signal_persistence', 'reason': f"signals {signal_count} < {self.min_signals}", 'signal_count': signal_count}
            return (0.0, False)

        # Gate 4: Actionability
        if not self._actionability_gate(thought):
            self.last_gate_result = {'gate': 'actionability', 'reason': 'no action verbs found'}
            return (0.0, False)

        # Gate 5: No duplicate task
        if not self._duplicate_gate(thought):
            self.last_gate_result = {'gate': 'duplicate_task', 'reason': 'similar active task exists'}
            return (0.0, False)

        # Gate 6: Active task count
        if not self._active_count_gate():
            self.last_gate_result = {'gate': 'active_count', 'reason': f"active tasks >= {self.max_active_tasks}"}
            return (0.0, False)

        # Gate 7: Cooldown (adaptive — base 48h * backoff multiplier)
        cooldown_passes, effective_hours = self._cooldown_gate()
        if not cooldown_passes:
            self.last_gate_result = {
                'gate': 'cooldown',
                'reason': f'{effective_hours:.0f}h cooldown active (backoff={self._get_effective_backoff():.1f}x)',
            }
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

            # Auto-accept all drift tasks — the 7 gates are sufficient protection.
            # The old cheap/expensive distinction left most tasks stranded in
            # 'proposed' because the WebSocket confirmation was unreliable.
            task_service.transition(task_id, 'accepted')
            self._surface_auto_start(thought, plan)

            # Set cooldown — store timestamp as value for backoff-extended checks
            self.store.setex(COOLDOWN_KEY, self.cooldown_seconds, str(time.time()))

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

    # -- Outcome learning ------------------------------------------------------

    def on_outcome(self, outcome: str, task_id: Optional[int] = None) -> None:
        """Learn from task outcome by adjusting the backoff multiplier.

        Args:
            outcome: One of 'completed', 'cancelled', 'expired'.
            task_id: Optional task ID for logging context.
        """
        if outcome == 'cancelled':
            self._handle_cancellation(task_id)
        elif outcome == 'completed':
            self._handle_completion(task_id)
        elif outcome == 'expired':
            self._handle_expiry(task_id)
        else:
            logger.debug(f"{LOG_PREFIX} Unknown outcome: {outcome}")

    def _handle_cancellation(self, task_id: Optional[int]) -> None:
        """Increase backoff on cancellation and record negative procedural memory."""
        # Update backoff multiplier
        current = float(self.store.get(BACKOFF_MULTIPLIER_KEY) or BACKOFF_MIN)
        new_backoff = min(current * BACKOFF_INCREASE_FACTOR, BACKOFF_MAX)
        self.store.set(BACKOFF_MULTIPLIER_KEY, str(new_backoff))
        self.store.set(LAST_CANCELLATION_KEY, str(time.time()))

        logger.info(
            f"{LOG_PREFIX} Cancellation backoff: {current:.1f}x -> {new_backoff:.1f}x"
            f" (task {task_id})" if task_id else ""
        )

        # Record negative outcome in procedural memory
        self._record_procedural_outcome(success=False, reward=-0.5, task_id=task_id)

        # Log to interaction_log
        self._log_outcome_event('plan_cancelled', task_id)

    def _handle_completion(self, task_id: Optional[int]) -> None:
        """Decrease backoff on completion and record positive procedural memory."""
        current = float(self.store.get(BACKOFF_MULTIPLIER_KEY) or BACKOFF_MIN)
        new_backoff = max(current * BACKOFF_DECREASE_FACTOR, BACKOFF_MIN)
        if new_backoff < BACKOFF_MIN + 0.01:
            new_backoff = BACKOFF_MIN
        self.store.set(BACKOFF_MULTIPLIER_KEY, str(new_backoff))

        logger.info(
            f"{LOG_PREFIX} Completion backoff: {current:.1f}x -> {new_backoff:.1f}x"
            f" (task {task_id})" if task_id else ""
        )

        # Record positive outcome in procedural memory
        self._record_procedural_outcome(success=True, reward=0.5, task_id=task_id)

        # Log to interaction_log
        self._log_outcome_event('plan_completed', task_id)

    def _handle_expiry(self, task_id: Optional[int]) -> None:
        """Record neutral outcome for expired tasks."""
        self._record_procedural_outcome(success=False, reward=0.0, task_id=task_id)
        self._log_outcome_event('plan_expired', task_id)

    def _record_procedural_outcome(self, success: bool, reward: float,
                                   task_id: Optional[int] = None) -> None:
        """Record outcome in procedural memory (non-fatal)."""
        try:
            from services.database_service import get_shared_db_service
            from services.procedural_memory_service import ProceduralMemoryService

            db = get_shared_db_service()
            proc = ProceduralMemoryService(db)
            proc.record_action_outcome(
                action_name='PLAN',
                success=success,
                reward=reward,
            )
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Procedural memory recording failed: {e}")

    def _log_outcome_event(self, event_type: str, task_id: Optional[int]) -> None:
        """Log outcome to interaction_log (non-fatal)."""
        try:
            from services.database_service import get_shared_db_service
            from services.interaction_log_service import InteractionLogService

            db = get_shared_db_service()
            log_service = InteractionLogService(db)
            log_service.log_event(
                event_type=event_type,
                payload={
                    'task_id': task_id,
                    'backoff_multiplier': float(
                        self.store.get(BACKOFF_MULTIPLIER_KEY) or BACKOFF_MIN
                    ),
                },
                source='plan_action',
            )
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Outcome event logging failed: {e}")

    # -- Helpers ---------------------------------------------------------------

    def _check_conversation_references(self, topic: str) -> int:
        """Count how many distinct conversations mention this topic (last 7 days)."""
        try:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT COUNT(DISTINCT thread_id) FROM interaction_log
                    WHERE topic = ?
                      AND created_at > datetime('now', '-7 days')
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
