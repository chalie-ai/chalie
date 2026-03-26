"""
Persistent Task Worker — Background processing for multi-session ACT tasks.

Event-driven: blocks on `persistent_task:execute` queue (blpop).
  - On signal: pick 1 eligible task → run organic ACT loop → checkpoint
  - On 5-min timeout: fallback scan for any eligible tasks

Every task runs the same organic ACT loop — no mandatory pre-planning, no
plan-aware vs flat distinction. If the LLM decides it needs a plan, it can
use the `persistent_task` innate skill to decompose the goal organically.

Crash-safe: if worker dies mid-cycle, task stays IN_PROGRESS and the next
wake resumes from the last checkpoint.
"""

import json
import logging
from datetime import datetime, timezone

from services.time_utils import utc_now

logger = logging.getLogger(__name__)
LOG_PREFIX = "[PERSISTENT TASK WORKER]"

EXECUTE_QUEUE_KEY = "persistent_task:execute"
BLPOP_TIMEOUT = 300  # 5-minute heartbeat fallback


def persistent_task_worker(shared_state):
    """
    Background worker for persistent task processing.

    Blocks on the `persistent_task:execute` queue. When a signal arrives
    (task accepted, resumed, etc.) or the 5-min timeout fires, picks one
    eligible task and runs an organic ACT loop.

    Args:
        shared_state: Shared state dict from WorkerManager
    """
    logger.info(f"{LOG_PREFIX} Starting persistent task worker (event-driven)")

    from services.memory_store import get_shared_store
    store = get_shared_store()

    while True:
        try:
            # Block until a signal arrives or timeout fires
            result = store.blpop([EXECUTE_QUEUE_KEY], timeout=BLPOP_TIMEOUT)

            if result:
                _key, raw = result
                try:
                    signal = json.loads(raw) if isinstance(raw, (str, bytes)) else {}
                    task_id = signal.get('task_id')
                    logger.debug(
                        f"{LOG_PREFIX} Signal received — task_id={task_id}, "
                        f"reason={signal.get('reason', 'unknown')}"
                    )
                except Exception:
                    pass
            else:
                logger.debug(f"{LOG_PREFIX} Heartbeat timeout — scanning for eligible tasks")

            _run_cycle()

        except Exception as e:
            logger.error(f"{LOG_PREFIX} Worker loop error: {e}", exc_info=True)


def _run_cycle():
    """Process all eligible tasks, running each to completion."""
    from services.database_service import get_shared_db_service
    from services.persistent_task_service import PersistentTaskService

    db = get_shared_db_service()
    task_service = PersistentTaskService(db)

    # Auto-expire stale tasks
    expired = task_service.expire_stale_tasks()
    if expired:
        logger.info(f"{LOG_PREFIX} Expired {expired} stale tasks")

    # Process eligible tasks one by one
    processed = set()
    while True:
        task = task_service.get_eligible_task()
        if not task:
            logger.debug(f"{LOG_PREFIX} No eligible tasks")
            return
        if task['id'] in processed:
            return  # Already ran this task this cycle

        logger.info(f"{LOG_PREFIX} Processing task {task['id']}: {task['goal'][:80]}")
        _process_task(task_service, task)
        processed.add(task['id'])


def _process_task(task_service, task):
    """Run continuous ACT cycles on a task until complete or naturally stopped.

    Steps run back-to-back — each cycle's results feed into the next so the
    DAG plan can adapt.  The worker only pauses when the ACT loop exits
    naturally (LLM decided it's done) or on error.
    """
    task_id = task['id']
    MAX_CONTINUOUS_CYCLES = 300  # Safety valve — not a throttle

    # Transition to IN_PROGRESS if needed
    if task['status'] == 'accepted':
        ok, msg = task_service.transition(task_id, 'in_progress')
        if not ok:
            logger.warning(f"{LOG_PREFIX} Cannot start task {task_id}: {msg}")
            return

    for _cycle_num in range(MAX_CONTINUOUS_CYCLES):
        # Refresh task from DB — may have been completed by innate skill
        task = task_service.get_task(task_id)
        if not task or task['status'] not in ('accepted', 'in_progress'):
            logger.info(
                f"{LOG_PREFIX} Task {task_id} no longer active "
                f"(status={task['status'] if task else 'deleted'})"
            )
            return

        # Respect iteration budget
        if task.get('iterations_used', 0) >= task.get('max_iterations', 20):
            logger.info(
                f"{LOG_PREFIX} Task {task_id} exhausted iteration budget "
                f"({task['iterations_used']}/{task['max_iterations']})"
            )
            task_service.stall_task(task_id, 'Iteration budget exhausted')
            return

        progress = task.get('progress', {}) or {}
        prev_coverage = progress.get('coverage_estimate', 0.0)

        # Run one ACT loop cycle
        try:
            result_data = _execute_task_act_loop(task)
        except Exception as e:
            logger.error(f"{LOG_PREFIX} ACT loop failed for task {task_id}: {e}", exc_info=True)
            return  # Crash-safe: task stays IN_PROGRESS

        # Merge progress
        new_progress = dict(progress)
        new_progress.update(result_data.get('progress_update', {}))
        cycles_completed = progress.get('cycles_completed', 0) + 1
        new_progress['cycles_completed'] = cycles_completed
        new_progress['last_cycle_at'] = utc_now().isoformat()

        # Atomic checkpoint
        task_service.checkpoint(
            task_id=task_id,
            progress=new_progress,
            result_fragment=result_data.get('result_fragment'),
        )

        # Safety check: checkpoint may have auto-stalled the task
        refreshed_after_cp = task_service.get_task(task_id)
        if refreshed_after_cp and refreshed_after_cp['status'] == 'stalled':
            logger.info(f"{LOG_PREFIX} Task {task_id} auto-stalled by checkpoint")
            return

        # Check if task is complete
        if result_data.get('task_complete', False):
            final_result = new_progress.get('last_summary', 'Task completed.')
            task_service.complete_task(task_id, final_result, result_data.get('artifact'))
            _surface_completion(task, final_result)
            logger.info(f"{LOG_PREFIX} Task {task_id} completed!")
            return

        # Check if completed by innate skill during the ACT loop
        refreshed = task_service.get_task(task_id)
        if refreshed and refreshed['status'] in ('completed', 'cancelled'):
            logger.info(f"{LOG_PREFIX} Task {task_id} {refreshed['status']} during ACT loop")
            return

        # Adaptive surfacing
        new_coverage = new_progress.get('coverage_estimate', prev_coverage)
        coverage_jump = new_coverage - prev_coverage
        if (cycles_completed == 2) or (coverage_jump > 0.15):
            _surface_progress(task, new_progress)

        # Decide whether to continue immediately
        termination = result_data.get('termination_reason', '')
        if termination in ('natural_exit', 'no_actions'):
            # LLM decided it's done for now — stop cycling
            logger.info(
                f"{LOG_PREFIX} Task {task_id} cycle {cycles_completed} — "
                f"natural stop ({termination}), coverage: {new_coverage:.0%}"
            )
            return

        # max_iterations or timeout — more work to do, continue immediately
        logger.info(
            f"{LOG_PREFIX} Task {task_id} cycle {cycles_completed} "
            f"(coverage: {prev_coverage:.0%} → {new_coverage:.0%}) — continuing"
        )

    logger.warning(
        f"{LOG_PREFIX} Task {task_id} hit safety limit ({MAX_CONTINUOUS_CYCLES} cycles)"
    )


def _build_task_chat_history(task):
    progress = task.get('progress', {}) or {}
    result_text = task.get('result', '') or ''
    history = []

    if progress.get('last_summary'):
        cycles = progress.get('cycles_completed', 0)
        history.append({
            'role': 'assistant',
            'content': f"[Previous cycle {cycles}] {progress['last_summary']}"
        })

    if result_text:
        truncated = result_text[-3000:] if len(result_text) > 3000 else result_text
        if truncated != result_text:
            truncated = '...' + truncated
        history.append({
            'role': 'assistant',
            'content': f"Intermediate findings from prior cycles:\n{truncated}"
        })

    return history


def _execute_task_act_loop(task: dict) -> dict:
    """
    Run the organic ACT loop for a persistent task.

    No plan decomposition is forced — the LLM gets the goal as context and
    decides what to do. If it needs a structured plan, it can invoke the
    `persistent_task` innate skill to decompose the goal.

    Returns dict with:
      - progress_update: dict
      - task_complete: bool
      - result_fragment: str (optional)
      - artifact: dict (optional)
    """
    from services.config_service import ConfigService
    from services import FrontalCortexService
    from services.act_orchestrator_service import ACTOrchestrator
    from services.innate_skills.registry import COGNITIVE_PRIMITIVES_ORDERED, SKILL_DESCRIPTIONS
    from services.plan_decomposition_service import PlanDecompositionService

    # Load config
    try:
        cortex_config = ConfigService.resolve_agent_config("frontal-cortex-act")
    except Exception:
        cortex_config = ConfigService.resolve_agent_config("frontal-cortex")

    act_prompt = ConfigService.get_agent_prompt("persistent-task-act")
    cortex_service = FrontalCortexService(cortex_config)

    # Retrieve cognitive context for this task
    assembled_context = None
    try:
        from services.context_assembly_service import ContextAssemblyService
        cas = ContextAssemblyService({})
        assembled_context = cas.assemble(
            prompt=task['goal'],
            topic=f"persistent_task_{task['id']}",
        )
    except Exception as e:
        logger.debug(f"{LOG_PREFIX} Context assembly failed (non-fatal): {e}")

    # Build task context for prompt
    progress = task.get('progress', {}) or {}
    task_context = {
        'task_id': str(task['id']),
        'task_goal': task['goal'],
        'task_scope': task.get('scope', 'No specific scope defined'),
        'task_progress': json.dumps(progress, indent=2) if progress else 'No previous progress',
        'task_intermediate_results': task.get('result', 'None yet'),
    }

    # Fill skill/tool/context placeholders
    skills_text = '\n'.join(
        f'- **{name}**: {SKILL_DESCRIPTIONS.get(name, "")}'
        for name in COGNITIVE_PRIMITIVES_ORDERED
    )
    tools_text = PlanDecompositionService._get_available_tools()
    task_context['injected_skills'] = skills_text
    task_context['available_tools'] = tools_text
    task_context['client_context'] = 'Background worker execution (persistent task cycle)'

    # Fill prompt template
    prompt_filled = act_prompt
    for key, value in task_context.items():
        prompt_filled = prompt_filled.replace(f'{{{{{key}}}}}', str(value))

    orchestrator = ACTOrchestrator(
        config=cortex_config,
        max_iterations=50,
        cumulative_timeout=120.0,
        per_action_timeout=15.0,
        critic_enabled=True,
        smart_repetition=True,
        escalation_hints=False,
        persistent_task_exit=False,
        execution_gate=False,  # User already approved the task by creating it
    )

    result = orchestrator.run(
        topic=f"persistent_task_{task['id']}",
        text=task['goal'],
        cortex_service=cortex_service,
        act_prompt=prompt_filled,
        classification={'topic': f"task_{task['id']}", 'confidence': 10},
        chat_history=_build_task_chat_history(task),
        session_id='persistent_task',
        exchange_id=f"ptask_{task['id']}",
        assembled_context=assembled_context,
        context_extras={'persistent_task_id': task['id']},
    )

    progress_update = {
        'last_summary': (
            f"Cycle completed with {result.iterations_used} iterations "
            f"(termination: {result.termination_reason})"
        ),
        'coverage_estimate': progress.get('coverage_estimate', 0.0),
    }

    return {
        'progress_update': progress_update,
        'task_complete': False,  # Completion determined by task skill or next cycle
        'result_fragment': None,
        'artifact': None,
        'termination_reason': result.termination_reason,
    }


def _surface_progress(task: dict, progress: dict):
    """Surface a progress update to the user via the communicate pipeline."""
    try:
        from services.output_service import OutputService
        output = OutputService()
        summary = progress.get('last_summary', 'Making progress...')
        coverage = progress.get('coverage_estimate', 0)

        message = (
            f"Update on your task \"{task['goal'][:60]}\" — "
            f"{summary} ({coverage:.0%} coverage)"
        )
        output.enqueue_proactive(task.get('thread_id'), message, source='persistent_task')
        logger.info(f"{LOG_PREFIX} Surfaced progress for task {task['id']}")
    except Exception as e:
        logger.debug(f"{LOG_PREFIX} Progress surfacing failed: {e}")


def _surface_completion(task: dict, result: str):
    """Surface task completion to the user."""
    try:
        from services.output_service import OutputService
        output = OutputService()
        message = (
            f"I've finished working on \"{task['goal'][:60]}\". "
            f"Here's what I found:\n\n{result}"
        )
        output.enqueue_proactive(task.get('thread_id'), message, source='persistent_task')
        logger.info(f"{LOG_PREFIX} Surfaced completion for task {task['id']}")
    except Exception as e:
        logger.debug(f"{LOG_PREFIX} Completion surfacing failed: {e}")
