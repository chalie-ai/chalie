"""
Persistent Task Worker — Background processing for multi-session ACT tasks.

Runs on a 30-minute cycle (±30% jitter). Each cycle:
  1. Picks 1 eligible task (FIFO within priority)
  2. Runs ACT loop with task context
  3. Atomic checkpoint after each cycle
  4. Adaptive surfacing of results

Crash-safe: if worker dies mid-cycle, task stays IN_PROGRESS and next
cycle resumes from the last checkpoint.
"""

import json
import time
import random
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)
LOG_PREFIX = "[PERSISTENT TASK WORKER]"

BASE_CYCLE_SECONDS = 1800  # 30 minutes
JITTER_FACTOR = 0.3        # ±30%

# Surfacing thresholds
FIRST_SURFACE_CYCLE = 2           # Surface early after cycle 2
COVERAGE_JUMP_THRESHOLD = 0.15    # Surface if coverage jumped >15%


def persistent_task_worker(shared_state):
    """
    Background service process for persistent task processing.

    Args:
        shared_state: Shared state dict from multiprocessing.Manager
    """
    logger.info(f"{LOG_PREFIX} Starting persistent task worker")

    while True:
        try:
            _run_cycle()
        except Exception as e:
            logger.error(f"{LOG_PREFIX} Cycle error: {e}", exc_info=True)

        # Sleep with jitter
        jitter = random.uniform(1 - JITTER_FACTOR, 1 + JITTER_FACTOR)
        sleep_time = BASE_CYCLE_SECONDS * jitter
        logger.debug(f"{LOG_PREFIX} Sleeping {sleep_time:.0f}s until next cycle")
        time.sleep(sleep_time)


def _run_cycle():
    """Execute one processing cycle: pick task → ACT loop → checkpoint."""
    from services.database_service import get_shared_db_service
    from services.persistent_task_service import PersistentTaskService

    db = get_shared_db_service()
    task_service = PersistentTaskService(db)

    # Auto-expire stale tasks
    expired = task_service.expire_stale_tasks()
    if expired:
        logger.info(f"{LOG_PREFIX} Expired {expired} stale tasks")

    # Pick eligible task
    task = task_service.get_eligible_task()
    if not task:
        logger.debug(f"{LOG_PREFIX} No eligible tasks")
        return

    task_id = task['id']
    logger.info(f"{LOG_PREFIX} Processing task {task_id}: {task['goal'][:80]}")

    # Rate limit check
    if not task_service.check_rate_limit(task_id):
        logger.info(f"{LOG_PREFIX} Task {task_id} rate-limited (max {3}/hr)")
        return

    # Transition to IN_PROGRESS if needed
    if task['status'] == 'accepted':
        ok, msg = task_service.transition(task_id, 'in_progress')
        if not ok:
            logger.warning(f"{LOG_PREFIX} Cannot start task {task_id}: {msg}")
            return

    # Run the ACT loop for this task
    progress = task.get('progress', {}) or {}
    prev_coverage = progress.get('coverage_estimate', 0.0)

    try:
        result_data = _execute_task_act_loop(task)
    except Exception as e:
        logger.error(f"{LOG_PREFIX} ACT loop failed for task {task_id}: {e}", exc_info=True)
        # Crash-safe: task stays IN_PROGRESS, checkpoint is NOT updated
        return

    # Merge progress update into existing progress (preserves plan, etc.)
    new_progress = dict(progress)
    new_progress.update(result_data.get('progress_update', {}))
    cycles_completed = progress.get('cycles_completed', 0) + 1
    new_progress['cycles_completed'] = cycles_completed
    new_progress['last_cycle_at'] = datetime.now(timezone.utc).isoformat()
    new_progress['cycles_this_hour'] = progress.get('cycles_this_hour', 0) + 1

    # Atomic checkpoint
    task_service.checkpoint(
        task_id=task_id,
        progress=new_progress,
        result_fragment=result_data.get('result_fragment'),
    )

    # Schedule next run
    task_service.set_next_run(task_id, BASE_CYCLE_SECONDS)

    # Check if task is complete
    if result_data.get('task_complete', False):
        final_result = new_progress.get('last_summary', 'Task completed.')
        task_service.complete_task(task_id, final_result, result_data.get('artifact'))
        _surface_completion(task, final_result)
        logger.info(f"{LOG_PREFIX} Task {task_id} completed!")
        return

    # Adaptive surfacing
    new_coverage = new_progress.get('coverage_estimate', prev_coverage)
    coverage_jump = new_coverage - prev_coverage

    should_surface = (
        (cycles_completed == FIRST_SURFACE_CYCLE) or
        (coverage_jump > COVERAGE_JUMP_THRESHOLD)
    )

    if should_surface:
        _surface_progress(task, new_progress)

    logger.info(
        f"{LOG_PREFIX} Task {task_id} cycle {cycles_completed} complete "
        f"(coverage: {prev_coverage:.0%} → {new_coverage:.0%})"
    )


def _execute_task_act_loop(task: dict) -> dict:
    """
    Run a bounded ACT loop for a persistent task.

    Branches to plan-aware execution if the task has a step DAG,
    otherwise falls through to the flat ACT loop.

    Returns dict with:
      - progress_update: dict
      - task_complete: bool
      - result_fragment: str (optional)
      - artifact: dict (optional)
    """
    progress = task.get('progress', {}) or {}
    plan = progress.get('plan')

    if plan and plan.get('steps'):
        return _execute_plan_aware_loop(task, plan)
    else:
        return _execute_flat_loop(task)


def _execute_flat_loop(task: dict) -> dict:
    """
    Original flat ACT loop — no step DAG, just iterate toward the goal.

    Preserved as-is for backward compatibility with planless tasks.
    """
    from services.config_service import ConfigService
    from services import FrontalCortexService
    from services.act_loop_service import ActLoopService

    # Load config
    try:
        cortex_config = ConfigService.resolve_agent_config("frontal-cortex-act")
    except Exception:
        cortex_config = ConfigService.resolve_agent_config("frontal-cortex")

    act_prompt = ConfigService.get_agent_prompt("persistent-task-act")

    cortex_service = FrontalCortexService(cortex_config)

    # Initialize ACT loop with task's fatigue budget
    act_loop = ActLoopService(
        config={**cortex_config, 'fatigue_budget': task.get('fatigue_budget', 15.0)},
        cumulative_timeout=120.0,  # 2min per cycle (generous for background work)
        per_action_timeout=15.0,
        max_iterations=5,          # Bounded per cycle
    )

    # Build task context for prompt
    progress = task.get('progress', {}) or {}
    task_context = {
        'task_goal': task['goal'],
        'task_scope': task.get('scope', 'No specific scope defined'),
        'task_progress': json.dumps(progress, indent=2) if progress else 'No previous progress',
        'task_intermediate_results': task.get('result', 'None yet'),
    }

    # Build prompt with task context
    prompt_filled = act_prompt
    for key, value in task_context.items():
        prompt_filled = prompt_filled.replace(f'{{{{{key}}}}}', str(value))

    # ACT loop — limited to 5 iterations per cycle
    last_response = {}
    while True:
        can_continue, reason = act_loop.can_continue()
        if not can_continue:
            break

        response_data = cortex_service.generate_response(
            system_prompt_template=prompt_filled,
            original_prompt=task['goal'],
            classification={'topic': f"task_{task['id']}", 'confidence': 10},
            chat_history=[],
            act_history=act_loop.get_history_context(),
        )

        last_response = response_data
        actions = response_data.get('actions', [])

        if not actions:
            break

        results = act_loop.execute_actions(
            topic=f"persistent_task_{task['id']}",
            actions=actions,
        )
        act_loop.append_results(results)
        act_loop.accumulate_fatigue(results, act_loop.iteration_number)
        act_loop.iteration_number += 1

    # Extract progress_update from last response
    progress_update = last_response.get('progress_update', {})
    if not progress_update:
        progress_update = {
            'last_summary': f"Cycle completed with {act_loop.iteration_number} iterations",
            'coverage_estimate': progress.get('coverage_estimate', 0.0),
        }

    return {
        'progress_update': progress_update,
        'task_complete': last_response.get('task_complete', False),
        'result_fragment': None,
        'artifact': None,
    }


# ── Plan-Aware Execution ─────────────────────────────────────────────

MAX_STEPS_PER_CYCLE = 3
MAX_ITERATIONS_PER_STEP = 3
MIN_PER_STEP_BUDGET = 3.0


def _execute_plan_aware_loop(task: dict, plan: dict) -> dict:
    """
    Execute steps from the plan DAG in dependency order.

    Each cycle processes up to MAX_STEPS_PER_CYCLE ready steps, each with
    a bounded ACT loop (MAX_ITERATIONS_PER_STEP iterations, fractional
    fatigue budget).
    """
    from services.plan_decomposition_service import PlanDecompositionService

    ready_steps = PlanDecompositionService.get_ready_steps(plan)

    # All done?
    all_statuses = [s.get('status') for s in plan.get('steps', [])]
    resolved = {'completed', 'skipped'}

    if not ready_steps and all(st in resolved for st in all_statuses):
        # Gather final summary from all step results
        results = [
            s.get('result_summary', '')
            for s in plan['steps']
            if s.get('status') == 'completed' and s.get('result_summary')
        ]
        summary = ' | '.join(results) if results else 'All steps completed.'
        coverage = PlanDecompositionService.get_plan_coverage(plan)

        return {
            'progress_update': {
                'plan': plan,
                'coverage_estimate': coverage,
                'last_summary': summary,
            },
            'task_complete': True,
            'result_fragment': None,
            'artifact': None,
        }

    # Blocked?
    if not ready_steps:
        plan = PlanDecompositionService._update_blocked_state(plan)
        coverage = PlanDecompositionService.get_plan_coverage(plan)
        blocked_reason = plan.get('blocked_reason', 'unknown')

        logger.warning(
            f"{LOG_PREFIX} Task {task['id']} plan blocked: {blocked_reason}"
        )

        return {
            'progress_update': {
                'plan': plan,
                'coverage_estimate': coverage,
                'last_summary': f"Blocked: {blocked_reason}",
            },
            'task_complete': False,
            'result_fragment': None,
            'artifact': None,
        }

    # Cap ready steps per cycle
    steps_to_run = ready_steps[:MAX_STEPS_PER_CYCLE]
    fatigue_budget = task.get('fatigue_budget', 15.0)
    per_step_budget = max(MIN_PER_STEP_BUDGET, fatigue_budget / len(steps_to_run))

    logger.info(
        f"{LOG_PREFIX} Task {task['id']}: executing {len(steps_to_run)} steps "
        f"(budget {per_step_budget:.1f} each)"
    )

    for step in steps_to_run:
        try:
            plan = _execute_single_step(task, plan, step, per_step_budget)
        except Exception as e:
            logger.error(
                f"{LOG_PREFIX} Step {step['id']} failed: {e}", exc_info=True
            )
            plan = PlanDecompositionService.update_step_status(
                plan, step['id'], 'failed',
                failure_reason=str(e)[:200],
                retryable=True,
            )

    coverage = PlanDecompositionService.get_plan_coverage(plan)
    completed_steps = [
        s for s in plan['steps']
        if s.get('status') in ('completed', 'skipped')
    ]
    last_completed = completed_steps[-1] if completed_steps else None
    summary = (
        last_completed.get('result_summary', 'Step completed')
        if last_completed else 'Processing steps...'
    )

    return {
        'progress_update': {
            'plan': plan,
            'coverage_estimate': coverage,
            'last_summary': summary,
        },
        'task_complete': False,
        'result_fragment': None,
        'artifact': None,
    }


def _execute_single_step(task: dict, plan: dict, step: dict,
                         budget: float) -> dict:
    """
    Run a bounded ACT loop for a single plan step.

    Returns the updated plan dict with the step's status mutated.
    """
    from services.config_service import ConfigService
    from services import FrontalCortexService
    from services.act_loop_service import ActLoopService
    from services.plan_decomposition_service import PlanDecompositionService

    step_id = step['id']
    logger.info(f"{LOG_PREFIX} Executing step {step_id}: {step['description'][:60]}")

    # Mark step as in_progress
    plan = PlanDecompositionService.update_step_status(plan, step_id, 'in_progress')

    # Load config
    try:
        cortex_config = ConfigService.resolve_agent_config("frontal-cortex-act")
    except Exception:
        cortex_config = ConfigService.resolve_agent_config("frontal-cortex")

    step_prompt = ConfigService.get_agent_prompt("plan-step-act")
    cortex_service = FrontalCortexService(cortex_config)

    # Build step-specific ACT loop
    act_loop = ActLoopService(
        config={**cortex_config, 'fatigue_budget': budget},
        cumulative_timeout=60.0,
        per_action_timeout=15.0,
        max_iterations=MAX_ITERATIONS_PER_STEP,
    )

    # Gather dependency results
    dep_results = _gather_dependency_results(plan, step)

    # Gather remaining steps info
    remaining = [
        f"- {s['id']}: {s['description']} [{s.get('status', 'pending')}]"
        for s in plan['steps']
        if s['id'] != step_id and s.get('status') not in ('completed', 'skipped', 'failed')
    ]

    # Fill prompt template
    prompt_filled = step_prompt
    prompt_filled = prompt_filled.replace('{{task_goal}}', task['goal'])
    prompt_filled = prompt_filled.replace('{{step_description}}', step['description'])
    prompt_filled = prompt_filled.replace('{{step_dependencies_results}}', dep_results or 'No prior results (this is a root step).')
    prompt_filled = prompt_filled.replace('{{remaining_steps}}', '\n'.join(remaining) or 'This is the last step.')
    prompt_filled = prompt_filled.replace('{{available_skills}}', cortex_service._get_available_skills())
    prompt_filled = prompt_filled.replace('{{available_tools}}', cortex_service._get_available_tools())
    prompt_filled = prompt_filled.replace('{{client_context}}', 'Background worker execution')

    # Run step ACT loop
    last_response = {}
    while True:
        can_continue, reason = act_loop.can_continue()
        if not can_continue:
            break

        response_data = cortex_service.generate_response(
            system_prompt_template=prompt_filled,
            original_prompt=step['description'],
            classification={'topic': f"task_{task['id']}_step_{step_id}", 'confidence': 10},
            chat_history=[],
            act_history=act_loop.get_history_context(),
        )

        last_response = response_data
        progress_update = response_data.get('progress_update', {})

        # Check for skip signal
        if progress_update.get('step_skip'):
            skip_reason = progress_update.get('skip_reason', 'Deemed unnecessary by LLM')
            plan = PlanDecompositionService.update_step_status(
                plan, step_id, 'skipped',
                skip_reason=skip_reason,
                skipped_by='llm',
            )
            logger.info(f"{LOG_PREFIX} Step {step_id} skipped: {skip_reason}")
            return plan

        # Check for step completion
        if progress_update.get('step_complete'):
            result_summary = progress_update.get('step_result', 'Step completed')
            plan = PlanDecompositionService.update_step_status(
                plan, step_id, 'completed',
                result_summary=result_summary,
            )
            logger.info(f"{LOG_PREFIX} Step {step_id} completed: {result_summary[:60]}")
            return plan

        # Execute actions
        actions = response_data.get('actions', [])
        if not actions:
            break

        results = act_loop.execute_actions(
            topic=f"persistent_task_{task['id']}",
            actions=actions,
        )
        act_loop.append_results(results)
        act_loop.accumulate_fatigue(results, act_loop.iteration_number)
        act_loop.iteration_number += 1

    # Iterations exhausted without explicit completion — save partial result
    progress_update = last_response.get('progress_update', {})
    step_result = progress_update.get('step_result', f"Partial after {act_loop.iteration_number} iterations")
    plan = PlanDecompositionService.update_step_status(
        plan, step_id, 'completed',
        result_summary=step_result,
    )
    logger.info(f"{LOG_PREFIX} Step {step_id} completed (iterations exhausted): {step_result[:60]}")

    return plan


def _gather_dependency_results(plan: dict, step: dict) -> str:
    """Gather result summaries from completed/skipped dependency steps."""
    dep_ids = step.get('depends_on', [])
    if not dep_ids:
        return ''

    step_map = {s['id']: s for s in plan.get('steps', [])}
    lines = []

    for dep_id in dep_ids:
        dep = step_map.get(dep_id)
        if not dep:
            continue

        status = dep.get('status', 'unknown')
        if status == 'completed':
            result = dep.get('result_summary', 'No summary')
            lines.append(f"[{dep_id}] (completed): {result}")
        elif status == 'skipped':
            reason = dep.get('skip_reason', 'No reason')
            lines.append(f"[{dep_id}] (skipped): {reason}")

    return '\n'.join(lines)


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
