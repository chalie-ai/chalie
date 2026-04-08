"""
Persistent Task Worker — Three-phase background task execution.

Event-driven: blocks on `persistent_task:execute` queue (blpop).
  Phase 1: PLAN    — Single LLM call, structured JSON plan
  Phase 2: EXECUTE — ACT loop, 200 iterations, no timeout
  Phase 3: SURFACE — Inject result into digest worker

Crash-safe: if worker dies mid-execution, task stays IN_PROGRESS.
"""

import json
import logging

logger = logging.getLogger(__name__)
LOG_PREFIX = "[PERSISTENT TASK WORKER]"

EXECUTE_QUEUE_KEY = "persistent_task:execute"
BLPOP_TIMEOUT = 300  # 5-minute heartbeat fallback

# Skills available in the outer execution loop
OUTER_SKILLS = ['memory', 'find_tools', 'document', 'read']


def persistent_task_worker(shared_state):
    """Background worker — blocks on execute queue, processes tasks."""
    logger.info(f"{LOG_PREFIX} Starting persistent task worker")

    from services.memory_client import MemoryClientService
    store = MemoryClientService.create_connection()

    while True:
        try:
            result = store.blpop([EXECUTE_QUEUE_KEY], timeout=BLPOP_TIMEOUT)

            if result:
                _key, raw = result
                try:
                    signal = json.loads(raw) if isinstance(raw, (str, bytes)) else {}
                    logger.debug(
                        f"{LOG_PREFIX} Signal: task_id={signal.get('task_id')}, "
                        f"reason={signal.get('reason', 'unknown')}"
                    )
                except Exception:
                    pass
            else:
                logger.debug(f"{LOG_PREFIX} Heartbeat — scanning for eligible tasks")

            _run_cycle()
        except Exception as e:
            logger.error(f"{LOG_PREFIX} Worker loop error: {e}", exc_info=True)


def _run_cycle():
    """Pick and process eligible tasks."""
    from services.database_service import get_shared_db_service
    from services.persistent_task_service import PersistentTaskService

    db = get_shared_db_service()
    task_service = PersistentTaskService(db)

    expired = task_service.expire_stale_tasks()
    if expired:
        logger.info(f"{LOG_PREFIX} Expired {expired} stale tasks")

    task = task_service.get_eligible_task()
    if not task:
        logger.debug(f"{LOG_PREFIX} No eligible tasks")
        return

    logger.info(f"{LOG_PREFIX} Processing task {task['id']}: {task['goal'][:80]}")
    _process_task(task_service, task)


def _process_task(task_service, task):
    """Three-phase task execution: Plan → Execute → Surface."""
    task_id = task['id']

    # Transition to in_progress
    if task['status'] == 'accepted':
        ok, msg = task_service.transition(task_id, 'in_progress')
        if not ok:
            logger.warning(f"{LOG_PREFIX} Cannot start task {task_id}: {msg}")
            return

    try:
        # Phase 1: Plan
        logger.info(f"{LOG_PREFIX} Phase 1: Planning task {task_id}")
        plan = _generate_plan(task['goal'])
        task_service.checkpoint(task_id, progress={'plan': plan, 'phase': 'planning_complete'})
        logger.info(f"{LOG_PREFIX} Plan generated: {len(plan.get('steps', []))} steps")

        # Check if paused/cancelled between phases
        task = task_service.get_task(task_id)
        if not task or task['status'] not in ('in_progress',):
            logger.info(f"{LOG_PREFIX} Task {task_id} no longer active after planning")
            return

        # Phase 2: Execute
        logger.info(f"{LOG_PREFIX} Phase 2: Executing task {task_id}")
        result = _execute_with_plan(task, plan)

        # Store result
        final_response = result.final_response or f"Task completed ({result.termination_reason})"
        task_service.complete_task(task_id, final_response)
        logger.info(
            f"{LOG_PREFIX} Task {task_id} completed — "
            f"{result.iterations_used} iterations, reason: {result.termination_reason}"
        )

        # Phase 3: Surface via digest worker
        logger.info(f"{LOG_PREFIX} Phase 3: Surfacing task {task_id}")
        _surface_via_digest(task, final_response)

    except Exception as e:
        logger.error(f"{LOG_PREFIX} Task {task_id} failed: {e}", exc_info=True)
        try:
            task_service.stall_task(task_id, str(e)[:500])
        except Exception as stall_err:
            logger.critical(f"{LOG_PREFIX} Failed to stall task {task_id}: {stall_err}")


def _generate_plan(goal: str) -> dict:
    """Phase 1: Single LLM call to generate a structured plan."""
    import re
    from services.config_service import ConfigService
    from services import FrontalCortexService

    try:
        config = ConfigService.resolve_agent_config("cognitive-triage")
    except Exception:
        config = ConfigService.resolve_agent_config("frontal-cortex")

    prompt_template = ConfigService.get_agent_prompt("persistent-task-plan")
    prompt = prompt_template.replace('{{goal}}', goal)

    cortex = FrontalCortexService(config)

    # Single LLM call — no ACT loop, just generate
    response = cortex.generate_response(
        system_prompt_template=prompt,
        original_prompt=goal,
        classification={'topic': 'persistent_task_planning', 'confidence': 10},
        chat_history=[],
    )

    text = response.get('response', '') if isinstance(response, dict) else str(response)

    try:
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            plan = json.loads(json_match.group())
            if 'steps' in plan:
                return plan
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback: single-step plan
    logger.warning(f"{LOG_PREFIX} Could not parse plan JSON, using single-step fallback")
    return {"steps": [{"id": "s1", "action": goal, "depends_on": [], "parallel": False}]}


def _execute_with_plan(task: dict, plan: dict):
    """Phase 2: Run ACT loop with plan context.

    # TODO: migrate to MessageProcessor tool loop
    # ACTOrchestrator has been deleted. This function needs to be rewired to use
    # MessageProcessor.send() with the new tool loop. Until then it raises to
    # surface the task as stalled rather than silently failing.
    """
    # TODO: migrate to MessageProcessor tool loop
    raise NotImplementedError(
        "ACTOrchestrator has been removed. _execute_with_plan must be rewired "
        "to use MessageProcessor.send() tool loop."
    )


def _surface_via_digest(task: dict, result: str):
    """Phase 3: Inject result into digest worker for natural surfacing."""
    try:
        from workers.digest_worker import digest_worker
        digest_worker(
            text=result,
            metadata={
                'source': 'persistent_task_result',
                'task_id': task['id'],
                'thread_id': task.get('thread_id'),
                'goal': task.get('goal', ''),
            }
        )
    except Exception as e:
        logger.critical(
            f"{LOG_PREFIX} SURFACING FAILED for task {task['id']} — result is stored in DB "
            f"but was not delivered to user: {e}", exc_info=True
        )
