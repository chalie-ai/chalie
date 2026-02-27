"""
Persistent Task Skill — Native innate skill for multi-session background tasks.

Provides natural language commands:
  - create: Propose a new persistent task (with plan decomposition)
  - status: Get progress summary
  - plan: Show step-level plan for a task
  - pause: Pause an active task
  - resume: Resume a paused task
  - cancel: Cancel a task
  - expand: Update task scope
  - priority: Change task priority

All DB access via get_shared_db_service() (lazy import inside function).
"""

import logging

logger = logging.getLogger(__name__)
LOG_PREFIX = "[PERSISTENT TASK SKILL]"


def handle_persistent_task(topic: str, params: dict) -> str:
    """
    Dispatch persistent task actions based on params['action'].

    Args:
        topic: Conversation topic for context
        params: Dict with action and parameters

    Returns:
        Human-readable result string
    """
    action = params.get("action", "status").lower()

    if action == "create":
        return _create(topic, params)
    elif action == "status":
        return _status(topic, params)
    elif action == "pause":
        return _pause(topic, params)
    elif action == "resume":
        return _resume(topic, params)
    elif action == "cancel":
        return _cancel(topic, params)
    elif action == "expand":
        return _expand(topic, params)
    elif action == "priority":
        return _priority(topic, params)
    elif action == "plan":
        return _show_plan(topic, params)
    elif action == "list":
        return _list_tasks(topic, params)
    else:
        return f"Unknown persistent task action: {action}"


def _get_service():
    """Lazy-import persistent task service."""
    from services.database_service import get_shared_db_service
    from services.persistent_task_service import PersistentTaskService
    return PersistentTaskService(get_shared_db_service())


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


def _create(topic: str, params: dict) -> str:
    """Create a new persistent task with plan decomposition."""
    goal = params.get("goal", params.get("query", params.get("text", "")))
    if not goal:
        return "No goal specified for the persistent task."

    service = _get_service()
    account_id = _get_account_id()

    # Check for duplicates
    duplicate = service.find_duplicate(account_id, goal)
    if duplicate:
        return (
            f"You already have a similar task in progress: \"{duplicate['goal'][:80]}\"\n"
            f"Status: {duplicate['status']} | Coverage: {duplicate.get('progress', {}).get('coverage_estimate', 0):.0%}\n"
            f"Would you like to continue that task or create a new one?"
        )

    scope = params.get("scope")
    priority = int(params.get("priority", 5))

    task = service.create_task(
        account_id=account_id,
        goal=goal,
        scope=scope,
        priority=priority,
    )
    task_id = task['id']

    # Attempt plan decomposition
    plan = None
    try:
        from services.plan_decomposition_service import PlanDecompositionService
        decomposer = PlanDecompositionService()
        plan = decomposer.decompose(goal, scope)
    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Plan decomposition failed: {e}")

    if plan:
        # Store plan in task progress via checkpoint
        progress = {'plan': plan, 'coverage_estimate': 0.0}
        service.checkpoint(task_id=task_id, progress=progress)

        cost_class = plan.get('cost_class', 'expensive')
        confidence = plan.get('decomposition_confidence', 0.0)
        auto_accept = decomposer.auto_accept_confidence

        step_list = '\n'.join(
            f"  {i+1}. {s['description']}"
            for i, s in enumerate(plan['steps'])
        )

        if cost_class == 'cheap' and confidence >= auto_accept:
            # Auto-accept cheap, high-confidence plans
            service.transition(task_id, 'accepted')
            return (
                f"Task created and auto-started: \"{goal[:80]}\"\n"
                f"Plan ({len(plan['steps'])} steps, confidence {confidence:.0%}):\n"
                f"{step_list}\n"
                f"Working on this in the background."
            )
        else:
            return (
                f"Task proposed: \"{goal[:80]}\"\n"
                f"Plan ({len(plan['steps'])} steps, confidence {confidence:.0%}):\n"
                f"{step_list}\n"
                f"Confirm to start, or adjust the scope."
            )

    # Fallback: no plan (decomposition failed or wasn't possible)
    if scope:
        return (
            f"Task proposed: \"{goal[:80]}\"\n"
            f"Scope: {scope}\n"
            f"I'll work on this in the background. Confirm to start."
        )
    else:
        return (
            f"Task proposed: \"{goal[:80]}\"\n"
            f"Could you help me scope this? For example, time range, specific topics, or constraints?"
        )


def _status(topic: str, params: dict) -> str:
    """Get status of a task."""
    service = _get_service()
    task_id = params.get("task_id")

    if task_id:
        return service.get_status_summary(int(task_id))

    # If no task_id, search by goal text
    goal = params.get("goal", params.get("query", params.get("text", "")))
    account_id = _get_account_id()
    active_tasks = service.get_active_tasks(account_id)

    if not active_tasks:
        return "You don't have any active background tasks."

    if goal:
        # Find best match
        from services.persistent_task_service import _jaccard_similarity
        best_match = max(active_tasks, key=lambda t: _jaccard_similarity(goal, t['goal']))
        if _jaccard_similarity(goal, best_match['goal']) > 0.3:
            return service.get_status_summary(best_match['id'])

    # Return all active task summaries
    summaries = []
    for t in active_tasks:
        progress = t.get('progress', {}) or {}
        coverage = progress.get('coverage_estimate', 0)
        summaries.append(f"- [{t['status']}] \"{t['goal'][:60]}\" ({coverage:.0%} done)")

    return "Active background tasks:\n" + "\n".join(summaries)


def _pause(topic: str, params: dict) -> str:
    """Pause an active task."""
    service = _get_service()
    task_id = _resolve_task_id(params)
    if not task_id:
        return "Could not identify which task to pause."

    ok, msg = service.transition(task_id, 'paused')
    return msg if ok else f"Cannot pause: {msg}"


def _resume(topic: str, params: dict) -> str:
    """Resume a paused task."""
    service = _get_service()
    task_id = _resolve_task_id(params)
    if not task_id:
        return "Could not identify which task to resume."

    ok, msg = service.transition(task_id, 'in_progress')
    return msg if ok else f"Cannot resume: {msg}"


def _cancel(topic: str, params: dict) -> str:
    """Cancel a task."""
    service = _get_service()
    task_id = _resolve_task_id(params)
    if not task_id:
        return "Could not identify which task to cancel."

    ok, msg = service.transition(task_id, 'cancelled')
    return msg if ok else f"Cannot cancel: {msg}"


def _expand(topic: str, params: dict) -> str:
    """Update task scope."""
    service = _get_service()
    task_id = _resolve_task_id(params)
    if not task_id:
        return "Could not identify which task to update."

    new_scope = params.get("scope", params.get("query", params.get("text", "")))
    if not new_scope:
        return "No new scope specified."

    ok, msg = service.update_scope(task_id, new_scope)
    return msg if ok else f"Cannot update scope: {msg}"


def _priority(topic: str, params: dict) -> str:
    """Change task priority."""
    service = _get_service()
    task_id = _resolve_task_id(params)
    if not task_id:
        return "Could not identify which task to update."

    new_priority = int(params.get("priority", params.get("value", 5)))
    ok, msg = service.update_priority(task_id, new_priority)
    return msg if ok else f"Cannot update priority: {msg}"


def _list_tasks(topic: str, params: dict) -> str:
    """List all active tasks."""
    service = _get_service()
    account_id = _get_account_id()
    active_tasks = service.get_active_tasks(account_id)

    if not active_tasks:
        return "You don't have any active background tasks."

    summaries = []
    for t in active_tasks:
        progress = t.get('progress', {}) or {}
        coverage = progress.get('coverage_estimate', 0)
        summaries.append(
            f"- #{t['id']} [{t['status']}] \"{t['goal'][:60]}\" "
            f"({coverage:.0%} done, priority {t['priority']})"
        )

    return "Active background tasks:\n" + "\n".join(summaries)


def _show_plan(topic: str, params: dict) -> str:
    """Show the plan (step DAG) for a task."""
    service = _get_service()
    task_id = _resolve_task_id(params)
    if not task_id:
        return "Could not identify which task to show the plan for."

    task = service.get_task(task_id)
    if not task:
        return "Task not found."

    progress = task.get('progress', {}) or {}
    plan = progress.get('plan')
    if not plan or not plan.get('steps'):
        return f"Task \"{task['goal'][:60]}\" has no step plan (running as flat ACT loop)."

    steps = plan['steps']
    lines = [f"Plan for: \"{task['goal'][:60]}\""]
    lines.append(f"Confidence: {plan.get('decomposition_confidence', 0):.0%} | "
                 f"Cost: {plan.get('cost_class', 'unknown')}")

    if plan.get('blocked_on'):
        lines.append(f"BLOCKED: {plan.get('blocked_reason', 'unknown reason')}")

    for s in steps:
        status_icon = {
            'pending': '○', 'ready': '◎', 'in_progress': '◉',
            'completed': '●', 'skipped': '⊘', 'failed': '✗',
        }.get(s.get('status', 'pending'), '?')

        line = f"  {status_icon} {s['id']}: {s['description']}"
        if s.get('result_summary'):
            line += f" → {s['result_summary'][:60]}"
        if s.get('skip_reason'):
            line += f" (skipped: {s['skip_reason'][:40]})"
        if s.get('failure_reason'):
            line += f" (failed: {s['failure_reason'][:40]})"
        lines.append(line)

    from services.plan_decomposition_service import PlanDecompositionService
    coverage = PlanDecompositionService.get_plan_coverage(plan)
    lines.append(f"Progress: {coverage:.0%} ({sum(1 for s in steps if s.get('status') in ('completed', 'skipped'))}/{len(steps)} steps)")

    return '\n'.join(lines)


def _resolve_task_id(params: dict) -> int | None:
    """Resolve task ID from params — by explicit ID or by goal text match."""
    task_id = params.get("task_id")
    if task_id:
        return int(task_id)

    goal = params.get("goal", params.get("query", params.get("text", "")))
    if not goal:
        return None

    service = _get_service()
    account_id = _get_account_id()
    active_tasks = service.get_active_tasks(account_id)

    if not active_tasks:
        return None

    from services.persistent_task_service import _jaccard_similarity
    best = max(active_tasks, key=lambda t: _jaccard_similarity(goal, t['goal']))
    if _jaccard_similarity(goal, best['goal']) > 0.3:
        return best['id']

    return None
