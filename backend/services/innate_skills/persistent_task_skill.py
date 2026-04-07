"""
Persistent Task Skill — Native innate skill for multi-session background tasks.

Provides natural language commands:
  - create: Propose a new persistent task
  - status: Get task status
  - list: List active tasks
  - pause: Pause an active task
  - resume: Resume a paused task
  - cancel: Cancel a task
  - complete: Mark a task as done with a final result

Planning and execution are handled internally by the worker (three-phase:
plan → execute → surface). No plan/expand/progress/priority actions.

All DB access via get_shared_db_service() (lazy import inside function).
"""

import json
import logging

logger = logging.getLogger(__name__)
LOG_PREFIX = "[PERSISTENT TASK SKILL]"

TOOL_SCHEMA = {
    "name": "persistent_task",
    "description": (
        "Multi-session background tasks. Create, monitor, or control tasks that require "
        "deeper work than the current ACT loop can handle. Use when work clearly exceeds "
        "3-5 iterations, requires systematic traversal of many resources, or user asks to "
        "research deeply. Do NOT use when a single action answers the question or work fits "
        "within the current ACT loop."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "create", "status", "list", "pause", "resume",
                    "cancel", "complete",
                ],
                "description": "Task operation to perform.",
            },
            "goal": {
                "type": "string",
                "description": "What to accomplish — clear, specific objective. Required for create; used as fuzzy match for status/pause/resume/cancel when task_id is omitted.",
            },
            "scope": {
                "type": "string",
                "description": "Focus constraints for create — topics, URL patterns, time ranges, depth limits.",
            },
            "task_id": {
                "type": "integer",
                "description": "Explicit task ID for status/pause/resume/cancel/complete.",
            },
            "result": {
                "type": "string",
                "description": "Final result text for complete action.",
            },
            "artifact": {
                "type": "string",
                "description": "Optional artifact data for complete action.",
            },
        },
        "required": ["action"],
    },
}


def handle_persistent_task(channel: str, params: dict) -> str:
    """
    Dispatch persistent task actions based on params['action'].

    Args:
        channel: Conversation channel for context
        params: Dict with action and parameters

    Returns:
        Human-readable result string
    """
    action = params.get("action", "status").lower()

    if action == "create":
        return _create(channel, params)
    elif action == "status":
        return _status(channel, params)
    elif action == "pause":
        return _pause(channel, params)
    elif action == "resume":
        return _resume(channel, params)
    elif action == "cancel":
        return _cancel(channel, params)
    elif action == "list":
        return _list_tasks(channel, params)
    elif action == "complete":
        return _complete(channel, params)
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


def _create(channel: str, params: dict) -> str:
    """Create a new persistent task (starts immediately in accepted state)."""
    goal = params.get("goal", params.get("query", params.get("text", "")))
    if not goal:
        return json.dumps({"success": False, "error": "No goal specified for the persistent task."})

    service = _get_service()
    account_id = _get_account_id()

    # Check for duplicates
    duplicate = service.find_duplicate(account_id, goal)
    if duplicate:
        return json.dumps({
            "success": False,
            "error": (
                f"You already have a similar task in progress: \"{duplicate['goal'][:80]}\". "
                f"Status: {duplicate['status']}. "
                f"Would you like to continue that task or create a new one?"
            ),
        })

    scope = params.get("scope")

    try:
        task = service.create_task(
            account_id=account_id,
            goal=goal,
            scope=scope,
            max_iterations=200,
        )
    except Exception as e:
        logger.error(f"{LOG_PREFIX} Task creation failed: {e}", exc_info=True)
        return json.dumps({"success": False, "error": str(e)})

    task_id = task['id']

    return json.dumps({
        "success": True,
        "task_id": task_id,
        "response": f"Working on \"{goal[:80]}\". This may take a few minutes.",
    })


def _status(channel: str, params: dict) -> str:
    """Get status of a task."""
    service = _get_service()
    task_id = params.get("task_id")

    if task_id:
        return service.get_status_summary(int(task_id))

    goal = params.get("goal", params.get("query", params.get("text", "")))
    account_id = _get_account_id()
    active_tasks = service.get_active_tasks(account_id)

    if not active_tasks:
        return "You don't have any active background tasks."

    if goal:
        from services.persistent_task_service import _jaccard_similarity
        best_match = max(active_tasks, key=lambda t: _jaccard_similarity(goal, t['goal']))
        if _jaccard_similarity(goal, best_match['goal']) > 0.3:
            return service.get_status_summary(best_match['id'])

    summaries = []
    for t in active_tasks:
        summaries.append(f"- [{t['status']}] \"{t['goal'][:60]}\"")

    return "Active background tasks:\n" + "\n".join(summaries)


def _pause(channel: str, params: dict) -> str:
    """Pause an active task."""
    service = _get_service()
    task_id = _resolve_task_id(params)
    if not task_id:
        return "Could not identify which task to pause."

    ok, msg = service.transition(task_id, 'paused')
    return msg if ok else f"Cannot pause: {msg}"


def _resume(channel: str, params: dict) -> str:
    """Resume a paused task."""
    service = _get_service()
    task_id = _resolve_task_id(params)
    if not task_id:
        return "Could not identify which task to resume."

    ok, msg = service.transition(task_id, 'in_progress')
    return msg if ok else f"Cannot resume: {msg}"


def _cancel(channel: str, params: dict) -> str:
    """Cancel a task."""
    service = _get_service()
    task_id = _resolve_task_id(params)
    if not task_id:
        return "Could not identify which task to cancel."

    ok, msg = service.transition(task_id, 'cancelled')
    return msg if ok else f"Cannot cancel: {msg}"


def _list_tasks(channel: str, params: dict) -> str:
    """List all active tasks."""
    service = _get_service()
    account_id = _get_account_id()
    active_tasks = service.get_active_tasks(account_id)

    if not active_tasks:
        return "You don't have any active background tasks."

    summaries = []
    for t in active_tasks:
        summaries.append(f"- #{t['id']} [{t['status']}] \"{t['goal'][:60]}\"")

    return "Active background tasks:\n" + "\n".join(summaries)


def _complete(channel: str, params: dict) -> str:
    """Mark a task as completed with a final result."""
    task_id = _resolve_task_id(params)
    if not task_id:
        task_id = params.get('persistent_task_id')
    if not task_id:
        return "Could not identify which task to complete."

    service = _get_service()
    task = service.get_task(int(task_id))
    if not task:
        return f"Task {task_id} not found."

    result = params.get("result", params.get("text", "Task completed."))
    artifact = params.get("artifact")

    service.complete_task(int(task_id), result, artifact)

    return f"Task #{task_id} \"{task['goal'][:60]}\" marked as completed."


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
