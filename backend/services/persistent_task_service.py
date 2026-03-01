"""
Persistent Task Service — Multi-session ACT task management.

State machine:
  PROPOSED → ACCEPTED → IN_PROGRESS → COMPLETED
                ↓           ↓
             CANCELLED    PAUSED → IN_PROGRESS
                            ↓
                         CANCELLED
  Auto-expiry: ACCEPTED/IN_PROGRESS/PAUSED → EXPIRED (14 days default)

Bounded scope enforcement: open-ended goals must be scoped before ACCEPTED.
Duplicate detection: Jaccard similarity > 0.6 against active tasks.
Fairness: max 3 cycles per task per hour, max 5 active tasks per user.
"""

import json
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)
LOG_PREFIX = "[PERSISTENT TASK]"

# Valid state transitions
VALID_TRANSITIONS = {
    'proposed': {'accepted', 'cancelled'},
    'accepted': {'in_progress', 'cancelled'},
    'in_progress': {'completed', 'paused', 'cancelled', 'expired'},
    'paused': {'in_progress', 'cancelled', 'expired'},
}

# States eligible for auto-expiry
EXPIRABLE_STATES = {'accepted', 'in_progress', 'paused'}

# Limits
MAX_ACTIVE_TASKS = 5
MAX_CYCLES_PER_HOUR = 3
DEFAULT_EXPIRY_DAYS = 14
DEFAULT_MAX_ITERATIONS = 20
DEFAULT_FATIGUE_BUDGET = 15.0

# Duplicate detection threshold (Jaccard similarity)
DUPLICATE_SIMILARITY_THRESHOLD = 0.6


from utils.text_utils import jaccard_similarity as _jaccard_similarity


class PersistentTaskService:
    """Manages persistent task lifecycle, scope enforcement, and duplicate detection."""

    def __init__(self, db_service):
        self.db = db_service

    # ── CRUD ──────────────────────────────────────────────────────────

    def create_task(
        self,
        account_id: int,
        goal: str,
        thread_id: Optional[int] = None,
        scope: Optional[str] = None,
        priority: int = 5,
        deadline: Optional[str] = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        fatigue_budget: float = DEFAULT_FATIGUE_BUDGET,
    ) -> Dict[str, Any]:
        """
        Create a new persistent task in PROPOSED state.

        Returns the created task dict.
        """
        expires_at = datetime.now(timezone.utc) + timedelta(days=DEFAULT_EXPIRY_DAYS)

        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO persistent_tasks
                    (account_id, thread_id, goal, scope, status, priority,
                     max_iterations, fatigue_budget, expires_at, deadline)
                VALUES (%s, %s, %s, %s, 'proposed', %s, %s, %s, %s, %s)
                RETURNING id, created_at
            """, (
                account_id, thread_id, goal, scope, priority,
                max_iterations, fatigue_budget, expires_at,
                deadline,
            ))
            row = cursor.fetchone()
            conn.commit()

        task_id = row[0]
        logger.info(f"{LOG_PREFIX} Created task {task_id}: {goal[:80]}")

        return self.get_task(task_id)

    def get_task(self, task_id: int) -> Optional[Dict[str, Any]]:
        """Get a single task by ID."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, account_id, thread_id, goal, scope, status, priority,
                       progress, result, result_artifact, iterations_used,
                       max_iterations, created_at, updated_at, expires_at,
                       deadline, next_run_after, fatigue_budget
                FROM persistent_tasks
                WHERE id = %s
            """, (task_id,))
            row = cursor.fetchone()

        if not row:
            return None

        return self._row_to_dict(row)

    def get_active_tasks(self, account_id: int) -> List[Dict[str, Any]]:
        """Get all non-terminal tasks for an account."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, account_id, thread_id, goal, scope, status, priority,
                       progress, result, result_artifact, iterations_used,
                       max_iterations, created_at, updated_at, expires_at,
                       deadline, next_run_after, fatigue_budget
                FROM persistent_tasks
                WHERE account_id = %s
                  AND status IN ('proposed', 'accepted', 'in_progress', 'paused')
                ORDER BY priority ASC, created_at ASC
            """, (account_id,))
            rows = cursor.fetchall()

        return [self._row_to_dict(r) for r in rows]

    def get_eligible_task(self) -> Optional[Dict[str, Any]]:
        """
        Pick the next eligible task for background processing.

        Eligibility: ACCEPTED or IN_PROGRESS, not expired, next_run_after passed.
        FIFO within priority (oldest first).
        """
        now = datetime.now(timezone.utc)
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, account_id, thread_id, goal, scope, status, priority,
                       progress, result, result_artifact, iterations_used,
                       max_iterations, created_at, updated_at, expires_at,
                       deadline, next_run_after, fatigue_budget
                FROM persistent_tasks
                WHERE status IN ('accepted', 'in_progress')
                  AND expires_at > %s
                  AND (next_run_after IS NULL OR next_run_after <= %s)
                  AND iterations_used < max_iterations
                ORDER BY priority ASC, created_at ASC
                LIMIT 1
            """, (now, now))
            row = cursor.fetchone()

        if not row:
            return None

        return self._row_to_dict(row)

    # ── State Transitions ────────────────────────────────────────────

    def transition(self, task_id: int, new_status: str) -> Tuple[bool, str]:
        """
        Transition a task to a new state.

        Returns (success, message).
        """
        task = self.get_task(task_id)
        if not task:
            return False, f"Task {task_id} not found"

        current = task['status']
        valid = VALID_TRANSITIONS.get(current, set())

        if new_status not in valid:
            return False, f"Cannot transition from '{current}' to '{new_status}'"

        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE persistent_tasks
                SET status = %s, updated_at = NOW()
                WHERE id = %s
            """, (new_status, task_id))
            conn.commit()

        logger.info(f"{LOG_PREFIX} Task {task_id}: {current} → {new_status}")
        return True, f"Task transitioned to {new_status}"

    def accept_task(self, task_id: int, scope: Optional[str] = None) -> Tuple[bool, str]:
        """Accept a proposed task, optionally setting scope."""
        task = self.get_task(task_id)
        if not task:
            return False, "Task not found"

        # Check active task limit
        active = self.get_active_tasks(task['account_id'])
        active_count = sum(1 for t in active if t['status'] in ('accepted', 'in_progress'))
        if active_count >= MAX_ACTIVE_TASKS:
            return False, f"Maximum {MAX_ACTIVE_TASKS} active tasks reached"

        if scope:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE persistent_tasks SET scope = %s WHERE id = %s
                """, (scope, task_id))
                conn.commit()

        return self.transition(task_id, 'accepted')

    def update_scope(self, task_id: int, new_scope: str) -> Tuple[bool, str]:
        """
        Update task scope mid-execution (goal drift handling).

        Retains previous progress — evaluates overlap for delta processing.
        """
        task = self.get_task(task_id)
        if not task:
            return False, "Task not found"

        if task['status'] not in ('accepted', 'in_progress', 'paused'):
            return False, f"Cannot update scope in '{task['status']}' state"

        old_scope = task.get('scope', '') or ''
        overlap = _jaccard_similarity(old_scope, new_scope) if old_scope else 0.0

        progress = task.get('progress', {}) or {}
        if overlap < 0.3:
            # Minimal overlap → soft restart
            progress['coverage_estimate'] = 0.0
            progress['last_summary'] = f"Scope updated (low overlap). Previous: {progress.get('last_summary', 'none')}"
            logger.info(f"{LOG_PREFIX} Task {task_id}: scope update with soft restart (overlap={overlap:.2f})")
        else:
            # Significant overlap → delta processing
            progress['last_summary'] = f"Scope expanded. {progress.get('last_summary', '')}"
            logger.info(f"{LOG_PREFIX} Task {task_id}: scope update with delta processing (overlap={overlap:.2f})")

        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE persistent_tasks
                SET scope = %s, progress = %s, updated_at = NOW()
                WHERE id = %s
            """, (new_scope, json.dumps(progress), task_id))
            conn.commit()

        return True, f"Scope updated (overlap: {overlap:.0%})"

    def update_priority(self, task_id: int, new_priority: int) -> Tuple[bool, str]:
        """Update task priority (1=highest, 10=lowest)."""
        new_priority = max(1, min(10, new_priority))
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE persistent_tasks
                SET priority = %s, updated_at = NOW()
                WHERE id = %s
            """, (new_priority, task_id))
            conn.commit()
        return True, f"Priority set to {new_priority}"

    # ── Checkpoint & Completion ──────────────────────────────────────

    def checkpoint(
        self,
        task_id: int,
        progress: Dict[str, Any],
        result_fragment: Optional[str] = None,
    ) -> bool:
        """
        Atomic checkpoint after a processing cycle.

        Saves progress state so the next cycle can resume.
        """
        with self.db.connection() as conn:
            cursor = conn.cursor()

            updates = [
                "progress = %s",
                "iterations_used = iterations_used + 1",
                "updated_at = NOW()",
            ]
            params = [json.dumps(progress)]

            if result_fragment:
                updates.append("result = COALESCE(result, '') || %s")
                params.append(result_fragment)

            params.append(task_id)
            cursor.execute(
                f"UPDATE persistent_tasks SET {', '.join(updates)} WHERE id = %s",
                params,
            )
            conn.commit()

        logger.debug(f"{LOG_PREFIX} Checkpoint for task {task_id}")
        return True

    def complete_task(self, task_id: int, result: str, artifact: Optional[Dict] = None) -> bool:
        """Mark a task as completed with final result."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE persistent_tasks
                SET status = 'completed', result = %s, result_artifact = %s,
                    updated_at = NOW()
                WHERE id = %s
            """, (result, json.dumps(artifact) if artifact else None, task_id))
            conn.commit()

        logger.info(f"{LOG_PREFIX} Task {task_id} completed")
        return True

    def set_next_run(self, task_id: int, delay_seconds: int):
        """Schedule the next run with jitter."""
        import random
        jitter = random.uniform(0.7, 1.3)
        delay = int(delay_seconds * jitter)
        next_run = datetime.now(timezone.utc) + timedelta(seconds=delay)

        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE persistent_tasks
                SET next_run_after = %s, updated_at = NOW()
                WHERE id = %s
            """, (next_run, task_id))
            conn.commit()

    def check_rate_limit(self, task_id: int) -> bool:
        """Check if a task has exceeded MAX_CYCLES_PER_HOUR."""
        task = self.get_task(task_id)
        if not task:
            return False

        progress = task.get('progress', {}) or {}
        last_cycle_at = progress.get('last_cycle_at')
        cycles_this_hour = progress.get('cycles_this_hour', 0)

        if last_cycle_at:
            try:
                last = datetime.fromisoformat(last_cycle_at)
                elapsed = (datetime.now(timezone.utc) - last).total_seconds()
                if elapsed < 3600 and cycles_this_hour >= MAX_CYCLES_PER_HOUR:
                    return False  # Rate limited
                if elapsed >= 3600:
                    # Reset hourly counter
                    progress['cycles_this_hour'] = 0
            except (ValueError, TypeError):
                pass

        return True

    # ── Duplicate Detection ──────────────────────────────────────────

    def find_duplicate(self, account_id: int, goal: str) -> Optional[Dict[str, Any]]:
        """
        Check for duplicate tasks using Jaccard similarity.

        Returns the matching task if similarity > threshold, else None.
        """
        active_tasks = self.get_active_tasks(account_id)

        for task in active_tasks:
            similarity = _jaccard_similarity(goal, task['goal'])
            if similarity > DUPLICATE_SIMILARITY_THRESHOLD:
                logger.info(
                    f"{LOG_PREFIX} Duplicate detected: '{goal[:50]}' ~ '{task['goal'][:50]}' "
                    f"(similarity={similarity:.2f})"
                )
                return task

        return None

    # ── Auto-Expiry ──────────────────────────────────────────────────

    def expire_stale_tasks(self) -> int:
        """Expire tasks past their expires_at timestamp. Returns count expired."""
        now = datetime.now(timezone.utc)
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE persistent_tasks
                SET status = 'expired', updated_at = NOW()
                WHERE status IN ('accepted', 'in_progress', 'paused')
                  AND expires_at <= %s
                RETURNING id
            """, (now,))
            expired_ids = [r[0] for r in cursor.fetchall()]
            conn.commit()

        if expired_ids:
            logger.info(f"{LOG_PREFIX} Expired {len(expired_ids)} tasks: {expired_ids}")

        return len(expired_ids)

    # ── Progress Summary ─────────────────────────────────────────────

    def get_status_summary(self, task_id: int) -> str:
        """Get a human-readable status summary for a task."""
        task = self.get_task(task_id)
        if not task:
            return "Task not found."

        progress = task.get('progress', {}) or {}
        status = task['status']
        goal = task['goal']

        summary = f"Task: {goal}\nStatus: {status}"

        if status == 'completed':
            result = task.get('result', 'No result recorded')
            summary += f"\nResult: {result}"
        elif status in ('in_progress', 'accepted'):
            last_summary = progress.get('last_summary', 'Not started yet')
            coverage = progress.get('coverage_estimate', 0)
            cycles = progress.get('cycles_completed', 0)
            summary += (
                f"\nProgress: {last_summary}"
                f"\nCoverage: {coverage:.0%}"
                f"\nCycles completed: {cycles}"
                f"\nIterations used: {task['iterations_used']}/{task['max_iterations']}"
            )
        elif status == 'paused':
            summary += "\nThis task is paused. Say 'resume' to continue."

        return summary

    # ── Helpers ───────────────────────────────────────────────────────

    def _row_to_dict(self, row) -> Dict[str, Any]:
        """Convert a database row tuple to a task dict."""
        return {
            'id': row[0],
            'account_id': row[1],
            'thread_id': row[2],
            'goal': row[3],
            'scope': row[4],
            'status': row[5],
            'priority': row[6],
            'progress': row[7] if isinstance(row[7], dict) else (json.loads(row[7]) if row[7] else {}),
            'result': row[8],
            'result_artifact': row[9] if isinstance(row[9], dict) else (json.loads(row[9]) if row[9] else None),
            'iterations_used': row[10],
            'max_iterations': row[11],
            'created_at': str(row[12]) if row[12] else None,
            'updated_at': str(row[13]) if row[13] else None,
            'expires_at': str(row[14]) if row[14] else None,
            'deadline': str(row[15]) if row[15] else None,
            'next_run_after': str(row[16]) if row[16] else None,
            'fatigue_budget': row[17],
        }
