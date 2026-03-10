"""
World State Service — Deterministic salience-based context aggregator.

Surfaces temporally and semantically relevant signals from:
- Scheduled items (reminders, upcoming events)
- Persistent tasks (active goals, recently completed work)
- Active ACT loop steps (in-thread work)

Zero LLM. Scoring is deterministic: temporal_proximity * W_t + semantic_similarity * W_s
"""

import json
import logging
import math
import struct

from services.time_utils import utc_now, parse_utc

logger = logging.getLogger(__name__)
LOG_PREFIX = "[WORLD STATE]"

# Salience weights
W_TEMPORAL = 0.4
W_SEMANTIC = 0.6
SALIENCE_THRESHOLD = 0.15  # Below this, item is not salient enough to surface

# Temporal decay constants
TEMPORAL_HALF_LIFE_HOURS = 12.0   # Score halves every 12 hours into the future
TEMPORAL_PAST_DECAY_HOURS = 24.0  # Completed items decay over 24 hours

# Limits
MAX_WORLD_STATE_ITEMS = 5
MAX_SCHEDULED_CANDIDATES = 10
MAX_TASK_CANDIDATES = 10
MAX_LIST_CANDIDATES = 10


class WorldStateService:
    """Deterministic salience aggregator for world state context."""

    def __init__(self, db=None, **kwargs):
        self._db = db

    def _get_db(self):
        if self._db:
            return self._db
        from services.database_service import get_shared_db_service
        return get_shared_db_service()

    def get_world_state(
        self,
        topic: str,
        thread_id: str = None,
        message_embedding: list = None,
    ) -> str:
        """
        Generate world state context from salient signals.

        Args:
            topic: Current topic (unused, kept for API compat)
            thread_id: Thread ID for in-thread ACT step lookup
            message_embedding: Embedding of current message for semantic scoring.
                               When None, falls back to temporal-only scoring.

        Returns:
            str: Formatted world state block (empty string if nothing is salient)
        """
        items = []

        # 1. Active ACT steps (always high salience when present)
        if thread_id:
            items.extend(self._get_active_steps(thread_id))

        # 2. Scheduled items (temporal + optional semantic)
        items.extend(self._get_salient_scheduled_items(message_embedding))

        # 3. Persistent tasks (temporal + optional semantic)
        items.extend(self._get_salient_tasks(message_embedding))

        # 4. Lists (temporal + semantic)
        items.extend(self._get_salient_lists(message_embedding))

        if not items:
            return ""

        # Sort by salience descending, cap at MAX_WORLD_STATE_ITEMS
        items.sort(key=lambda x: x['salience'], reverse=True)
        items = items[:MAX_WORLD_STATE_ITEMS]

        return self._format_world_state(items)

    # ── Signal Collectors ────────────────────────────────────────────────────

    def _get_active_steps(self, thread_id: str) -> list:
        """Get in-flight ACT loop steps — always treated as maximally salient."""
        try:
            from services.thread_conversation_service import ThreadConversationService
            conv_service = ThreadConversationService()
            active_steps = conv_service.get_active_steps(thread_id)
            return [
                {
                    'type': 'active_step',
                    'label': (
                        f"[{s.get('status', 'pending').upper()}] "
                        f"{s.get('type', 'task')}: {s.get('description', 'Unknown')}"
                    ),
                    'salience': 1.0,
                }
                for s in (active_steps or [])
            ]
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Active steps unavailable: {e}")
            return []

    def _get_salient_scheduled_items(self, message_embedding: list = None) -> list:
        """Retrieve scheduled items scored by temporal + semantic salience."""
        try:
            db = self._get_db()
            now = utc_now()
            items = []

            with db.connection() as conn:
                cursor = conn.cursor()
                # Pending items in the next 7 days, plus recently fired items
                cursor.execute("""
                    SELECT id, message, due_at, status, item_type, recurrence
                    FROM scheduled_items
                    WHERE (status = 'pending' AND due_at <= datetime(?, '+7 days'))
                       OR (status = 'fired' AND last_fired_at >= datetime(?, '-24 hours'))
                    ORDER BY due_at ASC
                    LIMIT ?
                """, (now.isoformat(), now.isoformat(), MAX_SCHEDULED_CANDIDATES))
                rows = cursor.fetchall()

                for row in rows:
                    item_id, message, due_at_str, status, item_type, recurrence = row

                    due_at = parse_utc(due_at_str)
                    temporal = self._temporal_score(now, due_at, status == 'fired')

                    semantic = 0.0
                    if message_embedding:
                        semantic = self._semantic_score_scheduled(
                            conn, item_id, message_embedding
                        )

                    salience = W_TEMPORAL * temporal + W_SEMANTIC * semantic

                    if salience >= SALIENCE_THRESHOLD:
                        if status == 'fired':
                            label = f"[DONE] {message}"
                        else:
                            time_str = self._relative_time(now, due_at)
                            recur_suffix = (
                                f" (recurring: {recurrence})" if recurrence else ""
                            )
                            label = f"[{time_str}] {message}{recur_suffix}"

                        items.append({
                            'type': 'scheduled',
                            'label': label,
                            'salience': salience,
                        })

            return items
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Scheduled items unavailable: {e}")
            return []

    def _get_salient_tasks(self, message_embedding: list = None) -> list:
        """Retrieve persistent tasks scored by temporal + semantic salience."""
        try:
            db = self._get_db()
            now = utc_now()
            items = []

            with db.connection() as conn:
                cursor = conn.cursor()
                # Active tasks + recently completed (last 48 hours)
                cursor.execute("""
                    SELECT id, goal, status, progress, updated_at, deadline
                    FROM persistent_tasks
                    WHERE status IN ('active', 'running', 'paused', 'accepted', 'in_progress')
                       OR (status = 'completed' AND updated_at >= datetime(?, '-48 hours'))
                    ORDER BY updated_at DESC
                    LIMIT ?
                """, (now.isoformat(), MAX_TASK_CANDIDATES))
                rows = cursor.fetchall()

                for row in rows:
                    task_id, goal, status, progress_json, updated_at_str, deadline_str = row

                    # Temporal score: prefer deadline if available
                    if deadline_str:
                        deadline = parse_utc(deadline_str)
                        temporal = self._temporal_score(
                            now, deadline, status == 'completed'
                        )
                    elif status == 'completed':
                        updated = parse_utc(updated_at_str)
                        temporal = self._past_decay_score(now, updated)
                    else:
                        # Active task with no deadline — moderate baseline salience
                        temporal = 0.5

                    semantic = 0.0
                    if message_embedding:
                        semantic = self._semantic_score_task(
                            conn, task_id, message_embedding
                        )

                    salience = W_TEMPORAL * temporal + W_SEMANTIC * semantic

                    if salience >= SALIENCE_THRESHOLD:
                        progress = (
                            json.loads(progress_json)
                            if progress_json
                            else {}
                        )
                        coverage = progress.get('coverage_estimate', 0)

                        if status == 'completed':
                            label = f"[COMPLETED] {goal[:80]}"
                        else:
                            deadline_hint = ""
                            if deadline_str:
                                deadline_dt = parse_utc(deadline_str)
                                deadline_hint = (
                                    f" — due {self._relative_time(now, deadline_dt)}"
                                )
                            label = (
                                f"[{status.upper()}] {goal[:80]} "
                                f"({coverage:.0%}){deadline_hint}"
                            )

                        items.append({
                            'type': 'task',
                            'label': label,
                            'salience': salience,
                        })

            return items
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Tasks unavailable: {e}")
            return []

    def _get_salient_lists(self, message_embedding: list = None) -> list:
        """Retrieve lists scored by temporal recency + semantic salience."""
        try:
            db = self._get_db()
            now = utc_now()
            items = []

            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT
                        l.id,
                        l.name,
                        l.updated_at,
                        SUM(CASE WHEN li.removed_at IS NULL AND li.id IS NOT NULL THEN 1 ELSE 0 END) AS item_count,
                        SUM(CASE WHEN li.removed_at IS NULL AND li.checked THEN 1 ELSE 0 END) AS checked_count
                    FROM lists l
                    LEFT JOIN list_items li ON li.list_id = l.id
                    WHERE l.deleted_at IS NULL
                      AND l.updated_at >= datetime(?, '-7 days')
                    GROUP BY l.id, l.name, l.updated_at
                    ORDER BY l.updated_at DESC
                    LIMIT ?
                """, (now.isoformat(), MAX_LIST_CANDIDATES))
                rows = cursor.fetchall()

                for row in rows:
                    list_id, name, updated_at_str, item_count, checked_count = row

                    updated_at = parse_utc(updated_at_str)
                    temporal = self._past_decay_score(now, updated_at)

                    semantic = 0.0
                    if message_embedding:
                        semantic = self._semantic_score_list(
                            conn, list_id, message_embedding
                        )

                    salience = W_TEMPORAL * temporal + W_SEMANTIC * semantic

                    if salience >= SALIENCE_THRESHOLD:
                        item_count = item_count or 0
                        checked_count = checked_count or 0
                        time_str = self._relative_time(now, updated_at)
                        if checked_count > 0:
                            count_str = f"{item_count} items, {checked_count} checked"
                        else:
                            count_str = f"{item_count} items"
                        label = f"[LIST] {name} ({count_str}) — updated {time_str}"

                        items.append({
                            'type': 'list',
                            'label': label,
                            'salience': salience,
                        })

            return items
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Lists unavailable: {e}")
            return []

    # ── Scoring Functions (deterministic, zero LLM) ──────────────────────────

    @staticmethod
    def _temporal_score(now, target_dt, is_past: bool = False) -> float:
        """
        Score based on temporal proximity.

        Future items: exponential decay from 1.0 as they get further away.
        Past/fired items: exponential decay over TEMPORAL_PAST_DECAY_HOURS.
        """
        delta_hours = (target_dt - now).total_seconds() / 3600.0

        if delta_hours < 0 or is_past:
            hours_ago = abs(delta_hours)
            return max(
                0.0,
                math.exp(-0.693 * hours_ago / TEMPORAL_PAST_DECAY_HOURS)
            )
        else:
            return max(
                0.0,
                math.exp(-0.693 * delta_hours / TEMPORAL_HALF_LIFE_HOURS)
            )

    @staticmethod
    def _past_decay_score(now, event_dt) -> float:
        """Score for completed/past items — decays over TEMPORAL_PAST_DECAY_HOURS."""
        hours_ago = (now - event_dt).total_seconds() / 3600.0
        if hours_ago < 0:
            return 0.5
        return max(
            0.0,
            math.exp(-0.693 * hours_ago / TEMPORAL_PAST_DECAY_HOURS)
        )

    def _semantic_score_scheduled(
        self, conn, item_id: str, message_embedding: list
    ) -> float:
        """
        Cosine similarity between the current message and a scheduled item.

        sqlite-vec stores embeddings; we query KNN and check whether the target
        rowid is in the results.  Falls back to 0.0 on any error.
        """
        try:
            packed = struct.pack(f'{len(message_embedding)}f', *message_embedding)
            cursor = conn.cursor()

            cursor.execute(
                "SELECT rowid FROM scheduled_items WHERE id = ?", (item_id,)
            )
            row = cursor.fetchone()
            if not row:
                return 0.0
            item_rowid = row[0]

            # KNN search: retrieve up to MAX_SCHEDULED_CANDIDATES nearest neighbours
            cursor.execute("""
                SELECT rowid, distance
                FROM scheduled_items_vec
                WHERE embedding MATCH ? AND k = ?
            """, (packed, MAX_SCHEDULED_CANDIDATES))

            for vec_row in cursor.fetchall():
                if vec_row[0] == item_rowid:
                    distance = vec_row[1]
                    # sqlite-vec cosine distance → similarity
                    return max(0.0, 1.0 - distance)

            return 0.0
        except Exception as e:
            logger.debug(
                f"{LOG_PREFIX} Semantic score failed for scheduled item {item_id}: {e}"
            )
            return 0.0

    def _semantic_score_task(
        self, conn, task_id: int, message_embedding: list
    ) -> float:
        """
        Cosine similarity between the current message and a persistent task goal.

        Falls back to 0.0 on any error.
        """
        try:
            packed = struct.pack(f'{len(message_embedding)}f', *message_embedding)
            cursor = conn.cursor()

            # KNN search: retrieve up to MAX_TASK_CANDIDATES nearest neighbours
            cursor.execute("""
                SELECT rowid, distance
                FROM persistent_tasks_vec
                WHERE embedding MATCH ? AND k = ?
            """, (packed, MAX_TASK_CANDIDATES))

            for vec_row in cursor.fetchall():
                if vec_row[0] == task_id:
                    distance = vec_row[1]
                    return max(0.0, 1.0 - distance)

            return 0.0
        except Exception as e:
            logger.debug(
                f"{LOG_PREFIX} Semantic score failed for task {task_id}: {e}"
            )
            return 0.0

    def _semantic_score_list(
        self, conn, list_id: str, message_embedding: list
    ) -> float:
        """
        Cosine similarity between the current message and a list name embedding.

        Falls back to 0.0 on any error.
        """
        try:
            packed = struct.pack(f'{len(message_embedding)}f', *message_embedding)
            cursor = conn.cursor()

            cursor.execute(
                "SELECT rowid FROM lists WHERE id = ?", (list_id,)
            )
            row = cursor.fetchone()
            if not row:
                return 0.0
            list_rowid = row[0]

            # KNN search: retrieve up to MAX_LIST_CANDIDATES nearest neighbours
            cursor.execute("""
                SELECT rowid, distance
                FROM lists_vec
                WHERE embedding MATCH ? AND k = ?
            """, (packed, MAX_LIST_CANDIDATES))

            for vec_row in cursor.fetchall():
                if vec_row[0] == list_rowid:
                    distance = vec_row[1]
                    return max(0.0, 1.0 - distance)

            return 0.0
        except Exception as e:
            logger.debug(
                f"{LOG_PREFIX} Semantic score failed for list {list_id}: {e}"
            )
            return 0.0

    # ── Formatting ───────────────────────────────────────────────────────────

    @staticmethod
    def _relative_time(now, target_dt) -> str:
        """Human-readable relative time string (e.g. 'in 3h', '2d ago')."""
        delta = target_dt - now
        total_minutes = delta.total_seconds() / 60.0

        if total_minutes < 0:
            mins_ago = abs(total_minutes)
            if mins_ago < 60:
                return f"{int(mins_ago)}m ago"
            hours_ago = mins_ago / 60
            if hours_ago < 24:
                return f"{int(hours_ago)}h ago"
            return f"{int(hours_ago / 24)}d ago"
        else:
            if total_minutes < 60:
                return f"in {int(total_minutes)}m"
            hours = total_minutes / 60
            if hours < 24:
                return f"in {int(hours)}h"
            days = hours / 24
            return f"in {int(days)}d"

    @staticmethod
    def _format_world_state(items: list) -> str:
        """Format salient items into a prompt-ready text block."""
        lines = ["\n## World State"]
        for item in items:
            lines.append(f"- {item['label']}")
        return "\n".join(lines)
