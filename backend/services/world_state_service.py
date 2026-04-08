"""
World State Service — Deterministic salience-based context aggregator.

Surfaces temporally and semantically relevant signals from:
- Scheduled items (reminders, upcoming events)
- Persistent tasks (active goals, recently completed work)
- Active ACT loop steps (in-thread work)

Zero LLM. Scoring is deterministic: temporal_proximity * W_t + semantic_similarity * W_s

M5 enhancement: continuously maintained via signal-driven MemoryStore cache.
Cache is refreshed during idle periods (refresh_model()) and updated in-place
by notify_task_changed() / notify_schedule_changed() on state transitions.
Ambient context, active topics, and reasoning focus are also surfaced.
"""

import json
import logging
import math

from services.embedding_utils import pack_embedding
import time

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
MAX_EXTERNAL_SIGNAL_CANDIDATES = 10

# External signal storage + decay
EXTERNAL_SIGNALS_KEY = "world_state:external_signals"
MAX_EXTERNAL_SIGNALS = 100  # Capped list in MemoryStore
EXTERNAL_SIGNAL_DECAY_HOURS = 6.0  # Half-life for external signal temporal decay

# Cache keys + TTL
WORLD_MODEL_KEY = "world_model:items"
WORLD_MODEL_TTL = 600   # 10 minutes — MemoryStore expiry
CACHE_MAX_AGE = 300     # 5 minutes — refresh from DB if older than this


class WorldStateService:
    """Deterministic salience aggregator for world state context."""

    def __init__(self, db=None, **kwargs):
        self._db = db
        self._store = None

    def _get_db(self):
        if self._db:
            return self._db
        from services.database_service import get_shared_db_service
        return get_shared_db_service()

    def _get_store(self):
        """Lazy-initialize the shared MemoryStore connection."""
        if self._store is None:
            from services.memory_client import MemoryClientService
            self._store = MemoryClientService.create_connection()
        return self._store

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

        # 0. User telemetry (time, location, device, energy, attention, mobility)
        telemetry_line = self._get_user_telemetry()
        if telemetry_line:
            items.append({'label': telemetry_line, 'salience': 100})

        # 1. Active ACT steps (always high salience when present)
        if thread_id:
            items.extend(self._get_active_steps(thread_id))

        # 2–4. Scheduled items, tasks, lists — try cache first, fall back to DB
        cache_items = self._get_items_from_cache(message_embedding)
        if cache_items is not None:
            items.extend(cache_items)
        else:
            db_items = []
            db_items.extend(self._get_salient_scheduled_items(message_embedding))
            db_items.extend(self._get_salient_tasks(message_embedding))
            db_items.extend(self._get_salient_lists(message_embedding))
            items.extend(self._collapse_items(db_items))

        # 5. External signals (from paired interfaces — zero LLM)
        items.extend(self._get_external_signals(message_embedding))

        # 8. Engagement signal (low/high engagement from proactive response scoring)
        items.extend(self._get_engagement_signal())

        if not items:
            return ""

        # Sort by salience descending, cap at MAX_WORLD_STATE_ITEMS
        items.sort(key=lambda x: x['salience'], reverse=True)
        items = items[:MAX_WORLD_STATE_ITEMS]

        return self._format_world_state(items)

    # ── Cache Layer ───────────────────────────────────────────────────────────

    def refresh_model(self) -> None:
        """
        Refresh the world model cache from the database.

        Queries all three data sources (scheduled items, tasks, lists) and
        writes the raw item data to MemoryStore. Called by the reasoning loop
        during idle periods. Fail-open: any error is logged at debug level.
        """
        try:
            db = self._get_db()
            now = utc_now()
            payload: dict = {
                'refreshed_at': time.time(),
                'scheduled_items': [],
                'persistent_tasks': [],
                'lists': [],
            }

            with db.connection() as conn:
                cursor = conn.cursor()

                # Scheduled items — same window as the live query
                cursor.execute("""
                    SELECT id, message, due_at, status, item_type, recurrence, metadata
                    FROM scheduled_items
                    WHERE ((status = 'pending' AND due_at <= datetime(?, '+7 days'))
                       OR (status = 'fired' AND last_fired_at >= datetime(?, '-24 hours')))
                      AND COALESCE(hidden, 0) = 0
                    ORDER BY due_at ASC
                    LIMIT ?
                """, (now.isoformat(), now.isoformat(), MAX_SCHEDULED_CANDIDATES))
                for row in cursor.fetchall():
                    item_id, message, due_at_str, status, item_type, recurrence, metadata = row
                    payload['scheduled_items'].append({
                        'id': item_id,
                        'message': message,
                        'due_at': due_at_str,
                        'status': status,
                        'item_type': item_type,
                        'recurrence': recurrence,
                        'metadata': metadata,
                    })

                # Persistent tasks
                cursor.execute("""
                    SELECT id, goal, status, progress, updated_at, deadline
                    FROM persistent_tasks
                    WHERE status IN ('active', 'running', 'paused', 'accepted', 'in_progress')
                       OR (status = 'completed' AND updated_at >= datetime(?, '-48 hours'))
                    ORDER BY updated_at DESC
                    LIMIT ?
                """, (now.isoformat(), MAX_TASK_CANDIDATES))
                for row in cursor.fetchall():
                    task_id, goal, status, progress_json, updated_at_str, deadline_str = row
                    payload['persistent_tasks'].append({
                        'id': task_id,
                        'goal': goal,
                        'status': status,
                        'progress': progress_json,
                        'updated_at': updated_at_str,
                        'deadline': deadline_str,
                    })

                # Lists
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
                for row in cursor.fetchall():
                    list_id, name, updated_at_str, item_count, checked_count = row
                    payload['lists'].append({
                        'id': list_id,
                        'name': name,
                        'updated_at': updated_at_str,
                        'item_count': item_count or 0,
                        'checked_count': checked_count or 0,
                    })

            store = self._get_store()
            store.setex(WORLD_MODEL_KEY, WORLD_MODEL_TTL, json.dumps(payload))
            logger.debug(
                f"{LOG_PREFIX} Cache refreshed: "
                f"{len(payload['scheduled_items'])} scheduled, "
                f"{len(payload['persistent_tasks'])} tasks, "
                f"{len(payload['lists'])} lists"
            )
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} refresh_model failed (non-fatal): {e}")

    def notify_task_changed(
        self,
        task_id,
        goal: str,
        status: str,
        progress_json,
        updated_at,
        deadline,
    ) -> None:
        """
        Update the cached world model when a persistent task changes state.

        If the cache doesn't exist, do nothing — the next refresh_model() call
        will populate it. Fail-open.

        Args:
            task_id: Integer or string task identifier.
            goal: Task goal text.
            status: New task status string.
            progress_json: Progress JSON string or dict (may be None).
            updated_at: Updated-at timestamp (float, string, or datetime).
            deadline: Deadline timestamp (may be None).
        """
        try:
            store = self._get_store()
            raw = store.get(WORLD_MODEL_KEY)
            if not raw:
                return

            payload = json.loads(raw)
            tasks = payload.get('persistent_tasks', [])

            # Normalise task_id for comparison
            try:
                tid = int(task_id) if task_id is not None else None
            except (TypeError, ValueError):
                tid = task_id

            # Serialise updated_at and deadline to ISO strings if needed
            def _to_iso(val):
                if val is None:
                    return None
                if isinstance(val, str):
                    return val
                if isinstance(val, (int, float)):
                    from datetime import datetime, timezone
                    return datetime.fromtimestamp(val, tz=timezone.utc).isoformat()
                if hasattr(val, 'isoformat'):
                    return val.isoformat()
                return str(val)

            updated_entry = {
                'id': tid,
                'goal': goal or '',
                'status': status or '',
                'progress': progress_json if isinstance(progress_json, str) else (
                    json.dumps(progress_json) if progress_json else None
                ),
                'updated_at': _to_iso(updated_at),
                'deadline': _to_iso(deadline),
            }

            # Update existing entry or append new one
            found = False
            for i, t in enumerate(tasks):
                try:
                    existing_id = int(t.get('id')) if t.get('id') is not None else None
                except (TypeError, ValueError):
                    existing_id = t.get('id')
                if existing_id == tid:
                    tasks[i] = updated_entry
                    found = True
                    break

            if not found:
                tasks.append(updated_entry)

            payload['persistent_tasks'] = tasks
            # Preserve remaining TTL — re-set with full TTL is acceptable here
            store.setex(WORLD_MODEL_KEY, WORLD_MODEL_TTL, json.dumps(payload))
            logger.debug(f"{LOG_PREFIX} notify_task_changed: task {tid} → {status}")
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} notify_task_changed failed (non-fatal): {e}")

    def notify_schedule_changed(
        self,
        item_id,
        message: str,
        status: str,
        due_at,
    ) -> None:
        """
        Update the cached world model when a scheduled item changes state.

        If the cache doesn't exist, do nothing — the next refresh_model() call
        will populate it. Fail-open.

        Args:
            item_id: UUID string for the scheduled item.
            message: Schedule item message text.
            status: New status string (e.g. 'fired', 'pending').
            due_at: Due timestamp (string, float, or datetime; may be None).
        """
        try:
            store = self._get_store()
            raw = store.get(WORLD_MODEL_KEY)
            if not raw:
                return

            payload = json.loads(raw)
            items = payload.get('scheduled_items', [])

            def _to_iso(val):
                if val is None:
                    return None
                if isinstance(val, str):
                    return val
                if isinstance(val, (int, float)):
                    from datetime import datetime, timezone
                    return datetime.fromtimestamp(val, tz=timezone.utc).isoformat()
                if hasattr(val, 'isoformat'):
                    return val.isoformat()
                return str(val)

            updated_entry = {
                'id': item_id,
                'message': message or '',
                'due_at': _to_iso(due_at),
                'status': status or '',
                'item_type': None,
                'recurrence': None,
            }

            found = False
            for i, item in enumerate(items):
                if item.get('id') == item_id:
                    # Preserve item_type and recurrence if already cached
                    updated_entry['item_type'] = item.get('item_type')
                    updated_entry['recurrence'] = item.get('recurrence')
                    items[i] = updated_entry
                    found = True
                    break

            if not found:
                items.append(updated_entry)

            payload['scheduled_items'] = items
            store.setex(WORLD_MODEL_KEY, WORLD_MODEL_TTL, json.dumps(payload))
            logger.debug(
                f"{LOG_PREFIX} notify_schedule_changed: item {item_id} → {status}"
            )
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} notify_schedule_changed failed (non-fatal): {e}")

    def notify_external_signal(
        self,
        signal_type: str,
        source: str,
        content: str,
        topic: str = None,
        activation_energy: float = 0.5,
        metadata: dict = None,
    ) -> None:
        """
        Write an external signal directly to world state (zero LLM).

        External signals represent passive world knowledge — things Chalie
        overheard, not things directed at Chalie. They are stored in a capped
        MemoryStore list and scored by temporal decay when the world state is
        assembled.

        Args:
            signal_type: Signal category (e.g. 'stock_price', 'weather', 'alert').
            source: Originating interface/wrapper identifier.
            content: Human-readable signal content.
            topic: Optional topic hint.
            activation_energy: Urgency weight 0–1 (affects salience scoring).
            metadata: Optional arbitrary context dict.
        """
        try:
            # Context updates bridge into the telemetry pipeline so that
            # backend services (ambient inference, tool telemetry, etc.)
            # see the user's device/location/behavioral data.
            if signal_type == 'context_update':
                self._bridge_context_update(content)

            store = self._get_store()
            entry = json.dumps({
                'signal_type': signal_type,
                'source': source,
                'content': content,
                'topic': topic,
                'activation_energy': activation_energy,
                'metadata': metadata,
                'timestamp': time.time(),
            })
            store.rpush(EXTERNAL_SIGNALS_KEY, entry)
            # Cap the list to prevent unbounded growth
            store.ltrim(EXTERNAL_SIGNALS_KEY, -MAX_EXTERNAL_SIGNALS, -1)
            store.expire(EXTERNAL_SIGNALS_KEY, 3600)
            logger.debug(
                f"{LOG_PREFIX} External signal: {signal_type} from {source}"
            )
        except Exception as e:
            logger.debug(
                f"{LOG_PREFIX} notify_external_signal failed (non-fatal): {e}"
            )

    def _bridge_context_update(self, content: str) -> None:
        """
        Bridge a context_update signal into the telemetry pipeline.

        When the chat interface (or any frontend) sends a context_update
        signal, the payload is the same heartbeat JSON that POST /health
        currently receives.  This bridge calls ClientContextService.save()
        so that all downstream consumers (ambient inference, tool telemetry,
        place learning, circadian data) see the fresh context — regardless
        of whether it arrived via POST /health or via the signal pathway.

        Fail-open: any error is logged and swallowed.
        """
        try:
            payload = json.loads(content) if isinstance(content, str) else content
            if not isinstance(payload, dict) or not payload:
                return
            from services.client_context_service import ClientContextService
            ClientContextService().save(payload)
            logger.debug(f"{LOG_PREFIX} context_update bridged to ClientContextService")
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} context_update bridge failed (non-fatal): {e}")

    def _get_external_signals(self, message_embedding: list = None) -> list:
        """
        Surface external signals from paired interfaces, grouped by source.

        Each source gets exactly one world state slot, regardless of how many
        signals it sent.  The slot shows the total count and the most salient
        signal as the representative.  This prevents a noisy source from
        monopolizing the world state and keeps context compact.

        Salience per signal: ``min(1.0, temporal_decay × activation_energy × 2)``
        with a 6-hour half-life.  The source-level salience is the salience of
        its most notable signal.

        Args:
            message_embedding: Current message embedding (unused — reserved for
                future semantic scoring).

        Returns:
            list: One item per source of type 'external_signal'.
        """
        try:
            store = self._get_store()
            raw_list = store.lrange(EXTERNAL_SIGNALS_KEY, 0, -1)
            if not raw_list:
                return []

            now_ts = time.time()

            # Group signals by source, track count and best signal per source
            # source → { 'count': int, 'best_salience': float, 'best_content': str }
            sources: dict = {}

            for raw in raw_list:
                try:
                    entry = json.loads(raw) if isinstance(raw, str) else raw
                    ts = entry.get('timestamp', 0)
                    hours_ago = (now_ts - ts) / 3600.0
                    if hours_ago < 0:
                        hours_ago = 0

                    # Temporal decay with external signal half-life
                    temporal = math.exp(
                        -0.693 * hours_ago / EXTERNAL_SIGNAL_DECAY_HOURS
                    )

                    # Activation energy amplifies salience
                    energy = entry.get('activation_energy', 0.5)
                    salience = min(1.0, temporal * energy * 2)

                    if salience < SALIENCE_THRESHOLD:
                        continue

                    source = entry.get('source', 'unknown')
                    content = entry.get('content', '')

                    if source not in sources:
                        sources[source] = {
                            'count': 0,
                            'best_salience': 0.0,
                            'best_content': '',
                        }

                    bucket = sources[source]
                    bucket['count'] += 1
                    if salience > bucket['best_salience']:
                        bucket['best_salience'] = salience
                        bucket['best_content'] = content

                except Exception as e:
                    logger.debug(f"[WORLD STATE] external signal parse failed: {e}")
                    continue

            # Build one world state item per source — most salient signal wins
            items = []
            for source, bucket in sources.items():
                content = bucket['best_content']
                label = f"[SIGNAL:{source}] {content}"

                items.append({
                    'type': 'external_signal',
                    'label': label,
                    'salience': bucket['best_salience'],
                })

            # Sort by salience descending, cap candidates
            items.sort(key=lambda x: x['salience'], reverse=True)
            return items[:MAX_EXTERNAL_SIGNAL_CANDIDATES]

        except Exception as e:
            logger.debug(
                f"{LOG_PREFIX} _get_external_signals failed (non-fatal): {e}"
            )
            return []

    def _get_items_from_cache(self, message_embedding: list = None):
        """
        Try to score items from the MemoryStore cache instead of querying SQLite.

        Returns a list of scored items if the cache is fresh, or None if the
        cache is stale / missing / errored (triggering DB fallback).

        Semantic scoring still uses sqlite-vec (KNN requires a DB connection),
        so a DB connection is opened only when message_embedding is provided.

        Args:
            message_embedding: Current message embedding for semantic scoring.

        Returns:
            list of item dicts with 'type', 'label', 'salience', or None.
        """
        try:
            store = self._get_store()
            raw = store.get(WORLD_MODEL_KEY)
            if not raw:
                return None

            payload = json.loads(raw)
            refreshed_at = payload.get('refreshed_at', 0)
            age = time.time() - refreshed_at
            if age > CACHE_MAX_AGE:
                logger.debug(f"{LOG_PREFIX} Cache stale ({age:.0f}s > {CACHE_MAX_AGE}s), falling back to DB")
                return None

            now = utc_now()
            items = []

            # Score cached items — semantic scoring needs a DB connection
            items = self._score_cached_items(
                payload, now, message_embedding
            )

            logger.debug(
                f"{LOG_PREFIX} Served {len(items)} items from cache (age={age:.0f}s)"
            )
            return items

        except Exception as e:
            logger.debug(f"{LOG_PREFIX} _get_items_from_cache failed (non-fatal): {e}")
            return None

    def _score_cached_items(self, payload, now, message_embedding=None):
        """Score cached scheduled/task/list items with temporal + semantic salience.

        Uses a single DB connection for all semantic KNN lookups when
        message_embedding is provided.

        Returns:
            list: Scored item dicts with 'type', 'label', 'salience'.
        """
        items = []

        # Semantic scoring closure — uses a single shared connection
        def _score_all(conn):
            # --- Scheduled items ---
            for entry in payload.get('scheduled_items', []):
                try:
                    item_id = entry.get('id')
                    message = entry.get('message', '')
                    due_at_str = entry.get('due_at')
                    status = entry.get('status', 'pending')
                    recurrence = entry.get('recurrence')

                    if not due_at_str:
                        continue

                    due_at = parse_utc(due_at_str)
                    temporal = self._temporal_score(now, due_at, status == 'fired')

                    semantic = 0.0
                    if message_embedding and conn:
                        semantic = self._semantic_score_scheduled(
                            conn, item_id, message_embedding
                        )

                    salience = W_TEMPORAL * temporal + W_SEMANTIC * semantic
                    if salience < SALIENCE_THRESHOLD:
                        continue

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
                except Exception as e:
                    logger.debug(f"{LOG_PREFIX} Cache: scheduled item error: {e}")

            # --- Persistent tasks ---
            for entry in payload.get('persistent_tasks', []):
                try:
                    task_id = entry.get('id')
                    goal = entry.get('goal', '')
                    status = entry.get('status', '')
                    progress_json = entry.get('progress')
                    updated_at_str = entry.get('updated_at')
                    deadline_str = entry.get('deadline')

                    if deadline_str:
                        deadline = parse_utc(deadline_str)
                        temporal = self._temporal_score(
                            now, deadline, status == 'completed'
                        )
                    elif status == 'completed':
                        updated = parse_utc(updated_at_str) if updated_at_str else now
                        temporal = self._past_decay_score(now, updated)
                    else:
                        temporal = 0.5

                    semantic = 0.0
                    if message_embedding and conn and task_id is not None:
                        semantic = self._semantic_score_task(
                            conn, task_id, message_embedding
                        )

                    salience = W_TEMPORAL * temporal + W_SEMANTIC * semantic
                    if salience < SALIENCE_THRESHOLD:
                        continue

                    progress = (
                        json.loads(progress_json) if progress_json else {}
                    ) if isinstance(progress_json, str) else (progress_json or {})
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
                except Exception as e:
                    logger.debug(f"{LOG_PREFIX} Cache: task error: {e}")

            # --- Lists ---
            for entry in payload.get('lists', []):
                try:
                    list_id = entry.get('id')
                    name = entry.get('name', '')
                    updated_at_str = entry.get('updated_at')
                    item_count = entry.get('item_count', 0) or 0
                    checked_count = entry.get('checked_count', 0) or 0

                    if not updated_at_str:
                        continue

                    updated_at = parse_utc(updated_at_str)
                    temporal = self._past_decay_score(now, updated_at)

                    semantic = 0.0
                    if message_embedding and conn:
                        semantic = self._semantic_score_list(
                            conn, list_id, message_embedding
                        )

                    salience = W_TEMPORAL * temporal + W_SEMANTIC * semantic
                    if salience < SALIENCE_THRESHOLD:
                        continue

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
                except Exception as e:
                    logger.debug(f"{LOG_PREFIX} Cache: list error: {e}")

        if message_embedding:
            try:
                db = self._get_db()
                with db.connection() as conn:
                    _score_all(conn)
            except Exception as e:
                logger.debug(f"{LOG_PREFIX} DB connection for semantic scoring failed: {e}")
                _score_all(None)  # Fall back to temporal-only
        else:
            _score_all(None)  # Temporal-only scoring

        return self._collapse_items(items)

    @staticmethod
    def _collapse_items(items: list) -> list:
        """Collapse items so each type gets at most one entry per unique entity.

        Same bucket pattern as _get_external_signals: group by (type, key),
        keep the highest-salience label per group.  The key is the entity
        identity — message text for scheduled items, goal for tasks, name for
        lists.

        Args:
            items: Scored item dicts with 'type', 'label', 'salience'.

        Returns:
            Deduplicated list, one item per unique (type, key).
        """
        buckets: dict[tuple[str, str], dict] = {}

        for item in items:
            item_type = item.get('type', '')
            label = item.get('label', '')
            salience = item.get('salience', 0.0)

            # Extract the entity key from the label depending on type.
            # Strips the prefix tag so "Check TSLA..." matches across
            # [DONE] and [in 3h] variants.
            if item_type == 'scheduled':
                # Label: "[DONE] message" or "[in 3h] message (recurring: ...)"
                key = label.split('] ', 1)[-1].split(' (recurring:')[0].strip()
            elif item_type == 'task':
                # Label: "[COMPLETED] goal" or "[ACTIVE] goal (85%)"
                key = label.split('] ', 1)[-1].split(' (')[0].strip()
            elif item_type == 'list':
                # Label: "[LIST] Name (N items) — updated Xh ago"
                key = label.split('] ', 1)[-1].split(' (')[0].strip()
            else:
                key = label

            bucket_key = (item_type, key)
            existing = buckets.get(bucket_key)
            if existing is None or salience > existing['salience']:
                buckets[bucket_key] = item

        return list(buckets.values())

    # ── Ambient Context Collectors ────────────────────────────────────────────

    def _get_engagement_signal(self) -> list:
        """
        Surface the rolling engagement score as a world-state item when it is
        outside the unremarkable band.

        Delegates to :class:`~services.engagement_signal_service.EngagementSignalService`
        which reads ``proactive:engagement_score`` from MemoryStore.

        This method is fail-open: any import error or runtime exception is caught,
        logged at debug level, and an empty list is returned so that the rest of
        ``get_world_state()`` is never blocked.

        Returns:
            list: Zero or one world-state dict with keys ``type`` (str),
            ``label`` (str), and ``salience`` (float).
        """
        try:
            from services.engagement_signal_service import EngagementSignalService
            return EngagementSignalService().get_engagement_items()
        except Exception as e:
            logger.debug(
                f"{LOG_PREFIX} _get_engagement_signal failed (non-fatal): {e}"
            )
            return []

    def get_world_model_summary(self) -> dict:
        """
        Return a structured dict representation of the current world model for
        consumption by the reasoning loop's _get_loop_context().

        Combines cached scheduled/task/list items with ambient context, active
        topics, and reasoning focus without formatting them for the prompt.

        Returns:
            dict: Keys — 'scheduled', 'tasks', 'lists', 'ambient', 'topics',
                  'reasoning_focus'. Each is a list of label strings.
        """
        summary = {
            'scheduled': [],
            'tasks': [],
            'lists': [],
            'ambient': [],
            'topics': [],
            'reasoning_focus': [],
            'external_signals': [],
        }
        try:
            store = self._get_store()
            raw = store.get(WORLD_MODEL_KEY)
            if raw:
                payload = json.loads(raw)
                now = utc_now()

                for entry in payload.get('scheduled_items', []):
                    try:
                        due_at_str = entry.get('due_at')
                        if not due_at_str:
                            continue
                        due_at = parse_utc(due_at_str)
                        status = entry.get('status', 'pending')
                        if status == 'fired':
                            summary['scheduled'].append(
                                f"[DONE] {entry.get('message', '')}"
                            )
                        else:
                            time_str = self._relative_time(now, due_at)
                            summary['scheduled'].append(
                                f"[{time_str}] {entry.get('message', '')}"
                            )
                    except Exception as e:
                        logger.debug(f"{LOG_PREFIX} Skipping malformed scheduled entry: {e}", exc_info=False)

                for entry in payload.get('persistent_tasks', []):
                    try:
                        status = entry.get('status', '')
                        goal = entry.get('goal', '')
                        summary['tasks'].append(f"[{status.upper()}] {goal[:80]}")
                    except Exception as e:
                        logger.debug(f"{LOG_PREFIX} Skipping malformed task entry: {e}", exc_info=False)

                for entry in payload.get('lists', []):
                    try:
                        summary['lists'].append(entry.get('name', ''))
                    except Exception as e:
                        logger.debug(f"{LOG_PREFIX} Skipping malformed list entry: {e}", exc_info=False)

        except Exception as e:
            logger.debug(f"{LOG_PREFIX} get_world_model_summary: cache read failed: {e}")

        # Topics, reasoning focus, external signals from MemoryStore
        for item in self._get_active_topics():
            summary['topics'].append(item['label'])

        for item in self._get_reasoning_focus():
            summary['reasoning_focus'].append(item['label'])

        for item in self._get_external_signals():
            summary['external_signals'].append(item['label'])

        return summary

    # ── Signal Collectors ────────────────────────────────────────────────────

    def _get_active_steps(self, thread_id: str) -> list:
        """Get in-flight ACT loop steps — always treated as maximally salient."""
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
                    WHERE ((status = 'pending' AND due_at <= datetime(?, '+7 days'))
                       OR (status = 'fired' AND last_fired_at >= datetime(?, '-24 hours')))
                      AND COALESCE(hidden, 0) = 0
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
                        elif item_type == 'event':
                            time_str = self._relative_time(now, due_at)
                            label = f"[CALENDAR {time_str}] {message}"
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
            packed = pack_embedding(message_embedding)
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
            packed = pack_embedding(message_embedding)
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
            packed = pack_embedding(message_embedding)
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
    def _get_user_telemetry() -> str:
        """Build user telemetry line — time, location, device, energy, attention, mobility.

        Returns middle-dot-joined string or empty string when no client context available.
        """
        try:
            from services.client_context_service import ClientContextService
            from services.ambient_inference_service import AmbientInferenceService
            from services.place_learning_service import PlaceLearningService
            from services.database_service import DatabaseService
            from zoneinfo import ZoneInfo
            from datetime import datetime

            ctx = ClientContextService().get()
            if not ctx:
                return ''

            parts = []

            if timezone := ctx.get('timezone'):
                user_dt = datetime.now(ZoneInfo(timezone))
                parts.append(f"{user_dt.strftime('%a %d %b')}, {user_dt.strftime('%H:%M')} {user_dt.strftime('%Z') or timezone.split('/')[-1]}")

            device = ctx.get('device', {})
            if device_class := device.get('class'):
                battery = ctx.get('battery', {})
                battery_level = battery.get('level') if isinstance(battery, dict) else None
                if battery_level is not None:
                    parts.append(f"{device_class} {int(battery_level)}%")
                else:
                    parts.append(device_class)

            location_name = ctx.get('location_name', '')
            db = DatabaseService()
            inferences = AmbientInferenceService(
                place_learning_service=PlaceLearningService(db)
            ).infer(ctx)

            place = inferences.get('place')
            if location_name and place:
                parts.append(f"{location_name} ({place})")
            elif location_name:
                parts.append(location_name)
            elif place:
                parts.append(place)

            if energy := inferences.get('energy'):
                parts.append(f"Energy: {energy}")

            attention_labels = {
                'deep_focus': 'Focus: deep', 'focused': 'Focus: active',
                'casual': 'Casual', 'distracted': 'Distracted', 'away': 'Away',
            }
            if attention := inferences.get('attention'):
                if label := attention_labels.get(attention):
                    parts.append(label)

            if mobility := inferences.get('mobility'):
                if mobility != 'stationary':
                    parts.append(mobility.capitalize())

            return ' \u00b7 '.join(parts) if parts else ''

        except Exception as e:
            logger.debug(f"[WORLD STATE] User telemetry unavailable: {e}")
            return ''

    @staticmethod
    def _format_world_state(items: list) -> str:
        """Format salient items into a prompt-ready text block."""
        lines = ["\n## World State"]
        for item in items:
            lines.append(f"- {item['label']}")
        return "\n".join(lines)
