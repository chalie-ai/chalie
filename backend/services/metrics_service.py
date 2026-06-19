"""Metrics Service — Redis-backed counters, timing records, and dashboard data."""

import time
import json
import uuid
from typing import Dict, cast
from services.memory_client import MemoryClientService
from services.memory_store import MemoryStore


class MetricsService:
    """Manages metrics collection with MemoryStore counters and timing records."""

    def __init__(self) -> None:
        self.store: MemoryStore = MemoryClientService.create_connection()

    def start_trace(self) -> str:
        trace_id = str(uuid.uuid4())[:8]
        trace_key = f"trace:{trace_id}"
        trace_data = {
            'trace_id': trace_id,
            'started_at': time.time(),
            'timings': {},
            'counters': {},
        }
        self.store.setex(trace_key, 3600, json.dumps(trace_data))
        return trace_id

    def record_timing(self, trace_id: str, operation: str, duration_ms: float) -> None:
        trace_key = f"trace:{trace_id}"
        trace_json = self.store.get(trace_key)
        if trace_json:
            trace_data = json.loads(trace_json)
            trace_data['timings'][operation] = duration_ms
            self.store.setex(trace_key, 3600, json.dumps(trace_data))

        day_key = time.strftime('%Y-%m-%d')
        rollup_key = f"metrics:timing:{operation}:{day_key}"
        pipe = self.store.pipeline()
        pipe.rpush(rollup_key, str(duration_ms))
        pipe.expire(rollup_key, 86400 * 7)
        pipe.execute()

    def record_counter(self, metric_name: str, value: int = 1) -> None:
        day_key = time.strftime('%Y-%m-%d')
        counter_key = f"metrics:counter:{metric_name}:{day_key}"
        pipe = self.store.pipeline()
        pipe.incrby(counter_key, value)
        pipe.expire(counter_key, 86400 * 7)
        pipe.execute()

    def get_dashboard_data(self) -> Dict[str, object]:
        """Get aggregated metrics for dashboard display."""
        day_key = time.strftime('%Y-%m-%d')
        dashboard: Dict[str, object] = {
            'date': day_key,
            'counters': {},
            'timing_averages': {},
        }

        counter_names = [
            'requests_total', 'responses_total', 'errors_total',
            'embeddings_total', 'facts_extracted',
            'memory_chunks_enqueued', 'episodes_generated',
            'dmn_turns_total', 'subagent_turns_total', 'scheduled_turns_total',
        ]
        for name in counter_names:
            counter_key = f"metrics:counter:{name}:{day_key}"
            value = self.store.get(counter_key)
            cast(Dict[str, object], dashboard['counters'])[name] = int(value) if value else 0

        # user_messages_total is no longer a stored counter (§4e). One user turn
        # writes exactly one transcript row with channel='user' AND role='user'
        # (UMP hardcodes both), so the value is an exact on-read COUNT. Exact as
        # long as history compaction never hard-deletes user transcript rows.
        cast(Dict[str, object], dashboard['counters'])['user_messages_total'] = self._count_user_messages()

        timing_operations = [
            'embedding', 'response_generation',
            'fact_extraction', 'context_assembly',
        ]
        for operation in timing_operations:
            rollup_key = f"metrics:timing:{operation}:{day_key}"
            values = self.store.lrange(rollup_key, 0, -1)
            if values:
                float_values = [float(v) for v in values]
                cast(Dict[str, object], dashboard['timing_averages'])[operation] = {
                    'count': len(float_values),
                    'avg_ms': sum(float_values) / len(float_values),
                    'min_ms': min(float_values),
                    'max_ms': max(float_values),
                }

        return dashboard

    @staticmethod
    def _count_user_messages() -> int:
        """COUNT of user-channel user-role transcript rows (§4e).

        Best-effort: returns 0 if the transcript table is unavailable (e.g.
        store-only test contexts) so the dashboard never fails on a missing DB.
        """
        try:
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()
            with db.connection() as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM transcript "
                    "WHERE role = 'user' AND channel = 'user'"
                ).fetchone()
            return int(row[0]) if row else 0
        except Exception:
            return 0
