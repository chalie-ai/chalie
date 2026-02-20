"""
Metrics Service - Structured metrics and per-request tracing.

Redis-backed counters and timing records with daily rollup keys.
No external dependencies required.
"""

import time
import json
import uuid
import logging
from typing import Dict, Any, Optional
from services.redis_client import RedisClientService


class MetricsService:
    """Manages metrics collection with Redis counters and timing records."""

    def __init__(self):
        """Initialize metrics service with Redis connection."""
        self.redis = RedisClientService.create_connection()

    def start_trace(self) -> str:
        """
        Start a new trace for a request.

        Returns:
            trace_id: Unique trace identifier
        """
        trace_id = str(uuid.uuid4())[:8]
        trace_key = f"trace:{trace_id}"

        trace_data = {
            'trace_id': trace_id,
            'started_at': time.time(),
            'timings': {},
            'counters': {}
        }

        # Store trace with 1-hour TTL
        self.redis.setex(trace_key, 3600, json.dumps(trace_data))
        return trace_id

    def record_timing(self, trace_id: str, operation: str, duration_ms: float):
        """
        Record a timing measurement for a traced operation.

        Args:
            trace_id: Trace identifier from start_trace()
            operation: Name of the operation (e.g., 'classification', 'response_generation')
            duration_ms: Duration in milliseconds
        """
        trace_key = f"trace:{trace_id}"
        trace_json = self.redis.get(trace_key)

        if trace_json:
            trace_data = json.loads(trace_json)
            trace_data['timings'][operation] = duration_ms
            self.redis.setex(trace_key, 3600, json.dumps(trace_data))

        # Also record in daily rollup
        day_key = time.strftime('%Y-%m-%d')
        rollup_key = f"metrics:timing:{operation}:{day_key}"

        pipe = self.redis.pipeline()
        pipe.rpush(rollup_key, str(duration_ms))
        pipe.expire(rollup_key, 86400 * 7)  # 7-day retention
        pipe.execute()

    def record_counter(self, metric_name: str, value: int = 1):
        """
        Increment a counter metric.

        Args:
            metric_name: Name of the metric (e.g., 'requests_total', 'errors_total')
            value: Value to increment by (default 1)
        """
        day_key = time.strftime('%Y-%m-%d')
        counter_key = f"metrics:counter:{metric_name}:{day_key}"

        pipe = self.redis.pipeline()
        pipe.incrby(counter_key, value)
        pipe.expire(counter_key, 86400 * 7)  # 7-day retention
        pipe.execute()

    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a trace by ID.

        Args:
            trace_id: Trace identifier

        Returns:
            Trace data dict or None
        """
        trace_key = f"trace:{trace_id}"
        trace_json = self.redis.get(trace_key)

        if trace_json:
            return json.loads(trace_json)
        return None

    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get aggregated metrics for dashboard display.

        Returns:
            Dict with counters, timing averages, and recent activity
        """
        day_key = time.strftime('%Y-%m-%d')
        dashboard = {
            'date': day_key,
            'counters': {},
            'timing_averages': {}
        }

        # Collect counters
        counter_names = [
            'requests_total', 'responses_total', 'errors_total',
            'classifications_total', 'facts_extracted',
            'memory_chunks_enqueued', 'episodes_generated'
        ]

        for name in counter_names:
            counter_key = f"metrics:counter:{name}:{day_key}"
            value = self.redis.get(counter_key)
            dashboard['counters'][name] = int(value) if value else 0

        # Collect timing averages
        timing_operations = [
            'classification', 'response_generation',
            'fact_extraction', 'context_assembly'
        ]

        for operation in timing_operations:
            rollup_key = f"metrics:timing:{operation}:{day_key}"
            values = self.redis.lrange(rollup_key, 0, -1)

            if values:
                float_values = [float(v) for v in values]
                dashboard['timing_averages'][operation] = {
                    'count': len(float_values),
                    'avg_ms': sum(float_values) / len(float_values),
                    'min_ms': min(float_values),
                    'max_ms': max(float_values)
                }

        return dashboard
