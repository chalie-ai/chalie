"""
Worker Health Monitor Service

Provides enhanced health monitoring for RQ workers beyond simple process.is_alive() checks.
Monitors Redis heartbeats and job processing activity to detect hung/zombie workers.
"""
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Set, Optional
from services.redis_client import RedisClientService


class WorkerHealthMonitor:
    """
    Enhanced worker health monitoring that checks both process status and Redis activity.

    Features:
    - Process liveness check (multiprocessing.Process.is_alive())
    - Redis heartbeat check (last_heartbeat timestamp)
    - Job processing activity check (detects hung workers)
    - Queue consumption check (detects growing-but-unconsumed queues)
    - Per-worker health metrics and restart counters
    """

    # Grace period after spawn before Redis checks kick in (seconds)
    SPAWN_GRACE_PERIOD = 30

    def __init__(self, heartbeat_timeout_seconds: int = 60, activity_timeout_seconds: int = 300):
        """
        Initialize worker health monitor.

        Args:
            heartbeat_timeout_seconds: Max seconds since last Redis heartbeat before declaring dead (default 60)
            activity_timeout_seconds: Max seconds a worker can be busy without progress (default 300)
        """
        self.redis = RedisClientService.create_connection()
        self.heartbeat_timeout = heartbeat_timeout_seconds
        self.activity_timeout = activity_timeout_seconds

        # Track worker health metrics
        self.worker_metrics: Dict[str, dict] = {}
        self.restart_counts: Dict[str, int] = {}
        # Track when workers were last spawned (for grace period)
        self.spawn_times: Dict[str, float] = {}

    def record_spawn(self, worker_id: str):
        """Record when a worker was spawned (for grace period tracking)."""
        self.spawn_times[worker_id] = time.time()
        # Reset activity metrics so stale stuck_since doesn't carry over
        self.worker_metrics.pop(worker_id, None)

    def check_process_health(self, worker_id: str, process) -> tuple[bool, str]:
        """
        Check if multiprocessing.Process is alive and responsive.

        Returns:
            (is_healthy, reason) tuple
        """
        if not process:
            return False, "process_not_found"

        if not process.is_alive():
            return False, "process_dead"

        # Check if process is a zombie (exitcode set but still "alive")
        if process.exitcode is not None:
            return False, f"process_zombie_exitcode_{process.exitcode}"

        return True, "ok"

    def check_redis_heartbeat(self, worker_id: str, queue_name: str) -> tuple[bool, str]:
        """
        Check RQ worker heartbeat in Redis.

        Returns:
            (is_healthy, reason) tuple
        """
        # Find RQ worker ID for this queue
        rq_workers = self.redis.smembers("rq:workers")

        matching_worker = None
        for worker_key in rq_workers:
            worker_key_str = worker_key.decode('utf-8') if isinstance(worker_key, bytes) else worker_key
            queues = self.redis.hget(worker_key_str, "queues")

            if queues:
                queues_str = queues.decode('utf-8') if isinstance(queues, bytes) else queues
                if queue_name in queues_str:
                    matching_worker = worker_key_str
                    break

        if not matching_worker:
            return False, "redis_worker_not_registered"

        # Check last heartbeat
        last_heartbeat = self.redis.hget(matching_worker, "last_heartbeat")
        if not last_heartbeat:
            return False, "redis_no_heartbeat"

        # Parse and compare heartbeat timestamp
        try:
            heartbeat_str = last_heartbeat.decode('utf-8') if isinstance(last_heartbeat, bytes) else last_heartbeat
            # RQ uses ISO format: 2026-02-16T10:35:42.431353Z
            heartbeat_time = datetime.fromisoformat(heartbeat_str.replace('Z', '+00:00'))
            age_seconds = (datetime.now(timezone.utc) - heartbeat_time).total_seconds()

            if age_seconds > self.heartbeat_timeout:
                return False, f"redis_heartbeat_stale_{int(age_seconds)}s"

            return True, "ok"

        except Exception as e:
            logging.warning(f"[HealthMonitor] Error parsing heartbeat for {worker_id}: {e}")
            # On parse error, don't fail — could be a clock issue
            return True, "ok_parse_error"

    def _sample_job_ids(self, queue_key: str, sample_size: int = 5) -> set:
        """Sample job IDs from head and tail of queue for turnover detection."""
        length = self.redis.llen(queue_key)
        if length == 0:
            return set()

        job_ids = set()
        # Sample from head (oldest jobs — first to be consumed)
        head = self.redis.lrange(queue_key, 0, min(sample_size - 1, length - 1))
        # Sample from tail (newest jobs)
        if length > sample_size:
            tail = self.redis.lrange(queue_key, -sample_size, -1)
        else:
            tail = []

        for item in head + tail:
            item_str = item.decode('utf-8') if isinstance(item, bytes) else item
            job_ids.add(item_str)

        return job_ids

    def check_job_activity(self, worker_id: str, queue_name: str) -> tuple[bool, str]:
        """
        Check if worker is making progress on jobs (not hung).
        Tracks actual job IDs in the queue — if different jobs appear between
        checks, the worker is making progress even if queue length stays constant.

        Returns:
            (is_healthy, reason) tuple
        """
        queue_key = f"rq:queue:{queue_name}"
        queue_length = self.redis.llen(queue_key)
        current_job_ids = self._sample_job_ids(queue_key) if queue_length > 0 else set()

        # Initialize metrics on first check
        if worker_id not in self.worker_metrics:
            self.worker_metrics[worker_id] = {
                'last_queue_length': queue_length,
                'last_check_time': time.time(),
                'last_job_ids': current_job_ids,
                'stuck_since': None,
                'growing_since': None,
            }
            return True, "ok_new_worker"

        metrics = self.worker_metrics[worker_id]
        current_time = time.time()
        last_job_ids = metrics.get('last_job_ids', set())

        # Job turnover: if the sampled IDs changed, work is being done
        has_turnover = current_job_ids != last_job_ids

        # Detect truly stuck queue (same jobs sitting for too long)
        if queue_length > 0 and not has_turnover:
            if metrics['stuck_since'] is None:
                metrics['stuck_since'] = current_time

            stuck_duration = current_time - metrics['stuck_since']
            if stuck_duration > self.activity_timeout:
                metrics['last_queue_length'] = queue_length
                metrics['last_check_time'] = current_time
                metrics['last_job_ids'] = current_job_ids
                return False, f"hung_stuck_for_{int(stuck_duration)}s"
        else:
            metrics['stuck_since'] = None

        # Detect growing queue (length only increasing, never decreasing)
        if queue_length > 0 and queue_length > metrics['last_queue_length'] and not has_turnover:
            if metrics['growing_since'] is None:
                metrics['growing_since'] = current_time

            growing_duration = current_time - metrics['growing_since']
            if growing_duration > self.activity_timeout:
                metrics['last_queue_length'] = queue_length
                metrics['last_check_time'] = current_time
                metrics['last_job_ids'] = current_job_ids
                return False, f"queue_only_growing_for_{int(growing_duration)}s"
        else:
            metrics['growing_since'] = None

        # Update metrics
        metrics['last_queue_length'] = queue_length
        metrics['last_check_time'] = current_time
        metrics['last_job_ids'] = current_job_ids

        return True, "ok"

    def comprehensive_health_check(self, worker_id: str, process, queue_name: str) -> tuple[bool, str]:
        """
        Run all health checks and return overall health status.

        Returns:
            (is_healthy, reason) tuple
        """
        # Check 1: Process liveness (always authoritative)
        process_healthy, process_reason = self.check_process_health(worker_id, process)
        if not process_healthy:
            return False, process_reason

        # Check 2: Redis heartbeat
        redis_healthy, redis_reason = self.check_redis_heartbeat(worker_id, queue_name)

        # After grace period, Redis heartbeat checks kick in
        spawn_time = self.spawn_times.get(worker_id, 0)
        since_spawn = time.time() - spawn_time

        if not redis_healthy and since_spawn > self.SPAWN_GRACE_PERIOD:
            # Process is confirmed alive (passed Check 1).
            # Only treat stale heartbeats as unhealthy — not registration issues.
            # "redis_worker_not_registered" can happen during RQ re-registration
            # cycles and doesn't mean the worker is broken.
            if "redis_heartbeat_stale" in redis_reason:
                return False, redis_reason
            else:
                logging.debug(f"[HealthMonitor] {worker_id}: {redis_reason} (process alive, ignoring)")

        # Check 3: Job activity (detect hung or unconsumed workers)
        activity_healthy, activity_reason = self.check_job_activity(worker_id, queue_name)
        if not activity_healthy:
            return False, activity_reason

        return True, "ok"

    def record_restart(self, worker_id: str):
        """Record a worker restart event."""
        self.restart_counts[worker_id] = self.restart_counts.get(worker_id, 0) + 1
        restart_count = self.restart_counts[worker_id]
        self.record_spawn(worker_id)

        if restart_count > 5:
            logging.error(f"[HealthMonitor] Worker {worker_id} has restarted {restart_count} times! Possible crash loop.")
        else:
            logging.warning(f"[HealthMonitor] Worker {worker_id} restart #{restart_count}")

    def get_worker_stats(self, worker_id: str) -> dict:
        """Get health statistics for a worker."""
        return {
            'restart_count': self.restart_counts.get(worker_id, 0),
            'metrics': self.worker_metrics.get(worker_id, {})
        }

    def get_all_stats(self) -> dict:
        """Get health statistics for all monitored workers."""
        return {
            'restart_counts': self.restart_counts,
            'worker_metrics': self.worker_metrics
        }
