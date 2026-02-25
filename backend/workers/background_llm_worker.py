"""
Background LLM Worker — single-process consumer for bg_llm:queue.

Processes background LLM calls sequentially with adaptive back-off:
  - Idle  (>5 min since last user message, no active prompt-queue): ~1s sleep
  - Normal (2-5 min since last message):                            ~5s sleep
  - Busy  (<2 min since last message OR prompt-queue has items):    ~10s sleep

±10% jitter on all tiers prevents rhythmic provider bursts.

Failed jobs (exception or None response) are retried at the back of the queue
(max 2 retries). After retries are exhausted, an error result is pushed so the
caller unblocks cleanly.

A heartbeat key (bg_llm:last_heartbeat, TTL 60s) is updated every loop so the
proxy can detect a stalled worker.

Observability: aggregated metrics are logged every 60s.
"""

import json
import time
import random
import logging
import concurrent.futures
from typing import Optional

from services.redis_client import RedisClientService
from services.llm_service import create_refreshable_llm_service

logger = logging.getLogger(__name__)

LOG_PREFIX = "[BG LLM]"

# Redis keys
QUEUE_KEY = "bg_llm:queue"
RESULT_KEY_PREFIX = "bg_llm:result:"
HEARTBEAT_KEY = "bg_llm:last_heartbeat"
LAST_INTERACTION_KEY = "proactive:default:last_interaction_ts"
PROMPT_QUEUE_KEY = "prompt-queue"

# Thresholds & limits
STALE_THRESHOLD = 300    # seconds — discard jobs older than this
LLM_CALL_TIMEOUT = 120   # seconds — hard thread-based timeout on provider call
MAX_RETRIES = 2          # additional attempts after first failure (3 total)
BLPOP_TIMEOUT = 30       # seconds — how long to block on empty queue before looping

# Adaptive back-off
BUSY_SLEEP = 10
NORMAL_SLEEP = 5
IDLE_SLEEP = 1

# Observability
METRICS_INTERVAL = 60    # seconds between metric log emissions


def _get_sleep_interval(redis) -> float:
    """
    Return adaptive sleep duration based on system activity signals.

    Checks two O(1) Redis reads:
      1. prompt-queue length (message being processed right now)
      2. last user interaction timestamp (recently chatting?)
    """
    try:
        if redis.llen(PROMPT_QUEUE_KEY) > 0:
            base = BUSY_SLEEP
        else:
            last_ts = redis.get(LAST_INTERACTION_KEY)
            if not last_ts:
                base = IDLE_SLEEP
            else:
                gap = time.time() - float(last_ts)
                if gap < 120:
                    base = BUSY_SLEEP
                elif gap < 300:
                    base = NORMAL_SLEEP
                else:
                    base = IDLE_SLEEP
    except Exception:
        base = NORMAL_SLEEP  # safe default on any Redis error

    # ±10% jitter to prevent rhythmic bursts across services
    return base * random.uniform(0.9, 1.1)


def background_llm_worker():
    """
    Entry point registered with consumer.py via manager.register_service().

    Runs indefinitely, consuming from bg_llm:queue one job at a time.
    """
    redis = RedisClientService.create_connection()

    # Cache of agent_name → RefreshableLLMService
    # Each service auto-refreshes on provider config change.
    llm_cache: dict = {}

    metrics = {
        "jobs_processed": 0,
        "retries": 0,
        "stale_discarded": 0,
        "failures": 0,
        "timeouts": 0,
        "total_processing_ms": 0,
        "max_queue_depth_seen": 0,
    }
    last_metrics_emit = time.time()

    logger.info("%s Worker started", LOG_PREFIX)

    while True:
        # Heartbeat — TTL 60s; any crash is detectable within 60s by the proxy
        try:
            redis.set(HEARTBEAT_KEY, str(time.time()), ex=60)
        except Exception as e:
            logger.warning("%s Heartbeat write failed: %s", LOG_PREFIX, e)

        # Track max queue depth
        try:
            depth = redis.llen(QUEUE_KEY)
            if depth > metrics["max_queue_depth_seen"]:
                metrics["max_queue_depth_seen"] = depth
        except Exception:
            pass

        # Emit metrics every METRICS_INTERVAL seconds then reset counters
        now = time.time()
        if now - last_metrics_emit >= METRICS_INTERVAL:
            processed = metrics["jobs_processed"]
            avg_ms = (
                int(metrics["total_processing_ms"] / processed)
                if processed > 0 else 0
            )
            logger.info(
                "%s metrics — processed=%d retries=%d stale=%d "
                "failures=%d timeouts=%d avg_ms=%d max_depth=%d",
                LOG_PREFIX,
                processed,
                metrics["retries"],
                metrics["stale_discarded"],
                metrics["failures"],
                metrics["timeouts"],
                avg_ms,
                metrics["max_queue_depth_seen"],
            )
            for k in metrics:
                metrics[k] = 0
            last_metrics_emit = now

        # Block until a job arrives; short timeout keeps the heartbeat loop alive
        try:
            raw = redis.blpop(QUEUE_KEY, timeout=BLPOP_TIMEOUT)
        except Exception as e:
            logger.error("%s BLPOP failed: %s", LOG_PREFIX, e)
            time.sleep(2)
            continue

        if raw is None:
            continue  # timeout — loop to refresh heartbeat and check metrics

        _, payload = raw
        try:
            job = json.loads(payload)
        except Exception as e:
            logger.error("%s Failed to parse job payload: %s", LOG_PREFIX, e)
            continue

        job_id = job.get("job_id", "unknown")
        agent_name = job.get("agent_name", "unknown")
        result_key = f"{RESULT_KEY_PREFIX}{job_id}"
        enqueued_at = job.get("enqueued_at", time.time())
        retry_count = job.get("retry_count", 0)

        # Staleness check — discard jobs queued too long ago so callers unblock
        age = time.time() - enqueued_at
        if age > STALE_THRESHOLD:
            logger.warning(
                "%s Discarding stale job (age=%.0fs, agent=%s, job=%s)",
                LOG_PREFIX, age, agent_name, job_id,
            )
            metrics["stale_discarded"] += 1
            try:
                redis.rpush(result_key, json.dumps({"error": "stale"}))
                redis.expire(result_key, 60)
            except Exception:
                pass
            continue

        # Get or create LLM service for this agent (cached for provider refresh)
        if agent_name not in llm_cache:
            try:
                llm_cache[agent_name] = create_refreshable_llm_service(agent_name)
                logger.info(
                    "%s Created LLM service for agent=%s", LOG_PREFIX, agent_name
                )
            except Exception as e:
                logger.error(
                    "%s Failed to create LLM service for agent=%s: %s",
                    LOG_PREFIX, agent_name, e,
                )
                try:
                    redis.rpush(result_key, json.dumps({"error": "llm_init_failed"}))
                    redis.expire(result_key, 60)
                except Exception:
                    pass
                continue

        llm = llm_cache[agent_name]
        system_prompt = job.get("system_prompt", "")
        user_message = job.get("user_message", "")

        # Execute LLM call with hard thread-based timeout
        t_start = time.time()
        response = None

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(llm.send_message, system_prompt, user_message)
                try:
                    response = future.result(timeout=LLM_CALL_TIMEOUT)
                except concurrent.futures.TimeoutError:
                    logger.error(
                        "%s Call timed out after %ds (agent=%s, job=%s)",
                        LOG_PREFIX, LLM_CALL_TIMEOUT, agent_name, job_id,
                    )
                    metrics["timeouts"] += 1
        except Exception as e:
            logger.error(
                "%s LLM call exception (agent=%s, job=%s): %s",
                LOG_PREFIX, agent_name, job_id, e,
            )

        elapsed_ms = int((time.time() - t_start) * 1000)

        if response is not None:
            # Success — push result so caller's BRPOP unblocks
            result_data = {
                "text": response.text,
                "model": response.model,
                "provider": response.provider,
                "tokens_input": response.tokens_input,
                "tokens_output": response.tokens_output,
                "latency_ms": response.latency_ms,
            }
            try:
                redis.rpush(result_key, json.dumps(result_data))
                redis.expire(result_key, 300)
            except Exception as e:
                logger.error(
                    "%s Failed to push result (agent=%s, job=%s): %s",
                    LOG_PREFIX, agent_name, job_id, e,
                )
            metrics["jobs_processed"] += 1
            metrics["total_processing_ms"] += elapsed_ms
            logger.debug(
                "%s Completed (agent=%s, job=%s, ms=%d)",
                LOG_PREFIX, agent_name, job_id, elapsed_ms,
            )
        else:
            # Failure — retry to back of queue, or exhaust and push error
            if retry_count < MAX_RETRIES:
                job["retry_count"] = retry_count + 1
                try:
                    redis.rpush(QUEUE_KEY, json.dumps(job))
                    metrics["retries"] += 1
                    logger.warning(
                        "%s retry #%d for agent=%s, job=%s — re-queued at back",
                        LOG_PREFIX, job["retry_count"], agent_name, job_id,
                    )
                except Exception as e:
                    logger.error("%s Re-enqueue failed: %s", LOG_PREFIX, e)
                    # Can't retry — unblock the caller with an error
                    try:
                        redis.rpush(
                            result_key, json.dumps({"error": "reenqueue_failed"})
                        )
                        redis.expire(result_key, 60)
                    except Exception:
                        pass
            else:
                # Max retries exhausted — unblock caller so it can move on
                metrics["failures"] += 1
                logger.error(
                    "%s failed after %d retries (agent=%s, job=%s)",
                    LOG_PREFIX, MAX_RETRIES, agent_name, job_id,
                )
                try:
                    redis.rpush(
                        result_key, json.dumps({"error": "max_retries_exceeded"})
                    )
                    redis.expire(result_key, 60)
                except Exception:
                    pass

        # Adaptive sleep with ±10% jitter before processing next job
        sleep_duration = _get_sleep_interval(redis)
        logger.debug(
            "%s Sleeping %.1fs before next job (agent=%s)",
            LOG_PREFIX, sleep_duration, agent_name,
        )
        time.sleep(sleep_duration)
