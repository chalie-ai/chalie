"""
Background LLM Queue — serializes all background LLM calls through a single Redis queue.

Prevents thundering herd on LLM providers when multiple background services fire
simultaneously. Drop-in replacement for RefreshableLLMService in background workers.

Usage:
    from services.background_llm_queue import create_background_llm_proxy
    llm = create_background_llm_proxy("cognitive-drift")
    response = llm.send_message(system_prompt, user_message)
    if response:
        text = response.text
"""

import json
import time
import uuid
import logging
from typing import Optional

from services.redis_client import RedisClientService
from services.llm_service import LLMResponse

logger = logging.getLogger(__name__)

QUEUE_KEY = "bg_llm:queue"
RESULT_KEY_PREFIX = "bg_llm:result:"
HEARTBEAT_KEY = "bg_llm:last_heartbeat"

MAX_QUEUE_DEPTH = 25
RESULT_WAIT_TIMEOUT = 180  # seconds caller blocks waiting for result
HEARTBEAT_STALE_THRESHOLD = 30  # seconds before logging critical warning


class BackgroundLLMProxy:
    """
    Drop-in proxy for RefreshableLLMService in background services.

    Routes LLM calls through the bg_llm:queue Redis list, consumed by the single
    background-llm-worker process. All background LLM calls are serialized,
    preventing thundering herd on the provider.

    Interface is identical to RefreshableLLMService: .send_message() → LLMResponse.
    Returns None on queue full, timeout, or error — callers already handle None.
    """

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self._redis = RedisClientService.create_connection()

    def send_message(
        self,
        system_prompt: str,
        user_message: str,
        stream: bool = False,
    ) -> Optional[LLMResponse]:
        """
        Enqueue the LLM call and block until the worker returns a result.

        Returns LLMResponse on success, None on queue full / timeout / error.
        """
        # Watchdog: warn if worker heartbeat is stale
        try:
            heartbeat = self._redis.get(HEARTBEAT_KEY)
            if heartbeat and (time.time() - float(heartbeat)) > HEARTBEAT_STALE_THRESHOLD:
                logger.critical(
                    "BG LLM worker heartbeat stale (>%ds) — worker may be down",
                    HEARTBEAT_STALE_THRESHOLD,
                )
        except Exception:
            pass  # never let a heartbeat check block the caller

        # Queue depth guard — drop if backlog is too large
        try:
            depth = self._redis.llen(QUEUE_KEY)
            if depth >= MAX_QUEUE_DEPTH:
                logger.warning(
                    "BG queue full (depth=%d, max=%d) — dropped job for %s",
                    depth, MAX_QUEUE_DEPTH, self.agent_name,
                )
                return None
        except Exception as e:
            logger.warning("BG LLM queue depth check failed: %s", e)

        job_id = str(uuid.uuid4())
        job = {
            "job_id": job_id,
            "agent_name": self.agent_name,
            "system_prompt": system_prompt,
            "user_message": user_message,
            "enqueued_at": time.time(),
            "retry_count": 0,
        }

        try:
            self._redis.rpush(QUEUE_KEY, json.dumps(job))
        except Exception as e:
            logger.error("BG LLM enqueue failed: %s", e)
            return None

        # Block waiting for the worker to push the result
        result_key = f"{RESULT_KEY_PREFIX}{job_id}"
        try:
            raw = self._redis.brpop(result_key, timeout=RESULT_WAIT_TIMEOUT)
        except Exception as e:
            logger.error("BG LLM result wait failed: %s", e)
            return None

        if raw is None:
            logger.warning(
                "BG LLM timeout after %ds (agent=%s, job=%s)",
                RESULT_WAIT_TIMEOUT, self.agent_name, job_id,
            )
            return None

        _, payload = raw
        try:
            data = json.loads(payload)
        except Exception as e:
            logger.error("BG LLM result deserialize failed: %s", e)
            return None

        if "error" in data:
            logger.warning(
                "BG LLM job error (agent=%s, error=%s)", self.agent_name, data["error"]
            )
            return None

        return LLMResponse(
            text=data.get("text", ""),
            model=data.get("model", ""),
            provider=data.get("provider"),
            tokens_input=data.get("tokens_input"),
            tokens_output=data.get("tokens_output"),
            latency_ms=data.get("latency_ms"),
        )


def create_background_llm_proxy(agent_name: str) -> BackgroundLLMProxy:
    """
    Factory matching create_refreshable_llm_service() signature.

    Drop-in replacement for background services that do not need
    low-latency LLM access. All calls are serialized through the
    background queue with adaptive back-off and retry.
    """
    return BackgroundLLMProxy(agent_name)
