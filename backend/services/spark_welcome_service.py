"""
Spark Welcome Service — Generates and delivers the first-contact welcome message.

Chalie speaks first. This service handles:
1. Checking if a welcome is needed (via SparkStateService)
2. Generating the welcome via LLM with fallback variants
3. Delivering via OutputService (drift stream path)
4. Atomic lock to prevent duplicate welcomes across tabs

Delivery is triggered from proactive.py after SSE subscription.
"""

import json
import logging
import random
import time
import threading
from typing import Optional

from services.redis_client import RedisClientService
from services.spark_state_service import SparkStateService

logger = logging.getLogger(__name__)

LOG_PREFIX = "[SPARK WELCOME]"

# Hardcoded fallback variants (used if LLM takes >8s)
_FALLBACK_VARIANTS = {
    'A': (
        "Hey \u2014 I noticed you're here. I'm around if you want to chat, "
        "or just hang out quietly for a bit."
    ),
    'B': (
        "There's something nice about a fresh start. I'm Chalie \u2014 no agenda, "
        "just here whenever you feel like talking."
    ),
    'C': (
        "Hi \u2014 I'm Chalie. I find the best conversations happen when "
        "neither side is trying too hard. Take your time."
    ),
}

# Lock key for cross-tab dedup
_WELCOME_LOCK_KEY = "spark_welcome_lock:{user_id}"
_WELCOME_LOCK_TTL = 30  # seconds


class SparkWelcomeService:
    """Generates and delivers the first-contact welcome message."""

    def __init__(self, user_id: str = 'primary'):
        self._user_id = user_id
        self._spark_state = SparkStateService(user_id=user_id)

    def maybe_send_welcome(self) -> bool:
        """
        Check if welcome is needed and send it if so.

        Uses atomic Redis lock to prevent duplicate sends across tabs.
        Generates welcome via LLM with tiered fallback.

        Returns:
            bool: True if welcome was sent, False otherwise
        """
        if not self._spark_state.needs_welcome():
            return False

        # Atomic lock: only one tab/process should generate the welcome
        redis = RedisClientService.create_connection()
        lock_key = _WELCOME_LOCK_KEY.format(user_id=self._user_id)

        if not redis.setnx(lock_key, '1'):
            logger.debug(f"{LOG_PREFIX} Lock already held, skipping")
            return False

        redis.expire(lock_key, _WELCOME_LOCK_TTL)

        try:
            # Generate welcome message with tiered fallback
            welcome_text, variant_used = self._generate_welcome()

            if not welcome_text:
                logger.warning(f"{LOG_PREFIX} Failed to generate welcome")
                redis.delete(lock_key)
                return False

            # Deliver via OutputService
            self._deliver_welcome(welcome_text)

            # Mark welcome as sent
            self._spark_state.mark_welcome_sent()

            # Log the event
            self._log_welcome_event(variant_used)

            logger.info(f"{LOG_PREFIX} Welcome sent (variant={variant_used})")
            return True

        except Exception as e:
            logger.error(f"{LOG_PREFIX} Failed to send welcome: {e}")
            redis.delete(lock_key)
            return False

    def _generate_welcome(self) -> tuple:
        """
        Generate welcome message with tiered latency fallback.

        Returns:
            tuple: (welcome_text, variant_label)
                variant_label is 'llm', 'A', 'B', or 'C'
        """
        fallback_key = random.choice(list(_FALLBACK_VARIANTS.keys()))
        fallback_text = _FALLBACK_VARIANTS[fallback_key]

        # Try LLM generation with timeout
        llm_result = [None]
        llm_done = threading.Event()

        def _llm_generate():
            try:
                from services.config_service import ConfigService
                from services.llm_service import create_llm_service

                # Use acknowledge config (lightweight, fast)
                try:
                    config = ConfigService.resolve_agent_config("frontal-cortex-acknowledge")
                except Exception:
                    config = ConfigService.resolve_agent_config("frontal-cortex")

                # Override format to plain text (not JSON)
                config = dict(config)
                config['format'] = ''

                prompt = ConfigService.get_agent_prompt("spark-welcome")
                llm = create_llm_service(config)
                response = llm.send_message(prompt, "Generate a welcome message.").text

                # Clean up response (strip quotes, whitespace)
                text = response.strip().strip('"').strip("'").strip()
                if text:
                    llm_result[0] = text
            except Exception as e:
                logger.warning(f"{LOG_PREFIX} LLM generation failed: {e}")
            finally:
                llm_done.set()

        thread = threading.Thread(target=_llm_generate, daemon=True)
        thread.start()

        # Wait up to 8s for LLM
        llm_done.wait(timeout=8.0)

        if llm_result[0]:
            return (llm_result[0], 'llm')

        # Fallback
        logger.info(f"{LOG_PREFIX} Using fallback variant {fallback_key}")
        return (fallback_text, fallback_key)

    def _deliver_welcome(self, text: str) -> None:
        """Deliver welcome message via OutputService (drift stream)."""
        from services.output_service import OutputService

        output_svc = OutputService()
        output_svc.enqueue_text(
            topic='spark_welcome',
            response=text,
            mode='RESPOND',
            confidence=1.0,
            generation_time=0.0,
            original_metadata={
                'source': 'spark_welcome',
            },
        )

    def _log_welcome_event(self, variant: str) -> None:
        """Log welcome event to interaction_log."""
        try:
            from services.database_service import get_shared_db_service
            from services.interaction_log_service import InteractionLogService

            db = get_shared_db_service()
            log_service = InteractionLogService(db)
            log_service.log_event(
                event_type='spark_welcome_sent',
                payload={
                    'variant': variant,
                },
                topic='spark_welcome',
                source='spark_welcome',
            )
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Welcome event logging failed: {e}")
