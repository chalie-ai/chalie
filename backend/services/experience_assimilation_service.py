# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Experience Assimilation Service — Tool output → episodic memory.

After every ACT loop that used tools, this service reflects on whether the
outputs contain novel information worth remembering. A multi-layer novelty
gate eliminates ephemeral or trivial outputs before any LLM call. Valuable
observations are stored as episodes and flow through the existing semantic
consolidation pipeline into concepts/relationships.

Flow:
  tool_worker pushes tool outputs to Redis list (tool_reflection:pending)
  → this service pops items every check_interval seconds
  → novelty gate layer 3: content hash dedup (layers 1+2 happen at enqueue time)
  → LLM reflection: "anything novel & conversationally useful?"
  → observation dedup check
  → store as episode with durability tag → semantic consolidation → concepts
"""

import hashlib
import json
import logging
import time
from typing import Optional

from services.redis_client import RedisClientService
from services.config_service import ConfigService

logger = logging.getLogger(__name__)

LOG_PREFIX = "[EXPERIENCE ASSIMILATION]"

PENDING_LIST_KEY = "tool_reflection:pending"
STATE_KEY = "experience_assimilation_state"
COOLDOWN_ZSET_KEY = "experience_assimilation_cooldowns"

SALIENCE_MAP = {
    'stable': 6,
    'evolving': 5,
    'transient': 4,
}


class ExperienceAssimilationService:
    """
    Idle-time service that reflects on tool outputs and extracts episodic memories.
    """

    def __init__(self, check_interval: int = 60):
        self.redis = RedisClientService.create_connection()
        self.config = ConfigService.resolve_agent_config("experience-assimilation")
        self.check_interval = self.config.get("check_interval", check_interval)
        self.max_sessions = self.config.get("max_sessions_per_day", 20)
        self.cooldown_per_topic = self.config.get("cooldown_per_topic", 300)
        self.dedup_ttl = self.config.get("observation_dedup_ttl", 86400)

        self.llm_config = self.config
        self.prompt_template = ConfigService.get_agent_prompt("tool-reflection")

        logger.info(
            f"{LOG_PREFIX} Service initialized "
            f"(interval={self.check_interval}s, max_sessions={self.max_sessions}/day)"
        )

    def run(self, shared_state: Optional[dict] = None) -> None:
        """Main service loop."""
        logger.info(f"{LOG_PREFIX} Service started")

        while True:
            try:
                time.sleep(self.check_interval)

                if self._daily_sessions_exceeded():
                    continue

                self._run_cycle()

            except KeyboardInterrupt:
                logger.info(f"{LOG_PREFIX} Service shutting down...")
                break
            except Exception as e:
                logger.error(f"{LOG_PREFIX} Error: {e}", exc_info=True)
                time.sleep(60)

    def _daily_sessions_exceeded(self) -> bool:
        """Check if we've hit the daily session cap."""
        count = int(self.redis.hget(STATE_KEY, 'sessions_today') or 0)
        day_key = self.redis.hget(STATE_KEY, 'session_day')
        today = time.strftime('%Y-%m-%d')

        if day_key != today:
            self.redis.hset(STATE_KEY, mapping={
                'sessions_today': 0,
                'session_day': today,
            })
            return False

        return count >= self.max_sessions

    def _is_topic_on_cooldown(self, topic: str) -> bool:
        """Check per-topic cooldown."""
        last_time = self.redis.zscore(COOLDOWN_ZSET_KEY, topic)
        if last_time and (time.time() - float(last_time)) < self.cooldown_per_topic:
            return True
        return False

    def _mark_topic_processed(self, topic: str):
        """Record topic cooldown."""
        self.redis.zadd(COOLDOWN_ZSET_KEY, {topic: time.time()})

    def _content_hash(self, tool_outputs: list) -> str:
        """Hash tool outputs for dedup (novelty gate layer 3)."""
        combined = json.dumps(tool_outputs, sort_keys=True)
        return hashlib.md5(combined.encode()).hexdigest()[:16]

    def _is_content_seen(self, content_hash: str) -> bool:
        """Check if identical content was processed in the last 24h."""
        key = f"tool_reflection:hash:{content_hash}"
        if self.redis.exists(key):
            return True
        self.redis.setex(key, 86400, "1")
        return False

    def _is_duplicate_observation(self, observation_text: str) -> bool:
        """Check if we've already stored a similar observation."""
        obs_hash = hashlib.md5(observation_text.lower().strip().encode()).hexdigest()[:12]
        key = f"tool_reflection:obs:{obs_hash}"
        if self.redis.exists(key):
            return True
        self.redis.setex(key, self.dedup_ttl, "1")
        return False

    def _run_cycle(self):
        """Pop and process up to 3 items from the pending list."""
        for _ in range(3):
            raw = self.redis.lpop(PENDING_LIST_KEY)
            if not raw:
                break

            try:
                item = json.loads(raw)
                self._process_item(item)
            except Exception as e:
                logger.error(f"{LOG_PREFIX} Failed to process item: {e}")

    def _process_item(self, item: dict):
        """Process one tool reflection item."""
        topic = item.get('topic', 'general')
        user_prompt = item.get('user_prompt', '')
        tool_outputs = item.get('tool_outputs', [])

        if not tool_outputs:
            return

        if self._is_topic_on_cooldown(topic):
            logger.debug(f"{LOG_PREFIX} Topic '{topic}' on cooldown, skipping")
            return

        # Novelty gate layer 3: content hash dedup
        content_hash = self._content_hash(tool_outputs)
        if self._is_content_seen(content_hash):
            logger.debug(f"{LOG_PREFIX} Content hash {content_hash} already seen, skipping")
            return

        # Run LLM reflection
        try:
            reflected = self._reflect(user_prompt, tool_outputs)
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} LLM reflection failed: {e}")
            return

        if not reflected.get('worth_reflecting'):
            logger.debug(f"{LOG_PREFIX} Topic '{topic}': nothing worth reflecting")
            return

        observations = reflected.get('observations', [])
        if not observations:
            return

        logger.info(
            f"{LOG_PREFIX} Topic '{topic}': {len(observations)} observation(s) to store"
        )

        stored = 0
        for obs in observations[:3]:
            obs_text = obs.get('text', '').strip()
            if not obs_text:
                continue

            if self._is_duplicate_observation(obs_text):
                logger.debug(f"{LOG_PREFIX} Duplicate observation skipped: {obs_text[:60]}")
                continue

            try:
                self._store_episode(obs, topic, user_prompt, tool_outputs)
                stored += 1
            except Exception as e:
                logger.error(f"{LOG_PREFIX} Failed to store episode: {e}")

        if stored > 0:
            self._mark_topic_processed(topic)
            self.redis.hincrby(STATE_KEY, 'sessions_today', 1)
            logger.info(f"{LOG_PREFIX} Stored {stored} episode(s) for topic '{topic}'")

    def _reflect(self, user_prompt: str, tool_outputs: list) -> dict:
        """Call LLM to evaluate tool outputs for novel knowledge."""
        from services.background_llm_queue import create_background_llm_proxy

        tool_outputs_text = "\n\n".join(
            f"[{o['tool']}]\n{o['result']}" for o in tool_outputs
        )

        prompt = self.prompt_template \
            .replace('{{user_prompt}}', user_prompt) \
            .replace('{{tool_outputs}}', tool_outputs_text)

        llm = create_background_llm_proxy("experience-assimilation")
        response = llm.send_message("", prompt).text

        return json.loads(response)

    def _store_episode(self, observation: dict, topic: str, user_prompt: str, tool_outputs: list):
        """Store a reflection observation as an episodic memory."""
        from services.database_service import get_lightweight_db_service
        from services.episodic_storage_service import EpisodicStorageService
        from services.embedding_service import get_embedding_service

        obs_text = observation['text']
        durability = observation.get('durability', 'evolving')
        salience = SALIENCE_MAP.get(durability, 5)

        tool_names = [o['tool'] for o in tool_outputs]

        embedding_service = get_embedding_service()
        embedding = embedding_service.generate_embedding(obs_text)

        episode_data = {
            'intent': {'type': 'tool_reflection', 'query': user_prompt[:200]},
            'context': f"Tool outputs from: {', '.join(tool_names)}",
            'action': 'reflected_on_tool_output',
            'emotion': 'neutral',
            'outcome': obs_text,
            'gist': obs_text,
            'salience': salience,
            'freshness': 1.0,
            'topic': topic,
            'embedding': embedding,
            'salience_factors': {
                'source': 'tool_reflection',
                'durability': durability,
                'retrieval_count': 0,
                'reference_count': 0,
                'initial_confidence': 0.6,
                'contradicts_existing': False,
            },
        }

        db_service = get_lightweight_db_service()
        try:
            storage = EpisodicStorageService(db_service)
            episode_id = storage.store_episode(episode_data)

            # Trigger profile enrichment for high-salience episodes
            if salience >= 5 and embedding is not None:
                try:
                    from services.tool_profile_service import ToolProfileService
                    ToolProfileService().check_episode_relevance(embedding, str(episode_id))
                except Exception as _enrich_err:
                    logger.warning(f"{LOG_PREFIX} Profile enrichment check failed: {_enrich_err}")

            logger.info(
                f"{LOG_PREFIX} Stored episode {episode_id}: "
                f"'{obs_text[:80]}' (durability={durability}, salience={salience})"
            )
        finally:
            db_service.close_pool()


def experience_assimilation_worker(shared_state: Optional[dict] = None):
    """Entry point for consumer.py service registration."""
    service = ExperienceAssimilationService()
    service.run(shared_state)
