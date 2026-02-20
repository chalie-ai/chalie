# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Cognitive Drift Engine — spontaneous thought generation during idle periods.

Models the Default Mode Network: when the brain is idle, residual activation
from recent experience, emotional salience, and semantic associations produce
fleeting internal thoughts (reflections, questions, hypotheses).

These are stored as drift gists and naturally surface in frontal cortex context.

After each thought, an ActionDecisionRouter evaluates what to do with it:
  - COMMUNICATE: Share with the user proactively (quality + timing + engagement gated)
  - REFLECT: Internal enrichment — store association-linked gist, boost concepts
  - NOTHING: Let the gist live in Redis as before (reactive surfacing only)
  - Future: USE_SKILL, PLAN, LEARN
"""

import json
import math
import time
import random
import uuid
import logging
from typing import Optional, List, Dict, Any

from services.redis_client import RedisClientService
from services.config_service import ConfigService
from services.gist_storage_service import GistStorageService
from services.semantic_storage_service import SemanticStorageService
from services.semantic_retrieval_service import SemanticRetrievalService
from services.episodic_retrieval_service import EpisodicRetrievalService
from services.embedding_service import EmbeddingService
from services.llm_service import create_llm_service
from services.database_service import DatabaseService, get_merged_db_config

logger = logging.getLogger(__name__)

LOG_PREFIX = "[COGNITIVE DRIFT]"

# Redis keys
COOLDOWN_KEY = "cognitive_drift_concept_cooldowns"
FATIGUE_KEY = "cognitive_drift_activations"
STATE_KEY = "cognitive_drift_state"


class CognitiveDriftEngine:
    """
    Background service that generates spontaneous thoughts during idle periods.

    Selects seed concepts via weighted random strategy, runs spreading activation,
    synthesizes a brief thought via LLM, and stores it as a drift gist.
    """

    def __init__(self, check_interval: int = 300):
        self.redis = RedisClientService.create_connection()
        self.config = ConfigService.resolve_agent_config("cognitive-drift")
        self.check_interval = check_interval

        # Load queue names for idle check
        conn_config = ConfigService.connections()
        topics = conn_config.get("redis", {}).get("topics", {})
        self.prompt_queue = topics.get("prompt_queue", "prompt-queue")
        self.memory_queue = topics.get("memory_chunker", "memory-chunker-queue")
        self.episodic_queue = topics.get("episodic_memory", "episodic-memory-queue")
        self.semantic_queue = conn_config.get("redis", {}).get("queues", {}).get(
            "semantic_consolidation_queue", {}
        ).get("name", "semantic_consolidation_queue")

        # Database + services
        db_config = get_merged_db_config()
        self.db_service = DatabaseService(db_config)
        self.embedding_service = EmbeddingService()
        self.semantic_storage = SemanticStorageService(self.db_service)
        self.semantic_retrieval = SemanticRetrievalService(
            self.db_service, self.embedding_service, self.semantic_storage
        )

        episodic_config = ConfigService.resolve_agent_config("episodic-memory")
        self.episodic_retrieval = EpisodicRetrievalService(self.db_service, episodic_config)

        # Gist storage
        self.gist_storage = GistStorageService()

        # LLM for thought synthesis (provider resolved from cognitive-drift config)
        self.ollama = create_llm_service(self.config)

        # Load prompt template + soul axioms
        self.prompt_template = ConfigService.get_agent_prompt("cognitive-drift")
        self.soul_axioms = ConfigService.get_agent_prompt("soul")

        # Config values
        self.seed_weights = self.config.get('seed_weights', {
            'decaying': 0.40, 'recent': 0.30, 'salient': 0.20, 'random': 0.10
        })
        self.max_activation_depth = self.config.get('max_activation_depth', 2)
        self.max_activated_concepts = self.config.get('max_activated_concepts', 5)
        self.episode_lookback_hours = self.config.get('episode_lookback_hours', 168)
        self.episode_recency_decay = self.config.get('episode_recency_decay', 0.02)
        self.salience_lookback_days = self.config.get('salience_lookback_days', 7)
        self.gist_confidence = self.config.get('gist_confidence', 8)
        self.cooldown_minutes = self.config.get('concept_cooldown_minutes', 60)
        self.jitter_range = self.config.get('jitter_range', [0.7, 1.3])
        self.fatigue_budget = self.config.get('fatigue_budget', 2.5)
        self.fatigue_window_minutes = self.config.get('fatigue_window_minutes', 30)
        self.long_gap_probability = self.config.get('long_gap_probability', 0.1)
        self.min_activation_energy = self.config.get('min_activation_energy', 0.4)
        self.decaying_reinforce_bump = self.config.get('decaying_reinforce_bump', 0.1)

        # Initialize autonomous action router
        self.action_router = self._init_action_router()

        logger.info(
            f"{LOG_PREFIX} Engine initialized "
            f"(check_interval={check_interval}s, "
            f"seed_weights={self.seed_weights})"
        )

    def _init_action_router(self):
        """Initialize the autonomous action decision router."""
        from services.autonomous_actions.decision_router import ActionDecisionRouter
        from services.autonomous_actions.nothing_action import NothingAction
        from services.autonomous_actions.communicate_action import CommunicateAction
        from services.autonomous_actions.reflect_action import ReflectAction

        router = ActionDecisionRouter()

        # Always register NOTHING (fallback)
        router.register(NothingAction())

        action_config = self.config.get('autonomous_actions', {})

        # Register COMMUNICATE if enabled
        communicate_config = action_config.get('communicate', {})
        if communicate_config.get('enabled', True):
            communicate = CommunicateAction(config=communicate_config)
            router.register(communicate)
            self._communicate_action = communicate
        else:
            self._communicate_action = None

        # Register REFLECT if enabled
        reflect_config = action_config.get('reflect', {})
        if reflect_config.get('enabled', True):
            reflect_config['total_fatigue_budget'] = self.fatigue_budget
            reflect_config['fatigue_window_minutes'] = self.fatigue_window_minutes
            reflect = ReflectAction(
                config=reflect_config,
                services={
                    'gist_storage': self.gist_storage,
                    'embedding_service': self.embedding_service,
                    'db_service': self.db_service,
                },
            )
            router.register(reflect)

        return router

    def run(self, shared_state: Optional[dict] = None) -> None:
        """Main service loop."""
        logger.info(f"{LOG_PREFIX} Service started")

        while True:
            try:
                # Check for deferred thoughts from quiet hours
                self._process_deferred()

                interval = self._next_interval()
                time.sleep(interval)

                if not self._are_workers_idle():
                    logger.debug(f"{LOG_PREFIX} Workers not idle, skipping")
                    continue

                if not self._has_recent_episodes():
                    logger.debug(f"{LOG_PREFIX} No recent episodes, skipping drift")
                    continue

                self._run_drift_cycle()

            except KeyboardInterrupt:
                logger.info(f"{LOG_PREFIX} Service shutting down...")
                break
            except Exception as e:
                logger.error(f"{LOG_PREFIX} Error: {e}", exc_info=True)
                logger.info(f"{LOG_PREFIX} Waiting 1 minute before retry...")
                time.sleep(60)

    def _process_deferred(self):
        """Check and process deferred thoughts from quiet hours."""
        if not self._communicate_action:
            return
        try:
            deferred = self._communicate_action.process_deferred_queue()
            if deferred:
                from services.autonomous_actions.base import ThoughtContext
                context = ThoughtContext(
                    thought_type=deferred.get('type', 'reflection'),
                    thought_content=deferred.get('content', ''),
                    activation_energy=deferred.get('activation_energy', 0.5),
                    seed_concept=deferred.get('seed_concept', ''),
                    seed_topic=deferred.get('topic', 'general'),
                    thought_embedding=deferred.get('embedding'),
                )
                result = self._communicate_action.execute(context)
                self._log_action_result('COMMUNICATE', context, result, source='deferred')
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Deferred processing failed: {e}")

    # ── Precondition checks ──────────────────────────────────────────

    def _are_workers_idle(self) -> bool:
        """Check if all worker queues are empty."""
        queues = [
            self.prompt_queue,
            self.memory_queue,
            self.episodic_queue,
            self.semantic_queue,
        ]
        for queue_name in queues:
            if self.redis.llen(queue_name) > 0:
                return False
        return True

    def _has_recent_episodes(self) -> bool:
        """Check if at least 1 episode was created in the lookback window."""
        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM episodes
                    WHERE deleted_at IS NULL
                      AND created_at > NOW() - INTERVAL '%s hours'
                    """,
                    (self.episode_lookback_hours,)
                )
                count = cursor.fetchone()[0]
                cursor.close()
                return count > 0
        except Exception as e:
            logger.error(f"{LOG_PREFIX} Failed to check recent episodes: {e}")
            return False

    # ── Timing ───────────────────────────────────────────────────────

    def _next_interval(self) -> float:
        """Stochastic jitter with occasional long gaps."""
        base = self.check_interval
        if random.random() < self.long_gap_probability:
            return base * random.uniform(1.8, 2.5)
        return base * random.uniform(self.jitter_range[0], self.jitter_range[1])

    # ── Fatigue ──────────────────────────────────────────────────────

    def _is_fatigued(self) -> bool:
        """Check if cumulative drift activation exceeds budget."""
        cutoff = time.time() - (self.fatigue_window_minutes * 60)
        recent = self.redis.zrangebyscore(FATIGUE_KEY, cutoff, '+inf', withscores=True)

        total_activation = 0.0
        for member, _ in recent:
            try:
                total_activation += float(member.split(':')[1])
            except (IndexError, ValueError):
                continue

        if total_activation >= self.fatigue_budget:
            logger.info(
                f"{LOG_PREFIX} Fatigued (activation={total_activation:.2f}, "
                f"budget={self.fatigue_budget})"
            )
            return True
        return False

    def _record_drift_activation(self, drift_id: str, max_activation: float):
        """Record drift activation for fatigue tracking."""
        self.redis.zadd(FATIGUE_KEY, {f"{drift_id}:{max_activation}": time.time()})
        cutoff = time.time() - (self.fatigue_window_minutes * 60)
        self.redis.zremrangebyscore(FATIGUE_KEY, '-inf', cutoff)

    # ── Cooldown ─────────────────────────────────────────────────────

    def _is_on_cooldown(self, concept_id: str) -> bool:
        """Check if a concept was recently used in drift."""
        last_used = self.redis.zscore(COOLDOWN_KEY, concept_id)
        if last_used and (time.time() - last_used) < self.cooldown_minutes * 60:
            return True
        return False

    def _mark_used(self, concept_id: str):
        """Record concept as recently used."""
        self.redis.zadd(COOLDOWN_KEY, {concept_id: time.time()})
        cutoff = time.time() - (self.cooldown_minutes * 60)
        self.redis.zremrangebyscore(COOLDOWN_KEY, '-inf', cutoff)

    # ── Activation energy ────────────────────────────────────────────

    def _has_sufficient_activation(self, activated_concepts: list) -> bool:
        """Check if spreading activation produced strong enough results."""
        if not activated_concepts:
            return False
        max_activation = max(c.get('activation_score', 0) for c in activated_concepts)
        return max_activation >= self.min_activation_energy

    # ── Seed selection ───────────────────────────────────────────────

    def _select_seed(self) -> Optional[Dict[str, Any]]:
        """
        Weighted random seed selection with cooldown retry.

        Strategies: decaying (40%), recent (30%), salient (20%), random (10%).
        Retries up to 3 times on cooldown hit.
        """
        strategies = list(self.seed_weights.keys())
        weights = list(self.seed_weights.values())

        for attempt in range(3):
            strategy = random.choices(strategies, weights=weights, k=1)[0]
            seed = self._select_seed_by_strategy(strategy)

            if seed and not self._is_on_cooldown(seed['concept_id']):
                logger.info(
                    f"{LOG_PREFIX} Selected seed: '{seed['concept_name']}' "
                    f"(strategy={strategy}, attempt={attempt + 1})"
                )
                return seed

            if seed:
                logger.debug(
                    f"{LOG_PREFIX} Seed '{seed['concept_name']}' on cooldown, retrying"
                )

        logger.info(f"{LOG_PREFIX} No viable seed found after 3 attempts")
        return None

    def _select_seed_by_strategy(self, strategy: str) -> Optional[Dict[str, Any]]:
        """Execute a specific seed selection strategy."""
        try:
            if strategy == 'decaying':
                return self._seed_decaying()
            elif strategy == 'recent':
                return self._seed_recent()
            elif strategy == 'salient':
                return self._seed_salient()
            elif strategy == 'random':
                return self._seed_random()
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Seed strategy '{strategy}' failed: {e}")
        return None

    def _seed_decaying(self) -> Optional[Dict[str, Any]]:
        """Select a concept whose strength is fading but not dead."""
        with self.db_service.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, concept_name, definition, strength, domain
                FROM semantic_concepts
                WHERE deleted_at IS NULL
                  AND strength > 0.2 AND strength < 2.0
                ORDER BY strength ASC, last_reinforced_at ASC
                LIMIT 5
            """)
            rows = cursor.fetchall()
            cursor.close()

        if not rows:
            return None

        row = random.choice(rows)
        return {
            'concept_id': str(row[0]),
            'concept_name': row[1],
            'definition': row[2],
            'seed_type': 'decaying',
            'topic': row[4] or 'general',
        }

    def _seed_recent(self) -> Optional[Dict[str, Any]]:
        """Select a concept from a recent episode, weighted toward more recent ones."""
        with self.db_service.connection() as conn:
            cursor = conn.cursor()

            # Fetch top-20 recent episodes within lookback window
            cursor.execute("""
                SELECT id,
                       EXTRACT(EPOCH FROM (NOW() - created_at)) / 3600.0 AS age_hours
                FROM episodes
                WHERE deleted_at IS NULL
                  AND created_at > NOW() - INTERVAL '%s hours'
                ORDER BY created_at DESC
                LIMIT 20
            """, (self.episode_lookback_hours,))
            episodes = cursor.fetchall()
            if not episodes:
                cursor.close()
                return None

            # Gradient sample: weight = exp(-decay * age_hours), most recent heaviest
            weights = [math.exp(-self.episode_recency_decay * float(row[1])) for row in episodes]
            episode_row = random.choices(episodes, weights=weights, k=1)[0]
            episode_id = str(episode_row[0])

            # Find concepts linked to the sampled episode
            cursor.execute("""
                SELECT id, concept_name, definition, strength, domain
                FROM semantic_concepts
                WHERE deleted_at IS NULL
                  AND source_episodes::jsonb @> to_jsonb(ARRAY[%s]::text[])
                ORDER BY strength DESC
                LIMIT 5
            """, (episode_id,))
            rows = cursor.fetchall()
            cursor.close()

        if not rows:
            return None

        row = random.choice(rows)
        return {
            'concept_id': str(row[0]),
            'concept_name': row[1],
            'definition': row[2],
            'seed_type': 'recent',
            'topic': row[4] or 'general',
        }

    def _seed_salient(self) -> Optional[Dict[str, Any]]:
        """Select a concept from a high-salience episode, weighted by salience × recency."""
        with self.db_service.connection() as conn:
            cursor = conn.cursor()

            # Fetch top-20 salient episodes within lookback window
            cursor.execute("""
                SELECT id, gist, topic,
                       EXTRACT(EPOCH FROM (NOW() - created_at)) / 3600.0 AS age_hours,
                       salience
                FROM episodes
                WHERE deleted_at IS NULL
                  AND created_at > NOW() - INTERVAL '%s days'
                ORDER BY salience DESC
                LIMIT 20
            """, (self.salience_lookback_days,))
            episodes = cursor.fetchall()
            cursor.close()

        if not episodes:
            return None

        # Combined weight: salience × exp(-decay * age_hours)
        weights = [
            max(float(row[4] or 0.1), 0.01) * math.exp(-self.episode_recency_decay * float(row[3]))
            for row in episodes
        ]
        episode_row = random.choices(episodes, weights=weights, k=1)[0]
        episode_id = str(episode_row[0])
        episode_gist = episode_row[1]
        episode_topic = episode_row[2]

        # Use episode gist as embedding query against concepts
        concepts = self.semantic_retrieval.retrieve_concepts(episode_gist, limit=5)

        if not concepts:
            return None

        concept = random.choice(concepts)
        return {
            'concept_id': concept['id'],
            'concept_name': concept['concept_name'],
            'definition': concept['definition'],
            'seed_type': 'salient',
            'topic': episode_topic or concept.get('domain', 'general'),
        }

    def _seed_random(self) -> Optional[Dict[str, Any]]:
        """Pure random pick from active concepts."""
        with self.db_service.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, concept_name, definition, domain
                FROM semantic_concepts
                WHERE deleted_at IS NULL AND confidence >= 0.4
                ORDER BY RANDOM()
                LIMIT 1
            """)
            row = cursor.fetchone()
            cursor.close()

        if not row:
            return None

        return {
            'concept_id': str(row[0]),
            'concept_name': row[1],
            'definition': row[2],
            'seed_type': 'random',
            'topic': row[3] or 'general',
        }

    # ── Decaying concept reinforcement ───────────────────────────────

    def _maybe_reinforce_seed(self, seed: Dict, drift_succeeded: bool):
        """Small strength bump for decaying seeds on successful drift only."""
        if seed['seed_type'] != 'decaying':
            return

        if not drift_succeeded:
            return

        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE semantic_concepts
                    SET strength = LEAST(10.0, strength + %s),
                        updated_at = NOW()
                    WHERE id = %s AND deleted_at IS NULL
                """, (self.decaying_reinforce_bump, seed['concept_id']))
                cursor.close()

            logger.debug(
                f"{LOG_PREFIX} Nudged decaying concept '{seed['concept_name']}' "
                f"by +{self.decaying_reinforce_bump}"
            )
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Failed to reinforce seed: {e}")

    # ── Thought synthesis ────────────────────────────────────────────

    def _synthesize_thought(self, seed: Dict, activated: List[Dict],
                            episode: Optional[Dict]) -> Optional[Dict]:
        """
        Call LLM to produce a brief internal thought.

        Returns:
            Dict with 'type' and 'content', or None on failure.
        """
        # Build prompt from template
        seed_text = f"{seed['concept_name']}: {seed['definition']}"

        activated_text = "\n".join(
            f"- {c['concept_name']}: {c.get('definition', 'N/A')} "
            f"(activation: {c.get('activation_score', 0):.2f})"
            for c in activated[:self.max_activated_concepts]
            if c['id'] != seed['concept_id']
        ) or "No additional associations."

        episode_text = "No related experience recalled."
        if episode:
            episode_text = (
                f"Topic: {episode.get('topic', 'unknown')}\n"
                f"Gist: {episode.get('gist', 'N/A')}\n"
                f"Outcome: {episode.get('outcome', 'N/A')}"
            )

        user_message = self.prompt_template \
            .replace("{{seed_concept}}", seed_text) \
            .replace("{{activated_concepts}}", activated_text) \
            .replace("{{grounding_episode}}", episode_text)

        # Soul axioms appended as stability anchor
        system_prompt = self.soul_axioms

        max_retries = 3
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                raw_response = self.ollama.send_message(system_prompt, user_message).text

                # Parse JSON response
                # Strip <think>...</think> tags (qwen3 thinking leakage)
                import re
                cleaned = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL).strip()

                if not cleaned:
                    logger.warning(
                        f"{LOG_PREFIX} Empty LLM response (attempt {attempt}/{max_retries}), "
                        f"raw length={len(raw_response)}"
                    )
                    last_error = ValueError("Empty response from LLM")
                    if attempt < max_retries:
                        time.sleep(1)
                        continue
                    break

                thought = json.loads(cleaned)

                thought_type = thought.get('type', '').strip()
                content = thought.get('content', '').strip()

                if thought_type not in ('reflection', 'question', 'hypothesis'):
                    logger.warning(f"{LOG_PREFIX} Invalid thought type: '{thought_type}'")
                    return None

                if not content or len(content) < 5:
                    logger.warning(f"{LOG_PREFIX} Thought content too short or empty")
                    return None

                return {'type': thought_type, 'content': content}

            except (json.JSONDecodeError, KeyError, TypeError) as e:
                last_error = e
                if attempt < max_retries:
                    logger.debug(f"{LOG_PREFIX} Parse failed (attempt {attempt}/{max_retries}), retrying: {e}")
                    time.sleep(1)
                    continue
            except Exception as e:
                logger.error(f"{LOG_PREFIX} LLM call failed: {e}", exc_info=True)
                return None

        logger.warning(f"{LOG_PREFIX} Failed to parse LLM response after {max_retries} attempts: {last_error}")
        return None

    # ── Gist storage ─────────────────────────────────────────────────

    def _store_drift(self, topic: str, drift_type: str, thought: str):
        """Store drift thought as a gist."""
        self.gist_storage.store_gists(
            topic=topic,
            gists=[{
                'content': f"[{drift_type}] {thought}",
                'type': 'drift',
                'confidence': self.gist_confidence,
            }],
            prompt='[cognitive-drift]',
            response=thought,
        )
        logger.info(f"{LOG_PREFIX} Stored drift gist: [{drift_type}] {thought[:80]}...")

    # ── State tracking ───────────────────────────────────────────────

    def _update_state(self, seed: Dict):
        """Update Redis state hash for debugging/monitoring."""
        total = int(self.redis.hget(STATE_KEY, 'total_drifts') or 0)
        self.redis.hset(STATE_KEY, mapping={
            'last_drift_time': time.time(),
            'total_drifts': total + 1,
            'last_seed_type': seed['seed_type'],
            'last_seed_concept': seed['concept_name'],
        })

    # ── Main drift cycle ─────────────────────────────────────────────

    def _run_drift_cycle(self):
        """Execute one complete drift cycle."""
        # 1. Check fatigue
        if self._is_fatigued():
            return

        # 2. Select seed
        seed = self._select_seed()
        if not seed:
            return

        # 3. Spreading activation from seed
        activated = self.semantic_retrieval.spreading_activation(
            [seed['concept_id']], max_depth=self.max_activation_depth
        )

        # 4. Check activation energy threshold
        if not self._has_sufficient_activation(activated):
            logger.info(
                f"{LOG_PREFIX} Insufficient activation energy from "
                f"'{seed['concept_name']}', no thought emerged"
            )
            self._maybe_reinforce_seed(seed, drift_succeeded=False)
            return

        max_activation = max(
            c.get('activation_score', 0) for c in activated
        ) if activated else 0

        # 5. Filter activated concepts through cooldown
        activated = [
            c for c in activated
            if not self._is_on_cooldown(c['id'])
        ][:self.max_activated_concepts]

        # 6. Retrieve grounding episode
        grounding_episode = None
        concept_names = [c['concept_name'] for c in activated[:3]]
        if concept_names:
            query_text = " ".join(concept_names)
            episodes = self.episodic_retrieval.retrieve_episodes(
                query_text=query_text, limit=1
            )
            if episodes:
                grounding_episode = episodes[0]

        # 7. Synthesize thought
        thought = self._synthesize_thought(seed, activated, grounding_episode)
        drift_succeeded = thought is not None

        if drift_succeeded:
            # 8. Store as gist (always — preserves current behavior)
            self._store_drift(seed['topic'], thought['type'], thought['content'])

            # 9. Mark seed + activated concepts as used
            self._mark_used(seed['concept_id'])
            for c in activated:
                self._mark_used(c['id'])

            # 10. Record drift activation for fatigue tracking
            drift_id = str(uuid.uuid4())
            self._record_drift_activation(drift_id, max_activation)

            # 11. Update state
            self._update_state(seed)

            # 12. Action decision: what to do with this thought?
            self._route_thought(thought, seed, max_activation, activated, grounding_episode)
        else:
            logger.info(f"{LOG_PREFIX} Thought synthesis failed for '{seed['concept_name']}'")

        # Maybe reinforce decaying seed
        self._maybe_reinforce_seed(seed, drift_succeeded)

    # ── Action routing ────────────────────────────────────────────

    def _route_thought(self, thought: Dict, seed: Dict, max_activation: float,
                       activated: List[Dict] = None, grounding_episode: Optional[Dict] = None):
        """Route a synthesized thought through the action decision router."""
        from services.autonomous_actions.base import ThoughtContext

        # Generate embedding for the thought (for topic relevance + novelty scoring)
        thought_embedding = None
        try:
            thought_embedding = self.embedding_service.generate_embedding(thought['content'])
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Failed to embed thought: {e}")

        context = ThoughtContext(
            thought_type=thought['type'],
            thought_content=thought['content'],
            activation_energy=max_activation,
            seed_concept=seed['concept_name'],
            seed_topic=seed.get('topic', 'general'),
            thought_embedding=thought_embedding,
            extra={
                'activated_concepts': activated or [],
                'grounding_episode': grounding_episode,
            },
        )

        result = self.action_router.decide_and_execute(context)
        self._log_action_result(result.action_name, context, result, source='drift')

    def _log_action_result(self, action_name: str, context, result, source: str = 'drift'):
        """Log the action result to interaction_log for observability."""
        try:
            db_config = get_merged_db_config()
            db_service = DatabaseService(db_config)
            try:
                from services.interaction_log_service import InteractionLogService
                log_service = InteractionLogService(db_service)

                if action_name == 'COMMUNICATE' and result.success:
                    event_type = 'proactive_sent'
                elif action_name == 'REFLECT' and result.success:
                    event_type = 'reflection_stored'
                else:
                    event_type = 'proactive_candidate'

                log_service.log_event(
                    event_type=event_type,
                    payload={
                        'thought_type': context.thought_type,
                        'thought_content': context.thought_content[:200],
                        'activation_energy': context.activation_energy,
                        'seed_concept': context.seed_concept,
                        'action_selected': action_name,
                        'action_success': result.success,
                        'action_details': result.details,
                        'source': source,
                    },
                    topic=context.seed_topic,
                    source='cognitive_drift_engine',
                )
            finally:
                db_service.close_pool()
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Failed to log action result: {e}")


def cognitive_drift_worker(shared_state=None):
    """Module-level wrapper for multiprocessing."""
    logging.basicConfig(level=logging.INFO)
    try:
        config = ConfigService.get_agent_config("cognitive-drift")
        check_interval = config.get('check_interval', 300)
    except Exception:
        check_interval = 300

    engine = CognitiveDriftEngine(check_interval=check_interval)
    engine.run(shared_state)
