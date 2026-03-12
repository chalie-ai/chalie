# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Cognitive Drift Engine — signal-driven spontaneous reasoning.

Replaces the timer-based Default Mode Network with an event-driven loop.
Reasoning is triggered by signals from other cognitive services:
  - memory_pressure: Decay engine finds fading concepts
  - new_knowledge: Semantic consolidation creates new concepts
  - novel_observation: Experience assimilation stores novel findings
  - ambient_context: Event bridge detects context transitions

When no signals arrive (idle timeout), the engine falls back to
salient and insight seed strategies from existing memory.

After each thought, an ActionDecisionRouter evaluates what to do:
  - COMMUNICATE: Share with the user proactively
  - REFLECT: Internal enrichment
  - PLAN: Create plan-backed persistent tasks
  - NOTHING: Let the gist live in MemoryStore
"""

import dataclasses
import json
import math
import time
import random
import uuid
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any

from services.memory_client import MemoryClientService
from services.config_service import ConfigService
from services.gist_storage_service import GistStorageService
from services.semantic_storage_service import SemanticStorageService
from services.semantic_retrieval_service import SemanticRetrievalService
from services.episodic_retrieval_service import EpisodicRetrievalService
from services.embedding_service import EmbeddingService
from services.background_llm_queue import create_background_llm_proxy
from services.database_service import get_lightweight_db_service

logger = logging.getLogger(__name__)

LOG_PREFIX = "[COGNITIVE DRIFT]"

# MemoryStore keys
COOLDOWN_KEY = "cognitive_drift_concept_cooldowns"
FATIGUE_KEY = "cognitive_drift_activations"
STATE_KEY = "cognitive_drift_state"

# Signal queue constants
SIGNAL_QUEUE_KEY = "reasoning:signals"
MAX_QUEUE_DEPTH = 50


@dataclasses.dataclass
class ReasoningSignal:
    """A signal that triggers reasoning in the cognitive drift engine."""
    signal_type: str       # memory_pressure, new_knowledge, novel_observation, ambient_context
    source: str            # decay_engine, semantic_consolidation, experience_assimilation, event_bridge
    concept_id: int | None = None
    concept_name: str | None = None
    topic: str | None = None
    content: str | None = None
    activation_energy: float = 0.5
    timestamp: float = dataclasses.field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self))

    @classmethod
    def from_json(cls, raw: str) -> 'ReasoningSignal':
        data = json.loads(raw if isinstance(raw, str) else raw.decode())
        return cls(**data)


def emit_reasoning_signal(signal: ReasoningSignal) -> bool:
    """Emit a reasoning signal to the signal queue. Called by other services."""
    try:
        store = MemoryClientService.create_connection()
        config = ConfigService.resolve_agent_config("cognitive-drift")
        signal_config = config.get('signal_queue', {})
        queue_key = signal_config.get('key', 'reasoning:signals')
        max_depth = signal_config.get('max_queue_depth', 50)

        # Cap queue depth
        if store.llen(queue_key) >= max_depth:
            logger.debug(f"[REASONING SIGNAL] Queue full ({max_depth}), dropping signal: {signal.signal_type}")
            return False

        store.rpush(queue_key, json.dumps(dataclasses.asdict(signal)))
        logger.debug(f"[REASONING SIGNAL] Emitted {signal.signal_type} from {signal.source}")
        return True
    except Exception as e:
        logger.debug(f"[REASONING SIGNAL] Failed to emit: {e}")
        return False


class CognitiveDriftEngine:
    """
    Background service that generates spontaneous thoughts driven by
    incoming reasoning signals or, when idle, by autonomous discovery.

    Signals are pushed by other services (decay engine, semantic consolidation,
    experience assimilation, event bridge) via emit_reasoning_signal(). The
    engine blocks on blpop() and processes each signal through the full
    spreading-activation → synthesis → action pipeline.

    When no signal arrives within idle_timeout seconds, the engine enters
    discovery mode: it picks a seed via salient or insight strategy and
    reasons from it directly.
    """

    def __init__(self, check_interval: int = 300):
        self.store = MemoryClientService.create_connection()
        self.config = ConfigService.resolve_agent_config("cognitive-drift")
        self.check_interval = check_interval

        # Load queue names for idle check
        conn_config = ConfigService.connections()
        topics = conn_config.get("memory", {}).get("topics", {})
        self.prompt_queue = topics.get("prompt_queue", "prompt-queue")
        self.memory_queue = topics.get("memory_chunker", "memory-chunker-queue")
        self.episodic_queue = topics.get("episodic_memory", "episodic-memory-queue")
        self.semantic_queue = conn_config.get("memory", {}).get("queues", {}).get(
            "semantic_consolidation_queue", {}
        ).get("name", "semantic_consolidation_queue")

        # Database + services
        self.db_service = get_lightweight_db_service()
        self.embedding_service = EmbeddingService()
        self.semantic_storage = SemanticStorageService(self.db_service)
        self.semantic_retrieval = SemanticRetrievalService(
            self.db_service, self.embedding_service, self.semantic_storage
        )

        episodic_config = ConfigService.resolve_agent_config("episodic-memory")
        self.episodic_retrieval = EpisodicRetrievalService(self.db_service, episodic_config)

        # Gist storage
        self.gist_storage = GistStorageService()

        # LLM for thought synthesis — refreshable so provider changes take effect without restart
        self.ollama = create_background_llm_proxy("cognitive-drift")

        # Load prompt template + soul axioms
        self.prompt_template = ConfigService.get_agent_prompt("cognitive-drift")
        self.soul_axioms = ConfigService.get_agent_prompt("soul")

        # Config values
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
        self.min_activation_energy = self.config.get('min_activation_energy', 0.4)
        self.decaying_reinforce_bump = self.config.get('decaying_reinforce_bump', 0.1)

        # Signal loop config
        self.idle_timeout = self.config.get('signal_queue', {}).get('idle_timeout_seconds', 600)
        self.min_signal_interval = self.config.get('signal_queue', {}).get('min_signal_interval_seconds', 30)

        # Initialize autonomous action router
        self.action_router = self._init_action_router()

        logger.info(
            f"{LOG_PREFIX} Engine initialized "
            f"(check_interval={check_interval}s, signal-driven)"
        )

    def _init_action_router(self):
        """Initialize the autonomous action decision router."""
        from services.autonomous_actions.decision_router import ActionDecisionRouter
        from services.autonomous_actions.nothing_action import NothingAction
        from services.autonomous_actions.communicate_action import CommunicateAction
        from services.autonomous_actions.reflect_action import ReflectAction
        from services.autonomous_actions.seed_thread_action import SeedThreadAction
        from services.autonomous_actions.suggest_action import SuggestAction

        router = ActionDecisionRouter()

        # Always register NOTHING (fallback)
        router.register(NothingAction())

        action_config = self.config.get('autonomous_actions', {})

        # Register SEED_THREAD (priority 6, above REFLECT=5)
        seed_config = action_config.get('seed_thread', {})
        if seed_config.get('enabled', True):
            router.register(SeedThreadAction(config=seed_config))

        # Register SUGGEST (priority 8, below COMMUNICATE=10, above SEED_THREAD=6)
        suggest_config = action_config.get('suggest', {})
        if suggest_config.get('enabled', True):
            router.register(SuggestAction(config=suggest_config))

        # Register NURTURE (priority 7, between SUGGEST and SEED_THREAD)
        from services.autonomous_actions.nurture_action import NurtureAction
        nurture_config = action_config.get('nurture', {})
        if nurture_config.get('enabled', True):
            router.register(NurtureAction(config=nurture_config))

        # Register PLAN (priority 7, same as NURTURE — ties broken by score)
        from services.autonomous_actions.plan_action import PlanAction
        plan_config = action_config.get('plan', {})
        if plan_config.get('enabled', True):
            router.register(PlanAction(config=plan_config))

        # Register AMBIENT_TOOL if enabled (priority 6, below SEED_THREAD)
        from services.autonomous_actions.ambient_tool_action import AmbientToolAction
        ambient_config = action_config.get('ambient_tool', {})
        if ambient_config.get('enabled', True):
            router.register(AmbientToolAction(config=ambient_config))

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

        # Register RECONCILE if enabled (priority 4, below REFLECT=5)
        from services.autonomous_actions.reconcile_action import ReconcileAction
        reconcile_config = action_config.get('reconcile', {})
        if reconcile_config.get('enabled', True):
            router.register(ReconcileAction(
                config=reconcile_config,
                services={'db_service': self.db_service},
            ))

        return router

    # -- Signal loop -----------------------------------------------------------

    def run_signal_loop(self, shared_state: Optional[dict] = None) -> None:
        """Main signal-driven reasoning loop. Replaces the old timer-based run()."""
        logger.info(f"{LOG_PREFIX} Signal loop started (idle_timeout={self.idle_timeout}s)")

        while True:
            try:
                # Check for deferred thoughts from quiet hours
                self._process_deferred()

                # Block until a signal arrives or idle_timeout elapses
                result = self.store.blpop(SIGNAL_QUEUE_KEY, timeout=self.idle_timeout)

                if result is None:
                    # Timeout — no signal arrived; enter discovery mode
                    logger.debug(f"{LOG_PREFIX} Idle timeout, entering discovery mode")
                    self._handle_idle_signal()
                else:
                    # result is (key, value) tuple
                    _, raw = result
                    try:
                        signal = ReasoningSignal.from_json(raw)
                        logger.info(
                            f"{LOG_PREFIX} Processing {signal.signal_type} signal "
                            f"from {signal.source}"
                        )
                        self._process_signal(signal)
                    except Exception as e:
                        logger.warning(f"{LOG_PREFIX} Failed to parse signal: {e}")

            except KeyboardInterrupt:
                logger.info(f"{LOG_PREFIX} Signal loop shutting down...")
                break
            except Exception as e:
                logger.error(f"{LOG_PREFIX} Error in signal loop: {e}", exc_info=True)
                logger.info(f"{LOG_PREFIX} Waiting 1 minute before retry...")
                time.sleep(60)

    def _process_signal(self, signal: ReasoningSignal) -> None:
        """Process one incoming reasoning signal through the full pipeline."""
        # Gate 1: Fatigue
        if self._is_fatigued():
            return

        # Gate 2: Deep focus
        if self._is_user_deep_focus():
            logger.info(f"{LOG_PREFIX} User in deep focus, skipping signal")
            return

        # Gate 3: Recent episodes required
        if not self._has_recent_episodes():
            logger.debug(f"{LOG_PREFIX} No recent episodes, skipping signal")
            return

        # Gate 4: Memory richness
        try:
            from services.self_model_service import SelfModelService
            richness = SelfModelService().get_memory_richness()
            if richness < 0.1:
                logger.debug(f"{LOG_PREFIX} Richness {richness:.2f} < 0.1, skipping signal")
                return
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Richness check failed, skipping signal: {e}")
            return

        # Gate 5: Workers idle
        if not self._are_workers_idle():
            logger.debug(f"{LOG_PREFIX} Workers not idle, skipping signal")
            return

        # Debounce: prevent signal storms
        if not self._debounce_check(signal):
            return

        # Occasionally generate a self-reflective thought instead
        if self._try_self_model_reflection():
            return

        # Convert signal to concept seed
        seed = self._signal_to_seed(signal)
        if not seed:
            logger.debug(
                f"{LOG_PREFIX} No viable seed for {signal.signal_type} signal "
                f"(concept_id={signal.concept_id})"
            )
            return

        self._reason_from_seed(seed, signal)

    def _signal_to_seed(self, signal: ReasoningSignal) -> Optional[Dict]:
        """Convert a reasoning signal to a concept seed dict."""
        # Direct concept reference — fastest path
        if signal.concept_id is not None:
            concept_id_str = str(signal.concept_id)
            if not self._is_on_cooldown(concept_id_str):
                return {
                    'concept_id': concept_id_str,
                    'concept_name': signal.concept_name or f'concept:{signal.concept_id}',
                    'definition': signal.content or '',
                    'seed_type': signal.signal_type,
                    'topic': signal.topic or 'general',
                }

        # Content-based lookup via semantic retrieval
        if signal.content:
            try:
                concepts = self.semantic_retrieval.retrieve_concepts(signal.content, limit=3)
                for concept in concepts:
                    cid = str(concept.get('id', ''))
                    if cid and not self._is_on_cooldown(cid):
                        return {
                            'concept_id': cid,
                            'concept_name': concept.get('concept_name', ''),
                            'definition': concept.get('definition', ''),
                            'seed_type': signal.signal_type,
                            'topic': signal.topic or concept.get('domain', 'general'),
                        }
            except Exception as e:
                logger.debug(f"{LOG_PREFIX} Semantic lookup for signal failed: {e}")

        return None

    def _handle_idle_signal(self) -> None:
        """Timeout handler — discovery mode when no signals arrive."""
        # Gate checks (same as _process_signal)
        if self._is_fatigued():
            return
        if self._is_user_deep_focus():
            logger.info(f"{LOG_PREFIX} User in deep focus, skipping idle drift")
            return
        if not self._has_recent_episodes():
            logger.debug(f"{LOG_PREFIX} No recent episodes, skipping idle drift")
            return
        try:
            from services.self_model_service import SelfModelService
            richness = SelfModelService().get_memory_richness()
            if richness < 0.1:
                logger.debug(f"{LOG_PREFIX} Richness {richness:.2f} < 0.1, skipping idle drift")
                return
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Richness check failed, skipping idle drift: {e}")
            return
        if not self._are_workers_idle():
            logger.debug(f"{LOG_PREFIX} Workers not idle, skipping idle drift")
            return

        # Pick strategy: 60% salient, 40% insight
        seed = None
        if random.random() < 0.6:
            seed = self._seed_salient()
        else:
            seed = self._seed_insight()

        if not seed:
            logger.debug(f"{LOG_PREFIX} Idle discovery: no seed found")
            return

        if self._is_on_cooldown(seed['concept_id']):
            logger.debug(f"{LOG_PREFIX} Idle discovery: seed '{seed['concept_name']}' on cooldown")
            return

        idle_signal = ReasoningSignal(
            signal_type='idle_discovery',
            source='cognitive_drift_engine',
            concept_id=int(seed['concept_id']) if seed['concept_id'].isdigit() else None,
            concept_name=seed.get('concept_name'),
            topic=seed.get('topic'),
            content=seed.get('definition'),
            activation_energy=0.5,
        )
        self._reason_from_seed(seed, idle_signal)

    def _reason_from_seed(self, seed: Dict, signal: ReasoningSignal) -> None:
        """Core reasoning pipeline: spreading activation → synthesis → action."""
        # Step 3: Spreading activation from seed
        activated = self.semantic_retrieval.spreading_activation(
            [seed['concept_id']], max_depth=self.max_activation_depth
        )

        # Step 4: Check activation energy threshold
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

        # Step 5: Filter activated concepts through cooldown
        activated = [
            c for c in activated
            if not self._is_on_cooldown(c['id'])
        ][:self.max_activated_concepts]

        # Step 6: Retrieve grounding episode
        grounding_episode = None
        concept_names = [c['concept_name'] for c in activated[:3]]
        if concept_names:
            query_text = " ".join(concept_names)
            episodes = self.episodic_retrieval.retrieve_episodes(
                query_text=query_text, limit=1
            )
            if episodes:
                grounding_episode = episodes[0]

        # Step 7: Synthesize thought
        thought = self._synthesize_thought(seed, activated, grounding_episode)
        drift_succeeded = thought is not None

        if drift_succeeded:
            # Step 8: Store as gist (always — preserves current behavior)
            self._store_drift(seed['topic'], thought['type'], thought['content'])

            # Step 9: Mark seed + activated concepts as used
            self._mark_used(seed['concept_id'])
            for c in activated:
                self._mark_used(c['id'])

            # Step 10: Record drift activation for fatigue tracking
            drift_id = str(uuid.uuid4())
            self._record_drift_activation(drift_id, max_activation)

            # Step 11: Update state
            self._update_state(seed)

            # Step 12: Route through action router
            self._route_thought(thought, seed, max_activation, activated, grounding_episode)
        else:
            logger.info(f"{LOG_PREFIX} Thought synthesis failed for '{seed['concept_name']}'")

        # Maybe reinforce decaying seed
        self._maybe_reinforce_seed(seed, drift_succeeded)

    def _debounce_check(self, signal: ReasoningSignal) -> bool:
        """Prevent signal storms by enforcing a minimum interval between processed signals."""
        last = self.store.get("reasoning:last_processed")
        if last:
            elapsed = time.time() - float(last)
            if elapsed < self.min_signal_interval:
                logger.debug(
                    f"{LOG_PREFIX} Signal debounced ({elapsed:.1f}s < {self.min_signal_interval}s interval)"
                )
                return False
        self.store.set("reasoning:last_processed", str(time.time()), ex=self.min_signal_interval * 2)
        return True

    # -- Deferred processing ---------------------------------------------------

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

    # -- Precondition checks --------------------------------------------------

    def _are_workers_idle(self) -> bool:
        """Check if all worker queues are empty."""
        queues = [
            self.prompt_queue,
            self.memory_queue,
            self.episodic_queue,
            self.semantic_queue,
        ]
        for queue_name in queues:
            if self.store.llen(queue_name) > 0:
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
                      AND created_at > datetime('now', ? || ' hours')
                    """,
                    (str(-self.episode_lookback_hours),)
                )
                count = cursor.fetchone()[0]
                cursor.close()
                return count > 0
        except Exception as e:
            logger.error(f"{LOG_PREFIX} Failed to check recent episodes: {e}")
            return False

    # -- Fatigue ---------------------------------------------------------------

    def _is_fatigued(self) -> bool:
        """Check if cumulative drift activation exceeds budget."""
        cutoff = time.time() - (self.fatigue_window_minutes * 60)
        recent = self.store.zrangebyscore(FATIGUE_KEY, cutoff, '+inf', withscores=True)

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
        self.store.zadd(FATIGUE_KEY, {f"{drift_id}:{max_activation}": time.time()})
        cutoff = time.time() - (self.fatigue_window_minutes * 60)
        self.store.zremrangebyscore(FATIGUE_KEY, '-inf', cutoff)

    def _is_user_deep_focus(self) -> bool:
        """Check if user is in deep focus — skip drift to avoid interrupting."""
        try:
            from services.ambient_inference_service import AmbientInferenceService
            inference = AmbientInferenceService()
            return inference.is_user_deep_focus()
        except Exception:
            return False

    # -- Cooldown --------------------------------------------------------------

    def _is_on_cooldown(self, concept_id: str) -> bool:
        """Check if a concept was recently used in drift."""
        last_used = self.store.zscore(COOLDOWN_KEY, concept_id)
        if last_used and (time.time() - last_used) < self.cooldown_minutes * 60:
            return True
        return False

    def _mark_used(self, concept_id: str):
        """Record concept as recently used."""
        self.store.zadd(COOLDOWN_KEY, {concept_id: time.time()})
        cutoff = time.time() - (self.cooldown_minutes * 60)
        self.store.zremrangebyscore(COOLDOWN_KEY, '-inf', cutoff)

    # -- Activation energy -----------------------------------------------------

    def _has_sufficient_activation(self, activated_concepts: list) -> bool:
        """Check if spreading activation produced strong enough results."""
        if not activated_concepts:
            return False
        max_activation = max(c.get('activation_score', 0) for c in activated_concepts)
        return max_activation >= self.min_activation_energy

    # -- Seed selection (kept for idle discovery) ------------------------------

    def _seed_salient(self) -> Optional[Dict[str, Any]]:
        """Select a concept from a high-salience episode, weighted by salience x recency."""
        with self.db_service.connection() as conn:
            cursor = conn.cursor()

            # Fetch top-20 salient episodes within lookback window
            cursor.execute("""
                SELECT id, gist, topic,
                       (CAST(strftime('%s', 'now') AS REAL) - CAST(strftime('%s', created_at) AS REAL)) / 3600.0 AS age_hours,
                       salience
                FROM episodes
                WHERE deleted_at IS NULL
                  AND created_at > datetime('now', ? || ' days')
                ORDER BY salience DESC
                LIMIT 20
            """, (str(-self.salience_lookback_days),))
            episodes = cursor.fetchall()
            cursor.close()

        if not episodes:
            return None

        # Combined weight: salience x exp(-decay * age_hours)
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

    def _seed_insight(self) -> Optional[Dict[str, Any]]:
        """
        Select seed from recurring interaction patterns: topics that appear
        3+ times in the interaction log, suggesting friction or unresolved interest.
        Caps to 1 insight seed per configured session count to avoid over-surfacing.
        """
        insight_cap = self.config.get('insight_sessions_cap', 5)
        last_insight_key = f"drift:insight_seed:last_used"
        last_used_epoch = self.store.get(last_insight_key)
        if last_used_epoch:
            # Check if enough drift cycles have passed (approximate: cap * check_interval)
            elapsed = time.time() - float(last_used_epoch)
            min_gap = insight_cap * self.check_interval
            if elapsed < min_gap:
                logger.debug(f"{LOG_PREFIX} Insight seed throttled (last used {elapsed:.0f}s ago)")
                return None

        try:
            with self.db_service.connection() as conn:
                cursor = conn.cursor()

                # Find topics recurring 3+ times in the last 7 days
                cursor.execute("""
                    SELECT topic, COUNT(*) AS cnt
                    FROM interaction_log
                    WHERE created_at > datetime('now', '-7 days')
                      AND topic IS NOT NULL
                      AND topic != 'general'
                    GROUP BY topic
                    HAVING COUNT(*) >= 3
                    ORDER BY cnt DESC
                    LIMIT 5
                """)
                recurring = cursor.fetchall()

                if not recurring:
                    cursor.close()
                    return None

                # Weight by occurrence count, pick topic
                topics = [row[0] for row in recurring]
                weights = [float(row[1]) for row in recurring]
                chosen_topic = random.choices(topics, weights=weights, k=1)[0]

                # Find a concept associated with the recurring topic
                cursor.execute("""
                    SELECT id, concept_name, definition, domain
                    FROM semantic_concepts
                    WHERE deleted_at IS NULL
                      AND confidence >= 0.4
                      AND (domain = ? OR concept_name LIKE ?)
                    ORDER BY strength DESC
                    LIMIT 5
                """, (chosen_topic, f'%{chosen_topic}%'))
                rows = cursor.fetchall()
                cursor.close()

            if not rows:
                return None

            self.store.set(last_insight_key, str(time.time()), ex=insight_cap * self.check_interval * 2)

            row = random.choice(rows)
            return {
                'concept_id': str(row[0]),
                'concept_name': row[1],
                'definition': row[2],
                'seed_type': 'insight',
                'topic': chosen_topic,
            }

        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Insight seed selection failed: {e}")
            return None

    # -- Decaying concept reinforcement ----------------------------------------

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
                    SET strength = MIN(10.0, strength + ?),
                        updated_at = datetime('now')
                    WHERE id = ? AND deleted_at IS NULL
                """, (self.decaying_reinforce_bump, seed['concept_id']))
                cursor.close()

            logger.debug(
                f"{LOG_PREFIX} Nudged decaying concept '{seed['concept_name']}' "
                f"by +{self.decaying_reinforce_bump}"
            )
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Failed to reinforce seed: {e}")

    # -- Thought synthesis -----------------------------------------------------

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

        # Temporal rhythm context (may be empty if no patterns yet)
        rhythm_text = ""
        try:
            from services.temporal_pattern_service import TemporalPatternService
            from services.database_service import get_shared_db_service
            db = get_shared_db_service()
            rhythm_text = TemporalPatternService(db).get_rhythm_summary()
        except Exception:
            pass

        # Inject constraint context so drift thoughts can factor in blocked paths
        constraint_context = ''
        try:
            from services.constraint_memory_service import ConstraintMemoryService
            constraint_context = ConstraintMemoryService().format_for_prompt(mode='drift')
        except Exception:
            pass

        user_message = self.prompt_template \
            .replace("{{seed_concept}}", seed_text) \
            .replace("{{activated_concepts}}", activated_text) \
            .replace("{{grounding_episode}}", episode_text) \
            .replace("{{temporal_rhythm}}", rhythm_text) \
            .replace("{{constraint_context}}", constraint_context)

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
                # Also strip unclosed <think> (truncated model output — no closing tag)
                if '<think>' in cleaned:
                    cleaned = re.sub(r'<think>.*', '', cleaned, flags=re.DOTALL).strip()

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

                if thought_type not in ('reflection', 'question', 'hypothesis', 'insight'):
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

    # -- Gist storage ----------------------------------------------------------

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

    # -- State tracking --------------------------------------------------------

    def _update_state(self, seed: Dict):
        """Update MemoryStore state hash for debugging/monitoring."""
        total = int(self.store.hget(STATE_KEY, 'total_drifts') or 0)
        self.store.hset(STATE_KEY, mapping={
            'last_drift_time': time.time(),
            'total_drifts': total + 1,
            'last_seed_type': seed['seed_type'],
            'last_seed_concept': seed['concept_name'],
        })

    # -- Self-model reflection -------------------------------------------------

    def _try_self_model_reflection(self) -> bool:
        """
        Occasionally generate a self-reflective thought from the self-model.

        Fires ~20% of cycles when noteworthy items exist. Returns True if
        a reflection was produced (consuming this drift cycle).
        """
        if random.random() > 0.20:
            return False

        try:
            from services.self_model_service import SelfModelService
            snapshot = SelfModelService().get_snapshot()
            noteworthy = snapshot.get('noteworthy', [])
            if not noteworthy:
                return False

            # Pick the highest-severity noteworthy item
            item = max(noteworthy, key=lambda n: n.get('severity', 0))
            signal = item.get('signal', '')

            # Map signal patterns to reflective seed thoughts
            if 'recall' in signal or 'activation' in signal:
                reflection = (
                    f"I notice my memory recall is struggling. "
                    f"I should consolidate what I know into stronger semantic concepts."
                )
            elif 'capability_gap' in signal:
                reflection = (
                    f"Users keep asking me to do something I can't: {signal.split(':', 1)[-1].strip()}. "
                    f"I should think about how to help them differently."
                )
            elif 'queue' in signal or 'congestion' in signal:
                reflection = (
                    "My background processing is congested. "
                    "I should be more careful about what I commit to right now."
                )
            elif 'dead_thread' in signal or 'provider' in signal:
                reflection = (
                    f"Something in my infrastructure feels off: {signal}. "
                    f"I should flag this if it affects my responses."
                )
            else:
                reflection = f"Internal observation: {signal}"

            # Store as a drift gist (same as normal thoughts)
            self._store_drift('self-reflection', 'reflection', reflection)

            logger.info(f"{LOG_PREFIX} Self-model reflection: '{reflection[:80]}...'")
            return True

        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Self-model reflection failed: {e}")
            return False

    # -- Action routing --------------------------------------------------------

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
                'seed_type': seed.get('seed_type', 'unknown'),
            },
        )

        result = self.action_router.decide_and_execute(context)
        self._log_action_result(result.action_name, context, result, source='drift')

    def _log_action_result(self, action_name: str, context, result, source: str = 'drift'):
        """Log the action result to interaction_log for observability.

        Logs two event types:
          1. The action outcome (proactive_sent, reflection_stored, proactive_candidate)
          2. Gate rejections from ineligible actions (action_gate_rejected) — feeds
             constraint decisions into the memory pipeline so Chalie can learn from
             what it considered but couldn't do.
        """
        try:
            db_service = get_lightweight_db_service()
            try:
                from services.interaction_log_service import InteractionLogService
                log_service = InteractionLogService(db_service)

                if action_name == 'COMMUNICATE' and result.success:
                    event_type = 'proactive_sent'
                elif action_name == 'REFLECT' and result.success:
                    event_type = 'reflection_stored'
                else:
                    event_type = 'proactive_candidate'

                # Extract gate rejections before logging (they're metadata, not action details)
                gate_rejections = result.details.pop('gate_rejections', [])

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

                # Log gate rejections as separate events — these are the
                # constraint decisions that previously vanished into debug logs.
                # Filtering out NOTHING (always eligible, no signal value) and
                # generic phase/cooldown gates that fire constantly.
                if gate_rejections:
                    # Only log rejections with meaningful signal (not just disabled/nothing)
                    meaningful = [
                        r for r in gate_rejections
                        if r.get('action') != 'NOTHING'
                    ]
                    if meaningful:
                        log_service.log_event(
                            event_type='action_gate_rejected',
                            payload={
                                'thought_type': context.thought_type,
                                'thought_content': context.thought_content[:200],
                                'activation_energy': context.activation_energy,
                                'seed_concept': context.seed_concept,
                                'action_selected': action_name,
                                'rejections': meaningful,
                                'source': source,
                            },
                            topic=context.seed_topic,
                            source='cognitive_drift_engine',
                        )

                        # Feed gate rejections into procedural memory as soft failures
                        try:
                            from services.procedural_memory_service import ProceduralMemoryService
                            proc = ProceduralMemoryService(db_service)
                            for r in meaningful:
                                proc.record_gate_rejection(
                                    action_name=r.get('action', 'unknown'),
                                    reason=r.get('reason', ''),
                                )
                        except Exception:
                            pass
            finally:
                db_service.close_pool()
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Failed to log action result: {e}")


def cognitive_drift_worker(shared_state=None):
    """Module-level wrapper for threading."""
    logging.basicConfig(level=logging.INFO)
    try:
        config = ConfigService.get_agent_config("cognitive-drift")
        check_interval = config.get('check_interval', 300)
    except Exception:
        check_interval = 300

    engine = CognitiveDriftEngine(check_interval=check_interval)
    engine.run_signal_loop(shared_state)
