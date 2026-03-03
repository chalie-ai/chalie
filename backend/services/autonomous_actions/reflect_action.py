"""
ReflectAction — Internal enrichment of drift thoughts via association linking.

When a drift thought connects meaningfully to existing knowledge (episodes,
concepts), REFLECT stores an enriched gist with association context and boosts
the related concepts' access counts. Unlike COMMUNICATE, this is entirely
internal — no user-facing output.

Two gates:
  1. Relevance: activation energy, type bonus with repeat decay, episode/concept
     similarity, novelty against recent reflections
  2. Fatigue: shared drift fatigue budget (40% allocation)

Priority 5 — beats NOTHING (-1), loses to COMMUNICATE (10).
"""

import json
import time
import hashlib
import math
import logging
from typing import Optional, Dict, Any, Tuple, List

from services.memory_client import MemoryClientService

from .base import AutonomousAction, ActionResult, ThoughtContext

logger = logging.getLogger(__name__)

LOG_PREFIX = "[REFLECT]"

# MemoryStore key namespace — per-topic
_NS = "reflection"

# Shared fatigue key (same as cognitive_drift_engine.py)
FATIGUE_KEY = "cognitive_drift_activations"


def _key(topic: str, suffix: str) -> str:
    return f"{_NS}:{topic}:{suffix}"


class ReflectAction(AutonomousAction):
    """
    Enriches drift thoughts with association links and stores as reflection gists.

    Relevance + Fatigue gates must pass for eligibility.
    Score = activation_energy * type_bonus * type_decay * max(episode_rel, concept_rel)
    """

    def __init__(self, config: dict = None, services: dict = None):
        super().__init__(name='REFLECT', enabled=True, priority=5)

        config = config or {}
        services = services or {}

        self.store = MemoryClientService.create_connection()

        # Injected services (from drift engine — no new connections)
        self._gist_storage = services.get('gist_storage')
        self._embedding_service = services.get('embedding_service')
        self._db_service = services.get('db_service')

        # Relevance gate config
        self.min_activation_energy = config.get('min_activation_energy', 0.35)
        self.type_bonuses = config.get('type_bonuses', {
            'reflection': 1.3,
            'hypothesis': 1.2,
            'question': 0.9,
        })
        self.type_repeat_decay = config.get('type_repeat_decay', 0.8)
        self.min_episode_similarity = config.get('min_episode_similarity', 0.3)
        self.min_concept_similarity = config.get('min_concept_similarity', 0.35)
        self.novelty_threshold = config.get('novelty_threshold', 0.75)
        self.max_recent_reflections = config.get('max_recent_reflections', 10)

        # Fatigue gate config
        self.fatigue_budget_fraction = config.get('fatigue_budget_fraction', 0.4)
        # Total budget read from drift engine config (default 2.5)
        self._total_fatigue_budget = config.get('total_fatigue_budget', 2.5)
        self.fatigue_window_minutes = config.get('fatigue_window_minutes', 30)

        # Execute config
        self.reflection_gist_confidence = config.get('reflection_gist_confidence', 8)
        self.concept_boost_limit = config.get('concept_boost_limit', 3)

        # Enabled flag
        if not config.get('enabled', True):
            self.enabled = False

    # -- Gate 1: Relevance -----------------------------------------------------

    def _relevance_score(self, thought: ThoughtContext) -> Tuple[float, bool, Dict]:
        """
        Evaluate thought relevance for reflection.

        Returns:
            (score, passes, details)
        """
        details = {}

        # 1. Activation energy floor
        if thought.activation_energy < self.min_activation_energy:
            details['rejected'] = 'activation_energy_below_threshold'
            details['activation_energy'] = thought.activation_energy
            details['threshold'] = self.min_activation_energy
            return (0.0, False, details)

        # 2. Type bonus with repeat decay
        type_bonus = self.type_bonuses.get(thought.thought_type, 1.0)
        details['type_bonus'] = type_bonus

        type_decay = 1.0
        last_type = self.store.get(_key(thought.seed_topic, 'last_type'))
        if last_type and last_type == thought.thought_type:
            type_decay = self.type_repeat_decay
            details['type_repeat_decay_applied'] = True
        details['type_decay'] = type_decay

        # 3. Episode relevance (from extra context)
        episode_relevance = 0.0
        grounding_episode = thought.extra.get('grounding_episode')
        if grounding_episode and thought.thought_embedding:
            episode_embedding = grounding_episode.get('embedding')
            if episode_embedding:
                episode_relevance = self._cosine_similarity(
                    thought.thought_embedding, episode_embedding
                )
        details['episode_relevance'] = episode_relevance

        # 4. Concept relevance (average activation from extra context)
        concept_relevance = 0.0
        activated_concepts = thought.extra.get('activated_concepts', [])
        if activated_concepts:
            scores = [c.get('activation_score', 0) for c in activated_concepts]
            concept_relevance = sum(scores) / len(scores) if scores else 0.0
        details['concept_relevance'] = concept_relevance

        # At least one relevance signal must pass threshold
        max_relevance = max(episode_relevance, concept_relevance)
        details['max_relevance'] = max_relevance

        if episode_relevance < self.min_episode_similarity and concept_relevance < self.min_concept_similarity:
            details['rejected'] = 'relevance_below_threshold'
            return (0.0, False, details)

        # 5. Novelty check against recent reflections
        is_novel, novelty_details = self._check_novelty(thought)
        details['novelty'] = novelty_details

        if not is_novel:
            details['rejected'] = 'not_novel'
            return (0.0, False, details)

        # Composite score
        score = thought.activation_energy * type_bonus * type_decay * max_relevance
        details['composite_score'] = score

        return (score, True, details)

    def _check_novelty(self, thought: ThoughtContext) -> Tuple[bool, Dict]:
        """Check thought isn't too similar to recent reflections via cosine similarity."""
        details = {}

        if thought.thought_embedding is None:
            return (True, {'reason': 'no_embedding_available'})

        recent_key = _key(thought.seed_topic, 'recent_embeddings')
        stored = self.store.lrange(recent_key, 0, -1)

        if not stored:
            return (True, {'reason': 'no_recent_reflections'})

        max_sim = 0.0
        for raw in stored:
            try:
                entry = json.loads(raw)
                emb = entry.get('embedding', [])
                sim = self._cosine_similarity(thought.thought_embedding, emb)
                if sim > max_sim:
                    max_sim = sim
            except (json.JSONDecodeError, TypeError):
                continue

        details['max_similarity'] = max_sim
        details['threshold'] = self.novelty_threshold

        if max_sim >= self.novelty_threshold:
            return (False, details)
        return (True, details)

    @staticmethod
    def _cosine_similarity(a: list, b: list) -> float:
        """Compute cosine similarity between two vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    # -- Gate 2: Fatigue -------------------------------------------------------

    def _fatigue_passes(self, thought: ThoughtContext) -> Tuple[bool, Dict]:
        """Check if REFLECT's share of fatigue budget is available."""
        details = {}
        budget = self._total_fatigue_budget * self.fatigue_budget_fraction
        details['reflect_budget'] = budget

        cutoff = time.time() - (self.fatigue_window_minutes * 60)
        recent = self.store.zrangebyscore(FATIGUE_KEY, cutoff, '+inf', withscores=True)

        total_activation = 0.0
        for member, _ in recent:
            try:
                total_activation += float(member.split(':')[1])
            except (IndexError, ValueError):
                continue

        details['total_activation'] = total_activation

        if total_activation >= budget:
            details['rejected'] = 'fatigue_budget_exceeded'
            return (False, details)

        return (True, details)

    # -- Main interface --------------------------------------------------------

    def should_execute(self, thought: ThoughtContext) -> tuple:
        """
        Evaluate relevance + fatigue gates.

        Returns:
            (score, eligible)
        """
        # Gate 1: Relevance
        score, passes, relevance_details = self._relevance_score(thought)
        if not passes:
            # Track rejections for adaptive threshold monitoring
            self._increment_rejected(thought.seed_topic)
            logger.debug(
                f"{LOG_PREFIX} Rejected: {relevance_details.get('rejected', 'unknown')} "
                f"(topic={thought.seed_topic})"
            )
            return (0.0, False)

        # Gate 2: Fatigue
        fatigue_passes, fatigue_details = self._fatigue_passes(thought)
        if not fatigue_passes:
            self._increment_rejected(thought.seed_topic)
            logger.debug(f"{LOG_PREFIX} Rejected: fatigue budget exceeded")
            return (0.0, False)

        logger.info(
            f"{LOG_PREFIX} Eligible: score={score:.3f} "
            f"(type={thought.thought_type}, topic={thought.seed_topic})"
        )
        return (score, True)

    def execute(self, thought: ThoughtContext) -> ActionResult:
        """
        Store enriched reflection gist and boost associated concepts.
        """
        activated_concepts = thought.extra.get('activated_concepts', [])

        # 1. Build enriched content with association links
        concept_names = [c['concept_name'] for c in activated_concepts[:self.concept_boost_limit]]
        connects_to = ", ".join(concept_names) if concept_names else "no direct associations"

        enriched_content = (
            f"[reflection on '{thought.seed_concept}'] "
            f"{thought.thought_content} "
            f"(connects to: {connects_to})"
        )

        # 2. Store as reflection gist
        if self._gist_storage:
            self._gist_storage.store_gists(
                topic=thought.seed_topic,
                gists=[{
                    'content': enriched_content,
                    'type': 'reflection',
                    'confidence': self.reflection_gist_confidence,
                }],
                prompt='[cognitive-drift-reflect]',
                response=thought.thought_content,
            )

        # 3. Record embedding for novelty tracking
        if thought.thought_embedding:
            recent_key = _key(thought.seed_topic, 'recent_embeddings')
            entry = json.dumps({
                'embedding': thought.thought_embedding,
                'ts': time.time(),
                'ts_created': time.time(),
                'seed_concept': thought.seed_concept,
                'thought_type': thought.thought_type,
            })
            self.store.rpush(recent_key, entry)
            self.store.ltrim(recent_key, -self.max_recent_reflections, -1)
            self.store.expire(recent_key, 5400)  # 90min TTL

        # 4. Update last_type for repeat decay
        self.store.setex(
            _key(thought.seed_topic, 'last_type'),
            600,  # 10min TTL
            thought.thought_type,
        )

        # 5. Boost associated concepts (access_count + last_accessed_at)
        boosted_ids = self._boost_concepts(activated_concepts[:self.concept_boost_limit])

        # 6. Strategy analysis (opportunistic)
        strategy_insight = self._analyze_act_strategies(thought.seed_topic)
        if strategy_insight and self._gist_storage:
            insight_text = (
                f"[strategy insight] Tool combo [{strategy_insight['best_strategy']}] "
                f"(avg value {strategy_insight['best_avg_value']:.2f}, "
                f"~{strategy_insight['best_avg_seconds']:.0f}s, {strategy_insight['best_complexity']}) "
                f"outperformed [{strategy_insight['worst_strategy']}] "
                f"(avg value {strategy_insight['worst_avg_value']:.2f}, "
                f"~{strategy_insight['worst_avg_seconds']:.0f}s) "
                f"over {strategy_insight['loops_analyzed']} recent loops"
            )
            self._gist_storage.store_gists(
                topic=thought.seed_topic,
                gists=[{'content': insight_text, 'type': 'strategy', 'confidence': self.reflection_gist_confidence}],
                prompt='[strategy-reflect]',
                response=insight_text,
            )

        logger.info(
            f"{LOG_PREFIX} Stored reflection: seed='{thought.seed_concept}', "
            f"type={thought.thought_type}, boosted={len(boosted_ids)} concepts"
        )

        return ActionResult(
            action_name='REFLECT',
            success=True,
            details={
                'seed_concept': thought.seed_concept,
                'thought_type': thought.thought_type,
                'topic': thought.seed_topic,
                'enriched_content': enriched_content[:200],
                'boosted_concept_ids': boosted_ids,
                'connects_to': concept_names,
            }
        )

    def on_outcome(self, result: ActionResult, user_feedback: Optional[Dict] = None) -> None:
        """Store structured metadata for future causal tracking (v2 hook)."""
        if not result.success:
            return

        topic = result.details.get('topic', 'general')
        outcomes_key = _key(topic, 'outcomes')

        content_hash = hashlib.md5(
            result.details.get('enriched_content', '').encode()
        ).hexdigest()[:12]

        entry = json.dumps({
            'ts': time.time(),
            'seed_concept': result.details.get('seed_concept'),
            'thought_type': result.details.get('thought_type'),
            'association_ids': result.details.get('boosted_concept_ids', []),
            'reflection_content_hash': content_hash,
        })

        self.store.rpush(outcomes_key, entry)
        self.store.ltrim(outcomes_key, -20, -1)

    # -- Helpers ---------------------------------------------------------------

    def _analyze_act_strategies(self, topic: str) -> Optional[Dict]:
        """Compare recent ACT loop tool combinations for strategy insights."""
        if not self._db_service:
            return None
        try:
            with self._db_service.connection() as conn:
                cursor = conn.cursor()
                # SQLite: use json_each to expand actions_executed array,
                # then json_extract to pull action_type from each element.
                cursor.execute("""
                    SELECT ci.loop_id,
                           ci.net_value,
                           ci.started_at,
                           ci.completed_at,
                           ci.termination_reason,
                           ci.actions_executed
                    FROM cortex_iterations ci
                    WHERE ci.topic = ?
                      AND ci.created_at > datetime('now', '-24 hours')
                      AND ci.actions_executed IS NOT NULL
                      AND json_array_length(ci.actions_executed) > 0
                    ORDER BY ci.created_at DESC
                    LIMIT 50
                """, (topic,))
                raw_rows = cursor.fetchall()
                cursor.close()

                if len(raw_rows) < 2:
                    return None

                # Group by loop_id and aggregate
                loop_data = {}
                for row in raw_rows:
                    loop_id = row[0]
                    net_value = row[1] or 0.0
                    started_at = row[2]
                    completed_at = row[3]
                    actions_json = row[5]

                    # Parse actions_executed JSON
                    tool_types = set()
                    if actions_json:
                        try:
                            actions = json.loads(actions_json) if isinstance(actions_json, str) else actions_json
                            for a in actions:
                                if isinstance(a, dict) and 'action_type' in a:
                                    tool_types.add(a['action_type'])
                        except (json.JSONDecodeError, TypeError):
                            pass

                    if loop_id not in loop_data:
                        loop_data[loop_id] = {
                            'tool_types': set(),
                            'total_net_value': 0.0,
                            'iteration_count': 0,
                            'loop_start': started_at,
                            'loop_end': completed_at,
                        }
                    entry = loop_data[loop_id]
                    entry['tool_types'] |= tool_types
                    entry['total_net_value'] += net_value
                    entry['iteration_count'] += 1
                    if started_at and (entry['loop_start'] is None or started_at < entry['loop_start']):
                        entry['loop_start'] = started_at
                    if completed_at and (entry['loop_end'] is None or completed_at > entry['loop_end']):
                        entry['loop_end'] = completed_at

                loops = list(loop_data.values())[:10]

                if len(loops) < 2:
                    return None

                strategy_outcomes = {}
                for loop in loops:
                    tools = frozenset(loop['tool_types'])
                    net_value = loop['total_net_value']
                    iterations = loop['iteration_count']
                    seconds = 0.0
                    if loop['loop_start'] and loop['loop_end']:
                        try:
                            from datetime import datetime
                            t_start = datetime.fromisoformat(loop['loop_start']) if isinstance(loop['loop_start'], str) else loop['loop_start']
                            t_end = datetime.fromisoformat(loop['loop_end']) if isinstance(loop['loop_end'], str) else loop['loop_end']
                            seconds = (t_end - t_start).total_seconds()
                        except Exception:
                            pass
                    complexity = 'simple' if iterations <= 2 else ('moderate' if iterations <= 4 else 'complex')

                    strategy_outcomes.setdefault(tools, []).append({
                        'net_value': net_value,
                        'iterations': iterations,
                        'seconds': seconds,
                        'complexity': complexity,
                    })

                if len(strategy_outcomes) < 2:
                    return None

                ranked = sorted(
                    strategy_outcomes.items(),
                    key=lambda x: sum(e['net_value'] for e in x[1]) / len(x[1]),
                    reverse=True,
                )
                best = ranked[0]
                worst = ranked[-1]
                best_entries = best[1]
                worst_entries = worst[1]

                return {
                    'best_strategy': ', '.join(sorted(best[0])),
                    'best_avg_value': sum(e['net_value'] for e in best_entries) / len(best_entries),
                    'best_avg_seconds': sum(e['seconds'] for e in best_entries) / len(best_entries),
                    'best_complexity': max(set(e['complexity'] for e in best_entries), key=list(e['complexity'] for e in best_entries).count),
                    'worst_strategy': ', '.join(sorted(worst[0])),
                    'worst_avg_value': sum(e['net_value'] for e in worst_entries) / len(worst_entries),
                    'worst_avg_seconds': sum(e['seconds'] for e in worst_entries) / len(worst_entries),
                    'loops_analyzed': len(loops),
                }
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Strategy analysis failed: {e}")
            return None

    def _boost_concepts(self, concepts: List[Dict]) -> List[str]:
        """Increment access_count and update last_accessed_at for top concepts."""
        if not self._db_service or not concepts:
            return []

        boosted = []
        try:
            with self._db_service.connection() as conn:
                cursor = conn.cursor()
                for concept in concepts:
                    concept_id = concept.get('id')
                    if not concept_id:
                        continue
                    cursor.execute("""
                        UPDATE semantic_concepts
                        SET access_count = access_count + 1,
                            last_accessed_at = datetime('now')
                        WHERE id = ? AND deleted_at IS NULL
                    """, (concept_id,))
                    boosted.append(str(concept_id))
                cursor.close()
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Failed to boost concepts: {e}")

        return boosted

    def _increment_rejected(self, topic: str):
        """Track gate rejections for future adaptive threshold monitoring."""
        key = _key(topic, 'rejected_count')
        self.store.incr(key)
        self.store.expire(key, 3600)  # 1h TTL
