"""
SuggestAction — Proactive skill suggestions based on user traits.

Bridges the gap between "Chalie knows things about you" and "Chalie helps
you with things." Fires when a drift thought connects to a high-confidence
user trait AND a relevant innate skill.

Priority: 8 (below COMMUNICATE=10, above SEED_THREAD=6)

Gates:
  1. Phase gate: Spark must be in connected or graduated
  2. Trait gate: At least 3 user traits with confidence >= 0.7
  3. Relevance gate: Thought embedding similarity to a trait > 0.4
  4. Skill match: At least one innate skill is relevant
  5. Rate limit: Max 1 suggestion per 24h + per-topic 7-day cooldown
  6. Engagement gate: Not paused, engagement score > 0.5
"""

import json
import logging
import math
import time
import uuid
from typing import Optional, Dict, Any, Tuple, List

from services.redis_client import RedisClientService
from .base import AutonomousAction, ActionResult, ThoughtContext

logger = logging.getLogger(__name__)

LOG_PREFIX = "[SUGGEST]"

_NS = "spark_suggest"

# Innate skills that can be suggested
_SUGGESTABLE_SKILLS = {
    'schedule': 'Set reminders and schedule tasks',
    'list': 'Create and manage lists',
    'recall': 'Search and retrieve memories',
    'memorize': 'Explicitly store important information',
}

# Rate limit keys
_DAILY_COOLDOWN = 86400  # 24h
_TOPIC_COOLDOWN = 604800  # 7 days
_REJECT_WINDOW = 604800  # 7 days for rejection tracking
_AUTO_PAUSE_DURATION = 1209600  # 14 days


def _key(user_id: str, suffix: str) -> str:
    return f"{_NS}:{user_id}:{suffix}"


class SuggestAction(AutonomousAction):
    """
    Evaluates whether a drift thought should trigger a proactive skill suggestion.

    Connects user traits to innate skills when drift thoughts create a natural bridge.
    """

    def __init__(self, config: dict = None):
        super().__init__(name='SUGGEST', enabled=True, priority=8)

        config = config or {}
        self.redis = RedisClientService.create_connection()
        self.user_id = config.get('user_id', 'default')

        self.min_trait_confidence = config.get('min_trait_confidence', 0.7)
        self.relevance_threshold = config.get('relevance_threshold', 0.4)
        self.min_traits_for_suggestion = config.get('min_traits_for_suggestion', 3)
        self.reject_pause_threshold = config.get('reject_pause_threshold', 0.4)

        self._embedding_service = None

    @property
    def embedding_service(self):
        if self._embedding_service is None:
            from services.embedding_service import EmbeddingService
            self._embedding_service = EmbeddingService()
        return self._embedding_service

    # ── Gate checks ───────────────────────────────────────────────

    def _phase_gate(self) -> bool:
        """Spark must be in connected or graduated phase."""
        try:
            from services.spark_state_service import SparkStateService
            phase = SparkStateService().get_phase()
            return phase in ('connected', 'graduated')
        except Exception:
            return False

    def _trait_gate(self) -> Tuple[bool, List[Dict]]:
        """At least 3 traits with confidence >= threshold."""
        try:
            from services.database_service import get_shared_db_service
            from services.user_trait_service import UserTraitService

            db = get_shared_db_service()
            trait_service = UserTraitService(db)
            traits = trait_service.get_all_traits()

            high_conf = [
                t for t in traits
                if t.get('confidence', 0) >= self.min_trait_confidence
            ]

            if len(high_conf) < self.min_traits_for_suggestion:
                return (False, [])

            return (True, high_conf)
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Trait gate failed: {e}")
            return (False, [])

    def _relevance_gate(
        self, thought: ThoughtContext, traits: List[Dict]
    ) -> Tuple[bool, Optional[Dict], float]:
        """Check embedding similarity between thought and traits."""
        if not thought.thought_embedding:
            return (False, None, 0.0)

        best_trait = None
        best_sim = 0.0

        for trait in traits:
            trait_text = f"{trait.get('key', '')}: {trait.get('value', '')}"
            try:
                trait_emb = self.embedding_service.embed(trait_text)
                sim = self._cosine_similarity(thought.thought_embedding, trait_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_trait = trait
            except Exception:
                continue

        if best_sim < self.relevance_threshold or not best_trait:
            return (False, None, best_sim)

        return (True, best_trait, best_sim)

    def _skill_match(self, trait: Dict, thought: ThoughtContext) -> Optional[str]:
        """Find a relevant skill for this trait + thought combination."""
        # Simple keyword matching against trait and thought content
        combined = f"{trait.get('key', '')} {trait.get('value', '')} {thought.thought_content}".lower()

        # Priority order of skill matching
        skill_keywords = {
            'schedule': ['morning', 'evening', 'routine', 'time', 'remind', 'deadline',
                         'meeting', 'appointment', 'daily', 'weekly', 'gym', 'workout',
                         'medication', 'habit'],
            'list': ['track', 'organize', 'items', 'groceries', 'todo', 'shopping',
                     'tasks', 'goals', 'project', 'plan', 'checklist'],
            'recall': ['remember', 'forgot', 'memory', 'earlier', 'mentioned',
                       'last time', 'history', 'previous'],
            'memorize': ['important', 'note', 'save', 'keep', 'store', 'reference',
                         'document', 'record'],
        }

        for skill, keywords in skill_keywords.items():
            if any(kw in combined for kw in keywords):
                return skill

        return None

    def _rate_limit_gate(self, topic: str = None) -> bool:
        """Check daily and per-topic cooldowns."""
        now = time.time()

        # Global daily cooldown
        last_sent = self.redis.get(_key(self.user_id, 'last_suggest_ts'))
        if last_sent and (now - float(last_sent)) < _DAILY_COOLDOWN:
            return False

        # Per-topic cooldown
        if topic:
            topic_key = _key(self.user_id, f'topic_cooldown:{topic}')
            if self.redis.get(topic_key):
                return False

        # Auto-pause check
        paused_until = self.redis.get(_key(self.user_id, 'paused_until'))
        if paused_until and now < float(paused_until):
            return False

        return True

    def _engagement_gate(self) -> bool:
        """Check engagement score is above threshold."""
        try:
            score = float(
                self.redis.get(f"proactive:{self.user_id}:engagement_score") or 1.0
            )
            paused = self.redis.get(f"proactive:{self.user_id}:paused")
            if paused == '1':
                return False
            return score > 0.5
        except Exception:
            return True

    def _check_suggestion_seeded(self) -> bool:
        """Check if first soft capability seed has been sent."""
        try:
            from services.spark_state_service import SparkStateService
            state = SparkStateService().get_state()
            return state.get('first_suggestion_seeded', False)
        except Exception:
            return False

    # ── Main interface ────────────────────────────────────────────

    def should_execute(self, thought: ThoughtContext) -> tuple:
        """Evaluate all gates. Returns (score, eligible)."""

        # Gate 1: Phase
        if not self._phase_gate():
            return (0.0, False)

        # Gate 2: Traits
        trait_passes, traits = self._trait_gate()
        if not trait_passes:
            return (0.0, False)

        # Gate 3: Relevance
        rel_passes, best_trait, rel_score = self._relevance_gate(thought, traits)
        if not rel_passes:
            return (0.0, False)

        # Gate 4: Skill match
        skill = self._skill_match(best_trait, thought)
        if not skill:
            return (0.0, False)

        # Gate 5: Rate limit
        if not self._rate_limit_gate(thought.seed_topic):
            return (0.0, False)

        # Gate 6: Engagement
        if not self._engagement_gate():
            return (0.0, False)

        # Store context for execute()
        self._pending_trait = best_trait
        self._pending_skill = skill
        self._pending_relevance = rel_score

        score = rel_score * thought.activation_energy
        return (score, True)

    def execute(self, thought: ThoughtContext) -> ActionResult:
        """Generate and deliver a skill suggestion."""
        trait = getattr(self, '_pending_trait', None)
        skill = getattr(self, '_pending_skill', None)

        if not trait or not skill:
            return ActionResult(action_name='SUGGEST', success=False,
                                details={'reason': 'no_pending_context'})

        # Check if this is the first suggestion — send soft seed instead
        if not self._check_suggestion_seeded():
            return self._send_capability_seed()

        # Generate suggestion
        suggestion = self._generate_suggestion(trait, skill, thought)
        if not suggestion:
            return ActionResult(action_name='SUGGEST', success=False,
                                details={'reason': 'generation_failed'})

        # Deliver
        self._deliver_suggestion(suggestion)

        # Update rate limits
        now = time.time()
        self.redis.set(_key(self.user_id, 'last_suggest_ts'), str(now))
        if thought.seed_topic:
            topic_key = _key(self.user_id, f'topic_cooldown:{thought.seed_topic}')
            self.redis.setex(topic_key, _TOPIC_COOLDOWN, '1')

        # Log
        self._log_suggestion_event(trait, skill, self._pending_relevance)

        logger.info(
            f"{LOG_PREFIX} Suggestion sent: skill={skill}, "
            f"trait={trait.get('key')}, relevance={self._pending_relevance:.2f}"
        )

        return ActionResult(
            action_name='SUGGEST',
            success=True,
            details={
                'skill': skill,
                'trait_key': trait.get('key'),
                'relevance': self._pending_relevance,
            },
        )

    def _send_capability_seed(self) -> ActionResult:
        """Send a soft introductory message before the first real suggestion."""
        seed_text = (
            "If you ever want, I can also help with things like reminders "
            "or keeping track of routines."
        )

        self._deliver_suggestion(seed_text)

        try:
            from services.spark_state_service import SparkStateService
            SparkStateService().mark_suggestion_seeded()
        except Exception:
            pass

        logger.info(f"{LOG_PREFIX} Capability seed sent")

        return ActionResult(
            action_name='SUGGEST',
            success=True,
            details={'type': 'capability_seed'},
        )

    def _generate_suggestion(
        self, trait: Dict, skill: str, thought: ThoughtContext
    ) -> Optional[str]:
        """Generate suggestion via LLM with fallback."""
        try:
            from services.config_service import ConfigService
            from services.llm_service import create_llm_service

            try:
                config = ConfigService.resolve_agent_config("frontal-cortex-acknowledge")
            except Exception:
                config = ConfigService.resolve_agent_config("frontal-cortex")

            config = dict(config)
            config['format'] = ''

            prompt_template = ConfigService.get_agent_prompt("spark-suggest")

            # Inject variables
            prompt = prompt_template.replace('{{trait_key}}', trait.get('key', ''))
            prompt = prompt.replace('{{trait_value}}', str(trait.get('value', '')))
            prompt = prompt.replace('{{trait_confidence}}', str(trait.get('confidence', 0)))
            prompt = prompt.replace('{{skill_name}}', skill)
            prompt = prompt.replace('{{skill_description}}', _SUGGESTABLE_SKILLS.get(skill, ''))
            prompt = prompt.replace('{{thought_content}}', thought.thought_content)

            llm = create_llm_service(config)
            response = llm.send_message(prompt, "Generate a suggestion.").text

            text = response.strip().strip('"').strip("'").strip()
            return text if text else None

        except Exception as e:
            logger.warning(f"{LOG_PREFIX} LLM suggestion generation failed: {e}")
            return None

    def _deliver_suggestion(self, text: str) -> None:
        """Deliver suggestion via OutputService (drift stream)."""
        try:
            from services.output_service import OutputService

            output_svc = OutputService()
            output_svc.enqueue_text(
                topic='spark_suggest',
                response=text,
                mode='RESPOND',
                confidence=0.8,
                generation_time=0.0,
                original_metadata={
                    'source': 'spark_suggest',
                },
            )
        except Exception as e:
            logger.error(f"{LOG_PREFIX} Delivery failed: {e}")

    def _log_suggestion_event(
        self, trait: Dict, skill: str, relevance: float
    ) -> None:
        """Log suggestion event to interaction_log."""
        try:
            from services.database_service import get_shared_db_service
            from services.interaction_log_service import InteractionLogService

            db = get_shared_db_service()
            log_service = InteractionLogService(db)
            log_service.log_event(
                event_type='spark_suggestion_sent',
                payload={
                    'trait_key': trait.get('key'),
                    'skill_name': skill,
                    'embedding_score': relevance,
                },
                topic='spark_suggest',
                source='spark_suggest',
            )
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Suggestion event logging failed: {e}")

    def on_outcome(
        self, result: ActionResult, user_feedback: Optional[Dict] = None
    ) -> None:
        """Track suggestion rejection rate for auto-pause."""
        if not user_feedback:
            return

        outcome = user_feedback.get('outcome', 'unknown')

        # Track rejections
        rejections_key = _key(self.user_id, 'rejection_log')
        entry = json.dumps({
            'outcome': outcome,
            'ts': time.time(),
        })
        self.redis.rpush(rejections_key, entry)
        self.redis.ltrim(rejections_key, -20, -1)
        self.redis.expire(rejections_key, _REJECT_WINDOW)

        # Check rejection rate
        raw = self.redis.lrange(rejections_key, 0, -1)
        if raw and len(raw) >= 3:
            now = time.time()
            window_entries = []
            for r in raw:
                try:
                    e = json.loads(r)
                    if now - e.get('ts', 0) < _REJECT_WINDOW:
                        window_entries.append(e)
                except Exception:
                    continue

            if window_entries:
                rejects = sum(
                    1 for e in window_entries
                    if e.get('outcome') in ('rejected', 'ignored', 'dismissed')
                )
                rate = rejects / len(window_entries)
                if rate > self.reject_pause_threshold:
                    pause_until = now + _AUTO_PAUSE_DURATION
                    self.redis.set(
                        _key(self.user_id, 'paused_until'), str(pause_until)
                    )
                    logger.info(
                        f"{LOG_PREFIX} Auto-paused for 14 days "
                        f"(rejection rate {rate:.0%})"
                    )

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
