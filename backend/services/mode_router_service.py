# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Mode Router Service — Deterministic mode selection for frontal cortex.

Decouples routing from generation: a mathematical router selects the mode (~5ms),
then a mode-specific prompt drives the LLM to generate the response.

Scoring is based on observable signals (context warmth, fact density, NLP features).
A small LLM tie-breaker handles ambiguous cases when top-2 scores are within margin.
"""

import re
import time
import json
import logging
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

LOG_PREFIX = "[MODE ROUTER]"

# NLP patterns (compiled once at module load)
GREETING_PATTERNS = re.compile(
    r'^(hey|hi|hello|yo|sup|what\'?s\s*up|howdy|hiya|heya|greetings|good\s*(morning|afternoon|evening))\b',
    re.IGNORECASE
)
INTERROGATIVE_WORDS = re.compile(
    r'\b(what|where|when|who|why|how|which|whose|whom|can|could|would|should|is|are|do|does|did|will|shall)\b',
    re.IGNORECASE
)
IMPLICIT_REFERENCE = re.compile(
    r'\b(you\s+remember|we\s+discussed|last\s+time|what\s+about|we\s+talked\s+about|'
    r'remember\s+when|earlier\s+you|you\s+said|you\s+mentioned|what\s+was|what\s+were)\b',
    re.IGNORECASE
)
POSITIVE_FEEDBACK = re.compile(
    r'\b(thanks|thank\s+you|great|perfect|awesome|exactly|that\s+works|correct|good|nice|helpful|got\s+it|understood)\b',
    re.IGNORECASE
)
NEGATIVE_FEEDBACK = re.compile(
    r'\b(wrong|incorrect|no\s+that|that\'?s\s+not|doesn\'?t\s+work|confused|not\s+what\s+i|try\s+again|still\s+not|misunderstood|not\s+helpful)\b',
    re.IGNORECASE
)

def compute_nlp_signals(text: str, intent: dict = None) -> Dict[str, Any]:
    """
    Compute regex-only NLP signals from user text (<1ms).

    Standalone function so callers that already have context signals from Redis
    can merge NLP signals without re-reading Redis.

    Args:
        text: User's raw prompt text
        intent: Optional intent classification dict

    Returns:
        Dict of NLP-derived signals
    """
    tokens = text.split()
    prompt_token_count = len(tokens)
    has_question_mark = '?' in text
    interrogative_words = bool(INTERROGATIVE_WORDS.search(text))
    greeting_pattern = bool(GREETING_PATTERNS.match(text.strip()))
    implicit_reference = bool(IMPLICIT_REFERENCE.search(text))

    # Explicit feedback detection
    explicit_feedback = None
    if POSITIVE_FEEDBACK.search(text):
        explicit_feedback = 'positive'
    elif NEGATIVE_FEEDBACK.search(text):
        explicit_feedback = 'negative'

    # Information density (unique tokens / total tokens)
    unique_tokens = len(set(t.lower() for t in tokens))
    information_density = unique_tokens / max(prompt_token_count, 1)

    signals = {
        'prompt_token_count': prompt_token_count,
        'has_question_mark': has_question_mark,
        'interrogative_words': interrogative_words,
        'greeting_pattern': greeting_pattern,
        'explicit_feedback': explicit_feedback,
        'information_density': information_density,
        'implicit_reference': implicit_reference,

        # Intent signals (from IntentClassifierService, if available)
        'intent_complexity': 'simple',
        'intent_type': None,
        'intent_confidence': 0.0,
    }

    if intent:
        signals['intent_complexity'] = intent.get('complexity', 'simple')
        signals['intent_type'] = intent.get('intent_type')
        signals['intent_confidence'] = intent.get('confidence', 0.0)

    return signals


def collect_routing_signals(
    text: str,
    topic: str,
    context_warmth: float,
    working_memory,
    gist_storage,
    fact_store,
    world_state_service,
    classification_result: dict,
    session_service,
    intent: dict = None,
) -> Dict[str, Any]:
    """
    Collect all routing signals from existing services and NLP analysis.

    All Redis reads (~5ms total). NLP signals from regex (<1ms).

    Args:
        text: User's raw prompt text
        topic: Resolved topic name
        context_warmth: Pre-computed warmth (0.0-1.0)
        working_memory: WorkingMemoryService instance
        gist_storage: GistStorageService instance
        fact_store: FactStoreService instance
        world_state_service: WorldStateService instance
        classification_result: Dict from topic classifier
        session_service: SessionService instance
        intent: Optional intent classification dict from IntentClassifierService
            (tool selection handled by CognitiveTriageService)

    Returns:
        Dict of routing signals
    """
    # Context signals (from existing services, all Redis reads)
    wm_turns = working_memory.get_recent_turns(topic) if topic else []
    working_memory_turns = len(wm_turns)

    gists = gist_storage.get_latest_gists(topic) if topic else []
    gist_count = sum(1 for g in gists if g.get('type') != 'cold_start')

    facts = fact_store.get_all_facts(topic) if topic else []
    fact_count = len(facts)
    fact_keys = [f.get('key', '') for f in facts]

    world_state = world_state_service.get_world_state(topic) if topic else ""
    world_state_present = bool(world_state and world_state.strip())

    # Classification signals
    topic_confidence = classification_result.get('confidence', 0.5)
    is_new_topic = classification_result.get('is_new_topic', False)

    # Session signals
    session_exchange_count = getattr(session_service, 'topic_exchange_count', 0) if session_service else 0

    # Memory confidence signal: FOK (Feeling-of-Knowing) per topic
    # Read from Redis (set by recall skill), compute composite confidence score
    from services.redis_client import RedisClientService
    redis_conn = RedisClientService.create_connection(decode_responses=True)
    raw_fok = redis_conn.get(f"fok:{topic}") if topic else None
    fok = float(raw_fok) if raw_fok else 0.0
    fok_score = min(1.0, fok / 5.0)

    density_score = min(1.0, (gist_count + fact_count) / 6.0)

    memory_confidence = (
        0.4 * fok_score
        + 0.4 * context_warmth
        + 0.2 * density_score
    )
    if is_new_topic:
        memory_confidence *= 0.7
    memory_confidence = round(memory_confidence, 3)

    # NLP signals via standalone function
    nlp = compute_nlp_signals(text, intent)

    signals = {
        # Context signals
        'context_warmth': context_warmth,
        'working_memory_turns': working_memory_turns,
        'gist_count': gist_count,
        'fact_count': fact_count,
        'fact_keys': fact_keys,
        'world_state_present': world_state_present,
        'topic_confidence': topic_confidence,
        'is_new_topic': is_new_topic,
        'session_exchange_count': session_exchange_count,
        'memory_confidence': memory_confidence,
    }

    # Merge NLP signals
    signals.update(nlp)

    return signals


class ModeRouterService:
    """
    Deterministic mode router with tie-breaker LLM fallback.

    Scores each mode based on weighted signal composites, selects highest.
    When top-2 are within margin, invokes small LLM for disambiguation.
    """

    MODES = ['RESPOND', 'CLARIFY', 'ACT', 'ACKNOWLEDGE', 'IGNORE']

    MODE_DESCRIPTIONS = {
        'RESPOND': 'Answer the user with available context and knowledge',
        'CLARIFY': 'Ask one clarifying question to build understanding',
        'ACT': 'Use tools and skills (web search, memory lookup, research) to gather information before responding',
        'ACKNOWLEDGE': 'Give a brief social acknowledgment (greeting, thanks, etc.)',
        'IGNORE': 'No response needed (empty or irrelevant input)',
    }

    def __init__(self, config: dict):
        """
        Initialize mode router.

        Args:
            config: Mode router configuration (from mode-router.json)
        """
        self.config = config

        # Base scores per mode
        self.bases = config.get('base_scores', {
            'RESPOND': 0.50,
            'CLARIFY': 0.30,
            'ACT': 0.20,
            'ACKNOWLEDGE': 0.10,
            'IGNORE': -0.50,
        })

        # Weight parameters (tunable)
        self.weights = config.get('weights', {})

        # Tie-breaker config
        self.tiebreaker_base_margin = config.get('tiebreaker_base_margin', 0.20)
        self.tiebreaker_min_margin = config.get('tiebreaker_min_margin', 0.08)

        # Hysteresis tracking (in-memory, per topic)
        self._confidence_history: Dict[str, List[float]] = {}

        # LLM tie-breaker (lazy init)
        self._tiebreaker = None
        self._tiebreaker_prompt = None

    def route(
        self,
        signals: Dict[str, Any],
        prompt_text: str,
        previous_mode: Optional[str] = None,
        previous_router_confidence: Optional[float] = None,
        skip_tiebreaker: bool = False,
    ) -> Dict[str, Any]:
        """
        Select the best engagement mode based on signals.

        Args:
            signals: Dict from collect_routing_signals()
            prompt_text: Raw user text (for tie-breaker context)
            previous_mode: Mode from last exchange (anti-oscillation)
            previous_router_confidence: Confidence from last routing decision
            skip_tiebreaker: Skip LLM tie-breaker even when scores are close.
                Use for post-ACT re-routing where terminal mode is already implicit.

        Returns:
            Dict with routing decision:
            {
                'mode': str,
                'scores': Dict[str, float],
                'router_confidence': float,
                'tiebreaker_used': bool,
                'tiebreaker_candidates': list | None,
                'margin': float,
                'effective_margin': float,
                'signal_snapshot': dict,
                'weight_snapshot': dict,
                'routing_time_ms': float,
            }
        """
        start_time = time.time()

        # Score all modes
        scores = self._score_all_modes(signals, previous_mode)

        # Sort by score (descending)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_mode, top_score = ranked[0]
        runner_up_mode, runner_up_score = ranked[1]

        # Calculate router confidence
        router_confidence = self._calculate_confidence(top_score, runner_up_score)

        # Track hysteresis
        topic = signals.get('topic', 'unknown')
        self._track_confidence(topic, router_confidence)

        # Calculate effective margin
        effective_margin = self._calculate_effective_margin(signals)

        # Apply hysteresis widening if low-confidence streak
        if self._is_low_confidence_streak(topic):
            effective_margin += 0.05
            logger.debug(f"{LOG_PREFIX} Hysteresis widening: +0.05 (low confidence streak)")

        # Check if tie-breaker needed
        margin = abs(top_score - runner_up_score) / max(abs(top_score), 0.001)
        tiebreaker_used = False
        tiebreaker_candidates = None

        if margin < effective_margin and top_mode != 'IGNORE' and not skip_tiebreaker:
            tiebreaker_candidates = [top_mode, runner_up_mode]
            tiebreaker_result = self._invoke_tiebreaker(
                prompt_text, signals, top_mode, runner_up_mode
            )
            if tiebreaker_result:
                tiebreaker_used = True
                top_mode = tiebreaker_result
                logger.info(
                    f"{LOG_PREFIX} Tie-breaker selected {top_mode} "
                    f"(margin={margin:.3f} < effective={effective_margin:.3f})"
                )

        routing_time_ms = (time.time() - start_time) * 1000

        logger.info(
            f"{LOG_PREFIX} Route: {top_mode} "
            f"(confidence={router_confidence:.3f}, "
            f"scores={{{', '.join(f'{k}:{v:.3f}' for k, v in ranked)}}}, "
            f"tiebreaker={'yes' if tiebreaker_used else 'no'}, "
            f"time={routing_time_ms:.1f}ms)"
        )

        return {
            'mode': top_mode,
            'scores': dict(scores),
            'router_confidence': router_confidence,
            'tiebreaker_used': tiebreaker_used,
            'tiebreaker_candidates': tiebreaker_candidates,
            'margin': margin,
            'effective_margin': effective_margin,
            'signal_snapshot': signals,
            'weight_snapshot': self._get_weight_snapshot(),
            'routing_time_ms': routing_time_ms,
        }

    def _score_all_modes(
        self,
        signals: Dict[str, Any],
        previous_mode: Optional[str] = None,
    ) -> Dict[str, float]:
        """Score each mode based on signal composites."""
        warmth = signals['context_warmth']
        fact_count = signals['fact_count']
        gist_count = signals['gist_count']
        has_q = signals['has_question_mark']
        interrog = signals['interrogative_words']
        greeting = signals['greeting_pattern']
        feedback = signals['explicit_feedback']
        density = signals['information_density']
        implicit_ref = signals['implicit_reference']
        wm_turns = signals['working_memory_turns']
        is_new = signals['is_new_topic']
        token_count = signals['prompt_token_count']

        # Derived signals
        fact_density = min(fact_count / 10.0, 1.0)
        gist_density = min(gist_count / 5.0, 1.0)
        is_question = has_q or interrog
        is_cold = warmth < 0.3
        is_warm = warmth > 0.6
        is_empty = token_count == 0

        w = self.weights

        # ── RESPOND ──────────────────────────────────────────────
        respond = self.bases['RESPOND']
        respond += warmth * w.get('respond.warmth_boost', 0.20)
        respond += fact_density * w.get('respond.fact_density', 0.15)
        respond += gist_density * w.get('respond.gist_density', 0.10)
        if is_question and warmth > 0.4:
            respond += w.get('respond.question_warm', 0.15)
        if is_cold:
            respond -= w.get('respond.cold_penalty', 0.15)
        if greeting:
            respond -= w.get('respond.greeting_penalty', 0.20)
        if feedback == 'positive':
            respond -= w.get('respond.feedback_penalty', 0.15)
        # Tool-needed penalty removed — tool dispatch now handled by CognitiveTriageService

        # ── CLARIFY ──────────────────────────────────────────────
        clarify = self.bases['CLARIFY']
        if is_cold:
            clarify += (1.0 - warmth) * w.get('clarify.cold_boost', 0.25)
        if is_question and is_cold:
            clarify += w.get('clarify.cold_question', 0.05)
        if is_question and fact_count == 0:
            clarify += w.get('clarify.question_no_facts', 0.20)
        if is_new and is_question:
            clarify += w.get('clarify.new_topic_question', 0.10)
        if is_warm:
            clarify -= w.get('clarify.warm_penalty', 0.20)

        # ── ACT ──────────────────────────────────────────────────
        act = self.bases['ACT']
        if is_question and 0.3 <= warmth <= 0.7:
            act += w.get('act.question_moderate_context', 0.20)
        if interrog and fact_count < 3:
            act += w.get('act.interrogative_gap', 0.15)
        if implicit_ref:
            act += w.get('act.implicit_reference', 0.15)
        # Tool relevance weights removed — CognitiveTriageService handles tool dispatch
        if warmth < 0.15:
            act -= w.get('act.very_cold_penalty', 0.10)
        if is_warm and fact_density > 0.5:
            act -= w.get('act.warm_facts_penalty', 0.10)
        # Graduated memory confidence: low recall confidence → lean toward ACT
        mem_conf = signals.get('memory_confidence', 1.0)
        if interrog or has_q:
            if mem_conf < 0.15:
                act += w.get('act.memory_confidence_very_low', 0.20)
            elif mem_conf < 0.30:
                act += w.get('act.memory_confidence_low', 0.10)

        # ── ACKNOWLEDGE ──────────────────────────────────────────
        acknowledge = self.bases['ACKNOWLEDGE']
        if greeting:
            acknowledge += w.get('acknowledge.greeting', 0.60)
        if feedback == 'positive':
            acknowledge += w.get('acknowledge.positive_feedback', 0.40)
        if is_question:
            acknowledge -= w.get('acknowledge.question_penalty', 0.30)

        # ── IGNORE ───────────────────────────────────────────────
        ignore = self.bases['IGNORE']
        if is_empty:
            ignore += w.get('ignore.empty_input', 1.00)

        scores = {
            'RESPOND': respond,
            'CLARIFY': clarify,
            'ACT': act,
            'ACKNOWLEDGE': acknowledge,
            'IGNORE': ignore,
        }

        # Apply anti-oscillation ephemeral adjustments
        if previous_mode:
            scores = self._apply_anti_oscillation(scores, previous_mode)

        return scores

    def _apply_anti_oscillation(
        self,
        scores: Dict[str, float],
        previous_mode: str,
    ) -> Dict[str, float]:
        """
        Ephemeral per-request adjustments to prevent routing oscillation.

        These are NOT persistent weight changes.
        """
        adjusted = dict(scores)

        if previous_mode == 'ACT':
            # ACT just ran — suppress ACT re-selection
            adjusted['ACT'] -= 0.15
            logger.debug(f"{LOG_PREFIX} Anti-oscillation: ACT suppressed (previous was ACT)")
        elif previous_mode == 'CLARIFY':
            # User just answered a question — respond to it
            adjusted['RESPOND'] += 0.05
            logger.debug(f"{LOG_PREFIX} Anti-oscillation: RESPOND boosted (previous was CLARIFY)")

        return adjusted

    def _calculate_confidence(self, top_score: float, runner_up_score: float) -> float:
        """
        Router confidence metric.

        Higher when top score clearly dominates runner-up.
        """
        denominator = max(abs(top_score), 0.001)
        return (top_score - runner_up_score) / denominator

    def _calculate_effective_margin(self, signals: Dict[str, Any]) -> float:
        """
        Margin narrows with warmth, widens with semantic uncertainty.

        Warm context → more deterministic (smaller margin → fewer tie-breakers).
        Ambiguous signals → wider margin (more tie-breaker consultation).
        """
        warmth = signals['context_warmth']
        base = self.tiebreaker_base_margin
        minimum = self.tiebreaker_min_margin

        # Base narrowing with warmth
        margin = base - (base - minimum) * warmth

        # Semantic uncertainty widens margin
        semantic_uncertainty = 0.0
        if signals['implicit_reference']:
            semantic_uncertainty += 0.05
        if signals['information_density'] < 0.3:
            semantic_uncertainty += 0.03
        if signals['interrogative_words'] and not signals['has_question_mark']:
            semantic_uncertainty += 0.03

        margin += semantic_uncertainty

        return margin

    def _track_confidence(self, topic: str, confidence: float):
        """Track confidence for hysteresis detection."""
        if topic not in self._confidence_history:
            self._confidence_history[topic] = []
        history = self._confidence_history[topic]
        history.append(confidence)
        # Keep last 3 only
        if len(history) > 3:
            self._confidence_history[topic] = history[-3:]

    def _is_low_confidence_streak(self, topic: str) -> bool:
        """Check if last 3 exchanges had low confidence (hysteresis trigger)."""
        history = self._confidence_history.get(topic, [])
        if len(history) < 3:
            return False
        return all(c < 0.15 for c in history[-3:])

    def _get_weight_snapshot(self) -> Dict:
        """Return current weight configuration for logging."""
        return {
            'bases': dict(self.bases),
            'tiebreaker_base_margin': self.tiebreaker_base_margin,
            'tiebreaker_min_margin': self.tiebreaker_min_margin,
        }

    # ── Tie-breaker ──────────────────────────────────────────────

    def _invoke_tiebreaker(
        self,
        prompt_text: str,
        signals: Dict[str, Any],
        mode_a: str,
        mode_b: str,
    ) -> Optional[str]:
        """
        Invoke small LLM to break tie between top-2 modes.

        Returns the selected mode, or None on failure (falls back to higher score).
        """
        try:
            if self._tiebreaker is None:
                self._init_tiebreaker()

            # Build tie-breaker prompt
            prompt = self._build_tiebreaker_prompt(prompt_text, signals, mode_a, mode_b)

            response = self._tiebreaker.send_message(
                self._tiebreaker_prompt, prompt
            ).text

            # Parse response with fallback tiers
            selected = self._extract_tiebreaker_choice(response, mode_a, mode_b)
            if selected:
                return selected
            logger.warning(f"{LOG_PREFIX} Tie-breaker returned unparseable response: '{response[:100]}'")
            return None

        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Tie-breaker failed: {e}")
            return None

    def _extract_tiebreaker_choice(self, response: str, mode_a: str, mode_b: str) -> Optional[str]:
        """Extract choice from tie-breaker response with fallback parsing."""
        # Tier 1: Direct JSON parse
        try:
            result = json.loads(response)
            choice = result.get('choice', '').upper().strip()
            if choice in ('A', 'B'):
                return mode_a if choice == 'A' else mode_b
        except (json.JSONDecodeError, AttributeError):
            pass

        # Tier 2: Extract JSON object from surrounding text
        match = re.search(r'\{[^}]+\}', response)
        if match:
            try:
                result = json.loads(match.group())
                choice = result.get('choice', '').upper().strip()
                if choice in ('A', 'B'):
                    return mode_a if choice == 'A' else mode_b
            except (json.JSONDecodeError, AttributeError):
                pass

        # Tier 3: Regex for raw "A" or "B" answer
        match = re.search(r'"choice"\s*:\s*"([AB])"', response, re.IGNORECASE)
        if match:
            choice = match.group(1).upper()
            return mode_a if choice == 'A' else mode_b

        return None

    def _init_tiebreaker(self):
        """Lazy-initialize tie-breaker LLM."""
        from services.llm_service import create_refreshable_llm_service
        from services.config_service import ConfigService

        self._tiebreaker = create_refreshable_llm_service("mode-tiebreaker")
        self._tiebreaker_prompt = ConfigService.get_agent_prompt("mode-tiebreaker")

    def _build_tiebreaker_prompt(
        self,
        prompt_text: str,
        signals: Dict[str, Any],
        mode_a: str,
        mode_b: str,
    ) -> str:
        """Build user message for tie-breaker LLM."""
        context_lines = [
            f"User message: \"{prompt_text}\"",
            f"Context warmth: {signals['context_warmth']:.2f}",
            f"Known facts: {signals['fact_count']}",
            f"Working memory turns: {signals['working_memory_turns']}",
            f"Topic: {signals.get('topic', 'unknown')} ({'new' if signals.get('is_new_topic') else 'existing'})",
            f"Gist count: {signals['gist_count']}",
        ]

        return (
            f"Context:\n" + "\n".join(context_lines) + "\n\n"
            f"A: {mode_a} — {self.MODE_DESCRIPTIONS.get(mode_a, '')}\n"
            f"B: {mode_b} — {self.MODE_DESCRIPTIONS.get(mode_b, '')}\n\n"
            f"Pick the best engagement mode. Respond with: {{\"choice\": \"A\" or \"B\"}}"
        )
