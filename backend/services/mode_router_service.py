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

    Standalone function so callers that already have context signals from MemoryStore
    can merge NLP signals without re-reading MemoryStore.

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
    world_state_service,
    classification_result: dict,
    session_service,
    intent: dict = None,
) -> Dict[str, Any]:
    """
    Collect all routing signals from existing services and NLP analysis.

    All MemoryStore reads (~5ms total). NLP signals from regex (<1ms).

    Args:
        text: User's raw prompt text
        topic: Resolved topic name
        context_warmth: Pre-computed warmth (0.0-1.0)
        working_memory: WorkingMemoryService instance
        world_state_service: WorldStateService instance
        classification_result: Dict from topic classifier
        session_service: SessionService instance
        intent: Optional intent classification dict from IntentClassifierService
            (tool selection handled by CognitiveTriageService)

    Returns:
        Dict of routing signals
    """
    # Context signals (from existing services, all MemoryStore reads)
    wm_turns = working_memory.get_recent_turns(topic) if topic else []
    working_memory_turns = len(wm_turns)

    world_state = world_state_service.get_world_state(topic) if topic else ""
    world_state_present = bool(world_state and world_state.strip())

    # Classification signals
    topic_confidence = classification_result.get('confidence', 0.5)
    is_new_topic = classification_result.get('is_new_topic', False)

    # Session signals
    session_exchange_count = getattr(session_service, 'topic_exchange_count', 0) if session_service else 0

    # Memory confidence signal: FOK (Feeling-of-Knowing) per topic
    # Read from MemoryStore (set by recall skill), compute composite confidence score
    from services.memory_client import MemoryClientService
    store = MemoryClientService.create_connection(decode_responses=True)
    raw_fok = store.get(f"fok:{topic}") if topic else None
    fok = float(raw_fok) if raw_fok else 0.0
    fok_score = min(1.0, fok / 5.0)

    wm_depth_score = min(1.0, working_memory_turns / 6.0)

    memory_confidence = (
        0.4 * fok_score
        + 0.4 * context_warmth
        + 0.2 * wm_depth_score
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
        'gist_count': 0,
        'fact_count': 0,
        'fact_keys': [],
        'world_state_present': world_state_present,
        'topic_confidence': topic_confidence,
        'is_new_topic': is_new_topic,
        'session_exchange_count': session_exchange_count,
        'memory_confidence': memory_confidence,
    }

    # Merge NLP signals
    signals.update(nlp)

    # Self-constraint from self-model (severity-weighted)
    self_constraint = 0.0
    try:
        from services.self_model_service import SelfModelService
        snapshot = SelfModelService().get_snapshot()
        noteworthy = snapshot.get("noteworthy", [])
        if noteworthy:
            self_constraint = max(item["severity"] for item in noteworthy)
    except Exception:
        pass
    signals['self_constraint'] = self_constraint

    return signals


class ModeRouterService:
    """
    Deterministic mode router with tie-breaker LLM fallback.

    Scores each mode based on weighted signal composites, selects highest.
    When top-2 are within margin, invokes small LLM for disambiguation.
    """

    MODES = ['RESPOND', 'ACT', 'IGNORE']

    MODE_DESCRIPTIONS = {
        'RESPOND': 'Answer the user with available context and knowledge',
        'ACT': 'Use tools and skills (web search, memory lookup, research) to gather information before responding',
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
            'ACT': 0.20,
            'IGNORE': -0.50,
        })

        # Weight parameters (tunable)
        self.weights = config.get('weights', {})

        # Tie-breaker config
        self.tiebreaker_base_margin = config.get('tiebreaker_base_margin', 0.20)
        self.tiebreaker_min_margin = config.get('tiebreaker_min_margin', 0.08)
        # onnx_confidence_threshold removed — ONNX always decides, no LLM fallback

        # Hysteresis tracking (in-memory, per topic)
        self._confidence_history: Dict[str, List[float]] = {}

        # LLM tie-breaker (lazy init)
        # LLM tiebreaker removed — ONNX classifier is the sole tiebreaker

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
            skip_tiebreaker: Skip ONNX tie-breaker even when scores are close.
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
        is_question = has_q or interrog
        is_cold = warmth < 0.3
        is_warm = warmth > 0.6
        is_empty = token_count == 0

        w = self.weights

        # ── RESPOND ──────────────────────────────────────────────
        respond = self.bases['RESPOND']
        respond += warmth * w.get('respond.warmth_boost', 0.20)
        if is_question and warmth > 0.4:
            respond += w.get('respond.question_warm', 0.15)
        if is_question and is_cold:
            # Questions in cold context: LLM can still answer general knowledge
            respond += w.get('respond.question_cold', 0.10)
        if is_cold:
            respond -= w.get('respond.cold_penalty', 0.15)
        # Tool-needed penalty removed — tool dispatch now handled by CognitiveTriageService

        # ── ACT ──────────────────────────────────────────────────
        act = self.bases['ACT']
        if is_question and 0.3 <= warmth <= 0.7:
            act += w.get('act.question_moderate_context', 0.20)
        if interrog and is_cold:
            act += w.get('act.interrogative_gap', 0.15)
        if implicit_ref:
            act += w.get('act.implicit_reference', 0.15)
        # Tool relevance weights removed — CognitiveTriageService handles tool dispatch
        if warmth < 0.15:
            act -= w.get('act.very_cold_penalty', 0.10)
        # Action intent — user wants Chalie to DO something (set reminder, manage list, etc.)
        if signals.get('intent_type') == 'action':
            act += w.get('act.action_intent', 0.40)
        # Graduated memory confidence: low recall confidence → lean toward ACT
        # Only for genuinely memory-seeking questions, not general knowledge
        mem_conf = signals.get('memory_confidence', 1.0)
        if interrog or has_q:
            if mem_conf < 0.15:
                act += w.get('act.memory_confidence_very_low', 0.10)
            elif mem_conf < 0.30:
                act += w.get('act.memory_confidence_low', 0.05)

        # ── IGNORE ───────────────────────────────────────────────
        ignore = self.bases['IGNORE']
        if is_empty:
            ignore += w.get('ignore.empty_input', 1.00)

        scores = {
            'RESPOND': respond,
            'ACT': act,
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

        suppressed_mode = None
        penalty = 0.0

        if previous_mode == 'ACT':
            # ACT just ran — suppress ACT re-selection
            adjusted['ACT'] -= 0.15
            suppressed_mode = 'ACT'
            penalty = -0.15
            logger.debug(f"{LOG_PREFIX} Anti-oscillation: ACT suppressed (previous was ACT)")

        if suppressed_mode:
            try:
                from services.database_service import get_shared_db_service
                from services.interaction_log_service import InteractionLogService

                db = get_shared_db_service()
                InteractionLogService(db).log_event(
                    event_type='routing_anti_oscillation',
                    payload={
                        'previous_mode': previous_mode,
                        'suppressed_mode': suppressed_mode,
                        'penalty': penalty,
                    },
                    source='mode_router',
                )
            except Exception:
                pass

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
        Break tie between top-2 modes using the ONNX classifier.

        The ONNX model is the sole tiebreaker — no LLM fallback. When the model
        isn't available, returns None (caller falls back to the higher-scoring mode).
        """
        try:
            from services.onnx_inference_service import get_onnx_inference_service

            svc = get_onnx_inference_service()
            input_text = self._build_onnx_input(prompt_text, signals, mode_a, mode_b)

            start = time.time()
            label, confidence = svc.predict("mode-tiebreaker", input_text)
            elapsed_ms = (time.time() - start) * 1000

            if label is None:
                logger.warning(f"{LOG_PREFIX} ONNX tiebreaker unavailable — using higher score")
                return None

            selected = mode_a if label == "A" else mode_b

            # ACT subsumes RESPOND — ACT can fall back to RESPOND mid-execution,
            # but RESPOND can't escalate to ACT. On low-confidence ties between
            # these two, prefer ACT as the safer default.
            if confidence < 0.65:
                pair = {mode_a, mode_b}
                if pair == {'ACT', 'RESPOND'}:
                    logger.info(
                        f"{LOG_PREFIX} ONNX tie-break: ACT (low-confidence "
                        f"ACT/RESPOND, {confidence:.2f}) — ACT preferred "
                        f"as safe default in {elapsed_ms:.1f}ms"
                    )
                    return 'ACT'

            logger.info(
                f"{LOG_PREFIX} ONNX tie-break: {selected} ({confidence:.2f}) "
                f"in {elapsed_ms:.1f}ms"
            )
            return selected

        except Exception as e:
            logger.warning(f"{LOG_PREFIX} ONNX tiebreaker failed: {e}")
            return None

    def _build_onnx_input(
        self,
        prompt_text: str,
        signals: Dict[str, Any],
        mode_a: str,
        mode_b: str,
    ) -> str:
        """
        Build input string for ONNX tie-breaker classifier.

        Must match the exact format used during training
        (see training/data/tasks/mode_tiebreaker/__init__.py _format_input).
        """
        def _bool(v):
            return 'true' if v else 'false'

        lines = [
            f'User message: "{prompt_text}"',
            f"context_warmth: {signals['context_warmth']:.2f}",
            f"fact_count: {signals['fact_count']}",
            f"working_memory_turns: {signals['working_memory_turns']}",
            f"topic: {signals.get('topic', 'unknown')}",
            f"is_new_topic: {_bool(signals.get('is_new_topic', False))}",
            f"gist_count: {signals['gist_count']}",
            f"has_question_mark: {_bool(signals['has_question_mark'])}",
            f"interrogative_words: {_bool(signals['interrogative_words'])}",
            f"greeting_pattern: {_bool(signals['greeting_pattern'])}",
            f"explicit_feedback: {signals['explicit_feedback'] or 'null'}",
            f"information_density: {signals['information_density']:.2f}",
            f"implicit_reference: {_bool(signals['implicit_reference'])}",
            f"intent_type: {signals.get('intent_type') or 'null'}",
            f"memory_confidence: {signals.get('memory_confidence', 0.5):.2f}",
            "",
            f"A: {mode_a} — {self.MODE_DESCRIPTIONS.get(mode_a, '')}",
            f"B: {mode_b} — {self.MODE_DESCRIPTIONS.get(mode_b, '')}",
        ]
        return "\n".join(lines)

