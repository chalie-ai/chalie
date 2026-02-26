"""Extended tests for ModeRouterService — anti-oscillation, confidence, margin, tiebreaker parsing."""

import pytest
from services.mode_router_service import ModeRouterService


pytestmark = pytest.mark.unit


def _make_config():
    return {
        'base_scores': {
            'RESPOND': 0.40, 'CLARIFY': 0.30, 'ACT': 0.20,
            'ACKNOWLEDGE': 0.10, 'IGNORE': -0.50,
        },
        'weights': {},
        'tiebreaker_base_margin': 0.20,
        'tiebreaker_min_margin': 0.08,
    }


def _make_signals(**overrides):
    base = {
        'context_warmth': 0.5,
        'working_memory_turns': 2,
        'gist_count': 2,
        'fact_count': 3,
        'fact_keys': [],
        'world_state_present': True,
        'topic_confidence': 0.8,
        'is_new_topic': False,
        'session_exchange_count': 2,
        'prompt_token_count': 10,
        'has_question_mark': False,
        'interrogative_words': False,
        'greeting_pattern': False,
        'explicit_feedback': None,
        'information_density': 0.8,
        'implicit_reference': False,
        'memory_confidence': 0.5,
        'intent_complexity': 'simple',
        'intent_type': None,
        'intent_confidence': 0.0,
    }
    base.update(overrides)
    return base


# ── _apply_anti_oscillation ──────────────────────────────────────────

class TestAntiOscillation:

    def test_suppresses_act_after_act(self):
        router = ModeRouterService(_make_config())
        scores = {'RESPOND': 0.5, 'CLARIFY': 0.3, 'ACT': 0.6, 'ACKNOWLEDGE': 0.1, 'IGNORE': -0.5}
        adjusted = router._apply_anti_oscillation(scores, previous_mode='ACT')
        assert adjusted['ACT'] == pytest.approx(0.45)  # 0.6 - 0.15

    def test_boosts_respond_after_clarify(self):
        router = ModeRouterService(_make_config())
        scores = {'RESPOND': 0.5, 'CLARIFY': 0.3, 'ACT': 0.2, 'ACKNOWLEDGE': 0.1, 'IGNORE': -0.5}
        adjusted = router._apply_anti_oscillation(scores, previous_mode='CLARIFY')
        assert adjusted['RESPOND'] == pytest.approx(0.55)  # 0.5 + 0.05

    def test_no_change_for_other_modes(self):
        router = ModeRouterService(_make_config())
        scores = {'RESPOND': 0.5, 'CLARIFY': 0.3, 'ACT': 0.2, 'ACKNOWLEDGE': 0.1, 'IGNORE': -0.5}
        adjusted = router._apply_anti_oscillation(scores, previous_mode='RESPOND')
        assert adjusted == scores


# ── _calculate_confidence ────────────────────────────────────────────

class TestCalculateConfidence:

    def test_high_confidence_when_clear_winner(self):
        router = ModeRouterService(_make_config())
        confidence = router._calculate_confidence(top_score=0.8, runner_up_score=0.2)
        # (0.8 - 0.2) / 0.8 = 0.75
        assert confidence == pytest.approx(0.75)

    def test_low_confidence_when_close_scores(self):
        router = ModeRouterService(_make_config())
        confidence = router._calculate_confidence(top_score=0.51, runner_up_score=0.50)
        # (0.51 - 0.50) / 0.51 ≈ 0.0196
        assert confidence < 0.05

    def test_no_divide_by_zero_when_top_near_zero(self):
        router = ModeRouterService(_make_config())
        confidence = router._calculate_confidence(top_score=0.0001, runner_up_score=0.0)
        # denominator = max(0.0001, 0.001) = 0.001
        assert isinstance(confidence, float)


# ── _calculate_effective_margin ──────────────────────────────────────

class TestCalculateEffectiveMargin:

    def test_narrows_with_warmth(self):
        router = ModeRouterService(_make_config())
        cold = router._calculate_effective_margin(_make_signals(context_warmth=0.0))
        warm = router._calculate_effective_margin(_make_signals(context_warmth=1.0))
        assert warm < cold

    def test_widens_with_semantic_uncertainty_implicit_ref(self):
        router = ModeRouterService(_make_config())
        base = router._calculate_effective_margin(_make_signals(implicit_reference=False))
        wider = router._calculate_effective_margin(_make_signals(implicit_reference=True))
        assert wider > base

    def test_widens_with_interrogative_without_question_mark(self):
        router = ModeRouterService(_make_config())
        base = router._calculate_effective_margin(
            _make_signals(interrogative_words=False, has_question_mark=False))
        wider = router._calculate_effective_margin(
            _make_signals(interrogative_words=True, has_question_mark=False))
        assert wider > base

    def test_widens_with_low_information_density(self):
        router = ModeRouterService(_make_config())
        normal = router._calculate_effective_margin(
            _make_signals(information_density=0.8))
        wider = router._calculate_effective_margin(
            _make_signals(information_density=0.2))  # < 0.3
        assert wider > normal


# ── _extract_tiebreaker_choice ───────────────────────────────────────

class TestExtractTiebreakerChoice:

    def test_clean_json_a(self):
        router = ModeRouterService(_make_config())
        result = router._extract_tiebreaker_choice('{"choice": "A"}', 'RESPOND', 'CLARIFY')
        assert result == 'RESPOND'

    def test_clean_json_b(self):
        router = ModeRouterService(_make_config())
        result = router._extract_tiebreaker_choice('{"choice": "B"}', 'RESPOND', 'CLARIFY')
        assert result == 'CLARIFY'

    def test_json_embedded_in_text(self):
        router = ModeRouterService(_make_config())
        response = 'I think the best choice is: {"choice": "A"} because...'
        result = router._extract_tiebreaker_choice(response, 'ACT', 'RESPOND')
        assert result == 'ACT'

    def test_regex_fallback(self):
        router = ModeRouterService(_make_config())
        response = 'My answer is "choice": "B" based on context.'
        result = router._extract_tiebreaker_choice(response, 'ACT', 'RESPOND')
        assert result == 'RESPOND'

    def test_garbage_returns_none(self):
        router = ModeRouterService(_make_config())
        result = router._extract_tiebreaker_choice('completely random text here', 'ACT', 'RESPOND')
        assert result is None
