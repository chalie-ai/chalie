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


# ── ACT score path ───────────────────────────────────────────────────

class TestActModeScoring:

    def test_act_moderate_warmth_question_boost(self):
        """ACT gets +question_moderate_context when question asked with moderate warmth (0.3-0.7)."""
        router = ModeRouterService(_make_config())
        # warmth in [0.3, 0.7] and is_question=True → act +0.20 (default weight)
        signals = _make_signals(context_warmth=0.45, has_question_mark=True, fact_count=5)
        result = router.route(signals, "How do I fix this?", skip_tiebreaker=True)
        assert result['scores']['ACT'] > 0.20

    def test_act_implicit_reference_boost(self):
        """ACT gets +0.15 when implicit reference detected ('you said', 'last time', etc.)."""
        router = ModeRouterService(_make_config())
        r_with = router.route(_make_signals(implicit_reference=True, context_warmth=0.5),
                              "you said earlier", skip_tiebreaker=True)
        r_without = router.route(_make_signals(implicit_reference=False, context_warmth=0.5),
                                 "something else", skip_tiebreaker=True)
        assert r_with['scores']['ACT'] > r_without['scores']['ACT']

    def test_act_very_cold_penalty(self):
        """ACT is penalized when warmth < 0.15 (user is a stranger — don't take actions)."""
        router = ModeRouterService(_make_config())
        signals = _make_signals(context_warmth=0.05)
        result = router.route(signals, "do something", skip_tiebreaker=True)
        assert result['scores']['ACT'] < 0.20  # below base due to penalty

    def test_act_warm_with_dense_facts_suppressed(self):
        """ACT suppressed when warm context AND high fact density — RESPOND should dominate."""
        router = ModeRouterService(_make_config())
        signals = _make_signals(context_warmth=0.8, fact_count=8)  # fact_density=0.8 > 0.5
        result = router.route(signals, "Tell me more", skip_tiebreaker=True)
        assert result['scores']['RESPOND'] > result['scores']['ACT']

    def test_act_memory_confidence_very_low_boosts_act_on_question(self):
        """Very low memory confidence (<0.15) + question → ACT boosted to favour external retrieval."""
        router = ModeRouterService(_make_config())
        sigs_low = {**_make_signals(context_warmth=0.5, has_question_mark=True), 'memory_confidence': 0.10}
        sigs_high = {**_make_signals(context_warmth=0.5, has_question_mark=True), 'memory_confidence': 0.80}
        r_low = router.route(sigs_low, "What is X?", skip_tiebreaker=True)
        r_high = router.route(sigs_high, "What is X?", skip_tiebreaker=True)
        assert r_low['scores']['ACT'] > r_high['scores']['ACT']

    def test_act_memory_confidence_mild_low_boosts_act_on_question(self):
        """Memory confidence in [0.15, 0.30) + question → ACT gets a smaller boost."""
        router = ModeRouterService(_make_config())
        sigs_mild = {**_make_signals(context_warmth=0.5, has_question_mark=True), 'memory_confidence': 0.20}
        sigs_high = {**_make_signals(context_warmth=0.5, has_question_mark=True), 'memory_confidence': 0.80}
        r_mild = router.route(sigs_mild, "Where is X?", skip_tiebreaker=True)
        r_high = router.route(sigs_high, "Where is X?", skip_tiebreaker=True)
        assert r_mild['scores']['ACT'] > r_high['scores']['ACT']

    def test_act_memory_confidence_boost_requires_question(self):
        """Memory confidence boost does not apply without a question signal."""
        router = ModeRouterService(_make_config())
        sigs_with_q = {**_make_signals(context_warmth=0.5, has_question_mark=True), 'memory_confidence': 0.05}
        sigs_no_q = {**_make_signals(context_warmth=0.5, has_question_mark=False,
                                     interrogative_words=False), 'memory_confidence': 0.05}
        r_with_q = router.route(sigs_with_q, "What?", skip_tiebreaker=True)
        r_no_q = router.route(sigs_no_q, "Something", skip_tiebreaker=True)
        assert r_with_q['scores']['ACT'] > r_no_q['scores']['ACT']


# ── IGNORE mode ──────────────────────────────────────────────────────

class TestIgnoreMode:

    def test_ignore_empty_input_wins_low_warmth_no_memory(self):
        """
        IGNORE wins for zero-token input when warmth is in the cold zone (0.20-0.30).

        At warmth=0.25: IGNORE=0.50, CLARIFY=0.49 (cold_boost=(1-0.25)*0.25=0.19).
        At warmth=0.0:  CLARIFY=0.55 (cold_boost=0.25 dominates), so CLARIFY beats IGNORE.
        The sweet spot is 0.20 < warmth < 0.30 where cold_boost is weak enough.
        """
        router = ModeRouterService(_make_config())
        signals = _make_signals(
            prompt_token_count=0,
            context_warmth=0.25,  # cold (< 0.30), but cold_boost weak enough that IGNORE wins
            fact_count=0,
            gist_count=0,
            has_question_mark=False,
            interrogative_words=False,
        )
        result = router.route(signals, "", skip_tiebreaker=True)
        assert result['mode'] == 'IGNORE'

    def test_ignore_score_boosted_on_empty_input(self):
        """Empty input should boost IGNORE score above its base -0.50."""
        router = ModeRouterService(_make_config())
        signals_empty = _make_signals(prompt_token_count=0)
        signals_content = _make_signals(prompt_token_count=10)
        r_empty = router.route(signals_empty, "", skip_tiebreaker=True)
        r_content = router.route(signals_content, "Hello", skip_tiebreaker=True)
        assert r_empty['scores']['IGNORE'] > r_content['scores']['IGNORE']

    def test_ignore_score_deeply_negative_with_content(self):
        """IGNORE stays at base -0.50 with actual non-empty input."""
        router = ModeRouterService(_make_config())
        signals = _make_signals(prompt_token_count=10, context_warmth=0.5)
        result = router.route(signals, "Hello there", skip_tiebreaker=True)
        assert result['scores']['IGNORE'] < 0

    def test_ignore_is_lowest_mode_with_content(self):
        """IGNORE should be the lowest-scoring mode for any non-empty input."""
        router = ModeRouterService(_make_config())
        signals = _make_signals(prompt_token_count=5)
        result = router.route(signals, "Hello", skip_tiebreaker=True)
        ignore_score = result['scores']['IGNORE']
        other_scores = [v for k, v in result['scores'].items() if k != 'IGNORE']
        assert all(s > ignore_score for s in other_scores)


# ── ACKNOWLEDGE feedback ─────────────────────────────────────────────

class TestAcknowledgeFeedback:

    def test_acknowledge_greeting_boosts_score(self):
        """Greeting should substantially boost ACKNOWLEDGE score."""
        router = ModeRouterService(_make_config())
        r_greet = router.route(_make_signals(greeting_pattern=True, context_warmth=0.0),
                               "Hey!", skip_tiebreaker=True)
        r_no_greet = router.route(_make_signals(greeting_pattern=False, context_warmth=0.0),
                                  "Something", skip_tiebreaker=True)
        assert r_greet['scores']['ACKNOWLEDGE'] > r_no_greet['scores']['ACKNOWLEDGE']

    def test_acknowledge_positive_feedback_boosts_score(self):
        """Positive explicit feedback should boost ACKNOWLEDGE score."""
        router = ModeRouterService(_make_config())
        r_pos = router.route(_make_signals(explicit_feedback='positive', context_warmth=0.0),
                             "Thanks!", skip_tiebreaker=True)
        r_none = router.route(_make_signals(explicit_feedback=None, context_warmth=0.0),
                              "OK", skip_tiebreaker=True)
        assert r_pos['scores']['ACKNOWLEDGE'] > r_none['scores']['ACKNOWLEDGE']

    def test_acknowledge_question_reduces_score(self):
        """Question mark should penalize ACKNOWLEDGE (questions deserve answers, not acknowledgements)."""
        router = ModeRouterService(_make_config())
        r_q = router.route(_make_signals(greeting_pattern=True, has_question_mark=True),
                           "Hi, how are you?", skip_tiebreaker=True)
        r_nq = router.route(_make_signals(greeting_pattern=True, has_question_mark=False),
                            "Hi there", skip_tiebreaker=True)
        assert r_q['scores']['ACKNOWLEDGE'] < r_nq['scores']['ACKNOWLEDGE']

    def test_acknowledge_wins_for_greeting_with_cold_context(self):
        """A greeting with no prior context should select ACKNOWLEDGE mode."""
        router = ModeRouterService(_make_config())
        signals = _make_signals(
            greeting_pattern=True,
            context_warmth=0.0,
            has_question_mark=False,
            fact_count=0,
            gist_count=0,
        )
        result = router.route(signals, "Hey!", skip_tiebreaker=True)
        assert result['mode'] == 'ACKNOWLEDGE'


# ── Hysteresis (low-confidence streak widening) ──────────────────────

class TestHysteresisWidening:

    def test_no_streak_by_default(self):
        """A fresh router has no low-confidence streak."""
        router = ModeRouterService(_make_config())
        assert not router._is_low_confidence_streak('any_topic')

    def test_two_low_confidence_decisions_not_enough(self):
        """Two low-confidence decisions do not trigger the streak — needs three."""
        router = ModeRouterService(_make_config())
        router._track_confidence('test', 0.05)
        router._track_confidence('test', 0.08)
        assert not router._is_low_confidence_streak('test')

    def test_three_low_confidence_decisions_triggers_streak(self):
        """Three consecutive low-confidence decisions trigger hysteresis."""
        router = ModeRouterService(_make_config())
        router._track_confidence('test', 0.05)
        router._track_confidence('test', 0.08)
        router._track_confidence('test', 0.12)
        assert router._is_low_confidence_streak('test')

    def test_high_confidence_decision_resets_streak(self):
        """One high-confidence decision after two low ones prevents the streak."""
        router = ModeRouterService(_make_config())
        router._track_confidence('test', 0.05)
        router._track_confidence('test', 0.05)
        router._track_confidence('test', 0.80)  # high confidence resets window
        assert not router._is_low_confidence_streak('test')

    def test_hysteresis_widens_effective_margin_by_0_05(self):
        """
        A low-confidence streak adds exactly 0.05 to effective_margin in route() output.

        Inside route(), _track_confidence runs BEFORE _is_low_confidence_streak, so
        we plant 2 values and rely on the route call's own confidence reading to
        complete the streak as the 3rd entry. Signals are tuned to produce confidence≈0
        (RESPOND and ACT nearly tied) so the route call contributes a low value.

        After route():
        - Fresh router history: [0.0]          → len < 3, no streak, margin unwidened
        - Streaked router history: [0.05, 0.05, 0.0] → all < 0.15, streak active → +0.05
        """
        topic = 'hysteresis_test'
        # warmth=0.5, question, interrog, fact=0, gist=0, mem_conf=0.20
        # → RESPOND = 0.40 + 0.10 + 0 + 0 + 0.15 = 0.65
        # → ACT     = 0.20 + 0.20 + 0.15 + 0.10  = 0.65 (nearly tied → confidence ≈ 0)
        tied_signals = {
            **_make_signals(
                context_warmth=0.5,
                has_question_mark=True,
                interrogative_words=True,
                fact_count=0,
                gist_count=0,
            ),
            'memory_confidence': 0.20,
            'topic': topic,
        }

        # Fresh router — route adds first confidence reading (≈0), len=1, no streak
        fresh_router = ModeRouterService(_make_config())
        r_fresh = fresh_router.route(tied_signals, "Where is X?", skip_tiebreaker=True)

        # Streaked router — 2 pre-planted readings, route adds the 3rd → streak triggers
        streaked_router = ModeRouterService(_make_config())
        streaked_router._track_confidence(topic, 0.05)
        streaked_router._track_confidence(topic, 0.05)
        r_streaked = streaked_router.route(tied_signals, "Where is X?", skip_tiebreaker=True)

        assert r_streaked['effective_margin'] == pytest.approx(
            r_fresh['effective_margin'] + 0.05
        )
