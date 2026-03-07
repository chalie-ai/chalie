"""Tests for digest_worker — calculate_context_warmth, NLP signal patterns, social triage."""

import pytest
from unittest.mock import MagicMock

from workers.digest_worker import calculate_context_warmth, _handle_social_triage, _is_innate_skill_only
from services.cognitive_triage_service import TriageResult
from services.mode_router_service import (
    GREETING_PATTERNS,
    INTERROGATIVE_WORDS,
    IMPLICIT_REFERENCE,
    POSITIVE_FEEDBACK,
    NEGATIVE_FEEDBACK,
)


pytestmark = pytest.mark.unit


# ── calculate_context_warmth ─────────────────────────────────────────

class TestCalculateContextWarmth:
    """
    warmth = (wm_score + gist_score + world_score) / 3
    wm_score   = min(working_memory_len / 4, 1.0)
    gist_score = min(real_gist_count / 5, 1.0)   # cold_start excluded
    world_score = 1.0 if world_state_nonempty else 0.0
    """

    def test_all_zeros_returns_zero(self):
        assert calculate_context_warmth(0, [], False) == 0.0

    def test_all_maxed_returns_one(self):
        gists = [{'type': 'observation'}] * 6  # 6 real gists, caps at 1.0
        result = calculate_context_warmth(8, gists, True)
        assert result == pytest.approx(1.0)

    def test_cold_start_gists_excluded_from_count(self):
        gists = [
            {'type': 'cold_start'},
            {'type': 'cold_start'},
            {'type': 'observation'},
        ]
        # real_gist_count = 1 → gist_score = 0.2
        # wm_score = 0, world_score = 0
        result = calculate_context_warmth(0, gists, False)
        expected = (0.0 + 0.2 + 0.0) / 3
        assert result == pytest.approx(expected)

    def test_wm_caps_at_one(self):
        # 8 turns → min(8/4, 1.0) = 1.0
        result = calculate_context_warmth(8, [], False)
        expected = (1.0 + 0.0 + 0.0) / 3
        assert result == pytest.approx(expected)

    def test_world_state_true_contributes_one_third(self):
        result = calculate_context_warmth(0, [], True)
        expected = (0.0 + 0.0 + 1.0) / 3
        assert result == pytest.approx(expected)

    def test_world_state_false_contributes_zero(self):
        result = calculate_context_warmth(0, [], False)
        assert result == 0.0

    def test_mixed_inputs(self):
        gists = [{'type': 'observation'}, {'type': 'observation'}]
        # wm=2 → 0.5, gist=2 → 0.4, world=True → 1.0
        result = calculate_context_warmth(2, gists, True)
        expected = (0.5 + 0.4 + 1.0) / 3
        assert result == pytest.approx(expected, abs=0.001)


# ── NLP signal patterns ──────────────────────────────────────────────

class TestNlpSignalPatterns:

    def test_greeting_match_on_hey(self):
        assert GREETING_PATTERNS.match("hey there") is not None

    def test_greeting_match_on_good_morning(self):
        assert GREETING_PATTERNS.match("good morning") is not None

    def test_greeting_no_match_on_normal_text(self):
        assert GREETING_PATTERNS.match("the weather is nice") is None

    def test_interrogative_match_on_what(self):
        assert INTERROGATIVE_WORDS.search("what is this") is not None

    def test_interrogative_no_match_on_plain_sentence(self):
        assert INTERROGATIVE_WORDS.search("the cat sat") is None

    def test_implicit_reference_match(self):
        assert IMPLICIT_REFERENCE.search("you remember that?") is not None

    def test_implicit_reference_no_match(self):
        assert IMPLICIT_REFERENCE.search("the sky is blue") is None

    def test_question_mark_detection(self):
        assert '?' in "What time is it?"
        assert '?' not in "Tell me the time"

    def test_token_count_via_split(self):
        tokens = "hello world foo".split()
        assert len(tokens) == 3

    def test_positive_feedback_match(self):
        assert POSITIVE_FEEDBACK.search("thanks a lot") is not None

    def test_negative_feedback_match(self):
        assert NEGATIVE_FEEDBACK.search("that's not what I meant") is not None

    def test_information_density_calculation(self):
        tokens = "the the the cat".split()
        unique = len(set(t.lower() for t in tokens))
        density = unique / max(len(tokens), 1)
        # 2 unique / 4 total = 0.5
        assert density == pytest.approx(0.5)


def _make_triage(mode):
    return TriageResult(
        branch='social', mode=mode, tools=[], skills=[],
        confidence_internal=1.0, confidence_tool_need=0.0,
        freshness_risk=0.0, decision_entropy=0.0,
        reasoning='test', triage_time_ms=0.0,
        fast_filtered=False, self_eval_override=False, self_eval_reason=None,
    )


class TestHandleSocialTriage:
    """
    _handle_social_triage must only fast-exit for CANCEL/IGNORE.
    Any other mode (e.g. RESPOND) must return None so callers route
    through generate_for_mode — preventing the NoneType crash that
    caused 'No response received' for ambiguous requests like 'Schedule it'.
    """

    def test_cancel_returns_empty_response(self):
        result = _handle_social_triage(
            _make_triage('CANCEL'), 'never mind', 'topic', None,
            None, {}, None, None, None,
        )
        assert result is not None
        assert result['mode'] == 'CANCEL'
        assert result['response'] == ''

    def test_ignore_returns_empty_response(self):
        result = _handle_social_triage(
            _make_triage('IGNORE'), '', 'topic', None,
            None, {}, None, None, None,
        )
        assert result is not None
        assert result['mode'] == 'IGNORE'
        assert result['response'] == ''

    def test_respond_mode_returns_none(self):
        """RESPOND must not be handled here — callers route to generate_for_mode."""
        result = _handle_social_triage(
            _make_triage('RESPOND'), 'Schedule it', 'topic', None,
            None, {}, None, None, None,
        )
        assert result is None, (
            "RESPOND mode in social branch should return None so the dispatch "
            "condition (mode in CANCEL/IGNORE) prevents this path from being reached, "
            "not silently produce an empty response."
        )

    def test_clarify_mode_returns_none(self):
        result = _handle_social_triage(
            _make_triage('CLARIFY'), 'Schedule it', 'topic', None,
            None, {}, None, None, None,
        )
        assert result is None


# ── _is_innate_skill_only / contextual_skills dispatch ───────────

def _make_act_triage(skills):
    return TriageResult(
        branch='act', mode='ACT', tools=[], skills=skills,
        confidence_internal=0.9, confidence_tool_need=0.1,
        freshness_risk=0.0, decision_entropy=0.0,
        reasoning='test', triage_time_ms=0.0,
        fast_filtered=False, self_eval_override=False, self_eval_reason=None,
    )


class TestInnateSkillOnly:
    """
    _is_innate_skill_only gates _handle_innate_skill_dispatch.
    Ensures only contextual skills (not primitives) trigger direct dispatch,
    and that the dispatch path passes only contextual_skills to the LLM
    (preventing introspect/recall from crowding out the intended skill).
    """

    def test_schedule_skill_is_innate_only(self):
        """schedule in skills + no tools → direct dispatch path."""
        triage = _make_act_triage(['recall', 'memorize', 'introspect', 'schedule'])
        assert _is_innate_skill_only(triage) is True

    def test_primitives_only_is_not_innate_only(self):
        """Only recall/memorize/introspect → not an innate-only dispatch (no contextual skill)."""
        triage = _make_act_triage(['recall', 'memorize', 'introspect'])
        assert _is_innate_skill_only(triage) is False

    def test_empty_skills_is_not_innate_only(self):
        triage = _make_act_triage([])
        assert _is_innate_skill_only(triage) is False

    def test_external_tool_present_is_not_innate_only(self):
        """External tools present → full ACT loop, not direct dispatch."""
        triage = _make_act_triage(['schedule'])
        triage.tools = ['duckduckgo_search']
        assert _is_innate_skill_only(triage) is False

    def test_list_skill_is_innate_only(self):
        triage = _make_act_triage(['recall', 'list'])
        assert _is_innate_skill_only(triage) is True
