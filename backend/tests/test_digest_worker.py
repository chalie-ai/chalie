"""Tests for digest_worker — calculate_context_warmth and NLP signal patterns."""

import pytest

from workers.digest_worker import calculate_context_warmth
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
