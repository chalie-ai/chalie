"""
Single-word repetitions ("no no no") are intentionally not collapsed because
they are natural emphasis in real speech.
"""

import pytest

from api.voice import _dedup_repetitions, _MAX_CONSECUTIVE_PHRASE_REPEATS


@pytest.mark.unit
class TestDedupRepetitions:

    # ── passthrough cases ──────────────────────────────────────────────────

    @pytest.mark.parametrize("text", [
        "hello world",
        "Check my calendar for tomorrow please",
    ])
    def test_non_repeated_text_is_unchanged(self, text):
        assert _dedup_repetitions(text) == text

    @pytest.mark.parametrize("text", [
        # 1-word n-grams are never collapsed — these are natural emphasis
        "no no no",
        "yes yes",
    ])
    def test_single_word_repetitions_are_preserved(self, text):
        assert _dedup_repetitions(text) == text

    def test_exactly_max_allowed_repeats_are_preserved(self):
        # Exactly _MAX_CONSECUTIVE_PHRASE_REPEATS copies must survive intact.
        phrase = "check my calendar"
        text = " ".join([phrase] * _MAX_CONSECUTIVE_PHRASE_REPEATS)
        assert _dedup_repetitions(text) == text

    # ── hallucination collapse ─────────────────────────────────────────────

    @pytest.mark.parametrize("phrase,count", [
        ("check my calendar", 15),
        ("turn off the lights", 5),
        ("please check my email and reply now", 17),
    ])
    def test_repeated_phrase_is_collapsed_to_max_allowed(self, phrase, count):
        repeated = " ".join([phrase] * count)
        result = _dedup_repetitions(repeated)
        expected = " ".join([phrase] * _MAX_CONSECUTIVE_PHRASE_REPEATS)
        assert result == expected

    def test_repeated_phrase_surrounded_by_normal_text(self):
        prefix = "please"
        phrase = "check my calendar"
        suffix = "for me"
        repeated = " ".join([phrase] * 12)
        text = f"{prefix} {repeated} {suffix}"
        result = _dedup_repetitions(text)
        expected_block = " ".join([phrase] * _MAX_CONSECUTIVE_PHRASE_REPEATS)
        assert result == f"{prefix} {expected_block} {suffix}"


    # ── natural speech boundaries ──────────────────────────────────────────

    def test_near_repetitions_with_differing_words_are_preserved(self):
        # "the cat sat" and "the cat ran" share a prefix but are different phrases.
        text = "the cat sat the cat ran"
        assert _dedup_repetitions(text) == text

    # ── case handling ──────────────────────────────────────────────────────

    def test_case_insensitive_collapse_preserves_original_casing(self):
        phrase = "Check My Calendar"
        repeated = " ".join([phrase] * 10)
        result = _dedup_repetitions(repeated)
        assert result == " ".join([phrase] * _MAX_CONSECUTIVE_PHRASE_REPEATS)
