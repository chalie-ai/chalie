"""
Feature tests for _dedup_repetitions in api.voice.

Moonshine (Whisper-family) hallucinates by repeating phrases 15–20× on
silence or noise.  _dedup_repetitions collapses any n-gram (2–20 words) that
appears more than _MAX_CONSECUTIVE_PHRASE_REPEATS times in a row down to
exactly that many occurrences.

Single-word repetitions ("no no no") are intentionally not collapsed because
they are natural emphasis in real speech.
"""

import pytest

from api.voice import _dedup_repetitions, _MAX_CONSECUTIVE_PHRASE_REPEATS


@pytest.mark.unit
class TestDedupRepetitions:
    """_dedup_repetitions collapses hallucination loops while leaving real speech alone."""

    # ── passthrough cases ──────────────────────────────────────────────────

    @pytest.mark.parametrize("text", [
        "",
        "hello",
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
        # Short to medium n-grams at various repeat depths
        ("my calendar", 10),
        ("check my calendar", 15),
        ("turn off the lights", 5),
        # Long n-grams (TKT-436 widened _MAX_NGRAM_WORDS to 20)
        ("please check my email and reply now", 17),
        ("remind me to call the dentist at nine am", 17),
        ("one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen", 10),
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

    def test_long_phrase_surrounded_by_normal_text(self):
        prefix = "okay"
        phrase = "set a reminder for tomorrow at nine in the morning"
        suffix = "thanks"
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
