import pytest

from api.voice import _dedup_repetitions, _MAX_CONSECUTIVE_PHRASE_REPEATS


@pytest.mark.unit
class TestDedupRepetitions:
    # ── passthrough ────────────────────────────────────────────────────────

    def test_empty_string_returns_empty(self):
        assert _dedup_repetitions("") == ""

    def test_single_word_unchanged(self):
        assert _dedup_repetitions("hello") == "hello"

    def test_two_words_unchanged(self):
        assert _dedup_repetitions("hello world") == "hello world"

    def test_normal_sentence_unchanged(self):
        text = "Check my calendar for tomorrow please"
        assert _dedup_repetitions(text) == text

    # ── hallucination collapse ─────────────────────────────────────────────

    def test_3word_phrase_repeated_15_times_collapsed(self):
        phrase = "check my calendar"
        repeated = " ".join([phrase] * 15)
        result = _dedup_repetitions(repeated)
        words = result.split()
        # Should contain exactly _MAX_CONSECUTIVE_PHRASE_REPEATS copies.
        expected_words = ("check my calendar " * _MAX_CONSECUTIVE_PHRASE_REPEATS).split()
        assert words == expected_words

    def test_2word_phrase_repeated_10_times_collapsed(self):
        phrase = "my calendar"
        repeated = " ".join([phrase] * 10)
        result = _dedup_repetitions(repeated)
        words = result.split()
        expected_words = ("my calendar " * _MAX_CONSECUTIVE_PHRASE_REPEATS).split()
        assert words == expected_words

    def test_4word_phrase_repeated_5_times_collapsed(self):
        phrase = "turn off the lights"
        repeated = " ".join([phrase] * 5)
        result = _dedup_repetitions(repeated)
        words = result.split()
        expected_words = (phrase + " ") * _MAX_CONSECUTIVE_PHRASE_REPEATS
        assert words == expected_words.split()

    # ── natural speech — must NOT be touched ──────────────────────────────

    def test_single_word_triple_not_touched(self):
        # "no no no" is natural emphasis — 1-word n-grams are not deduplicated.
        text = "no no no"
        assert _dedup_repetitions(text) == text

    def test_single_word_double_not_touched(self):
        assert _dedup_repetitions("yes yes") == "yes yes"

    def test_allowed_repeats_not_touched(self):
        # Exactly _MAX_CONSECUTIVE_PHRASE_REPEATS copies should survive.
        phrase = "check my calendar"
        text = " ".join([phrase] * _MAX_CONSECUTIVE_PHRASE_REPEATS)
        assert _dedup_repetitions(text) == text

    # ── mixed content ──────────────────────────────────────────────────────

    def test_repeated_phrase_surrounded_by_normal_text(self):
        prefix = "please"
        phrase = "check my calendar"
        suffix = "for me"
        repeated = " ".join([phrase] * 12)
        text = f"{prefix} {repeated} {suffix}"
        result = _dedup_repetitions(text)
        expected_phrase_block = " ".join([phrase] * _MAX_CONSECUTIVE_PHRASE_REPEATS)
        assert result == f"{prefix} {expected_phrase_block} {suffix}"

    def test_legitimate_near_repetitions_preserved(self):
        # "the cat sat" and "the cat ran" share a prefix but are different phrases.
        text = "the cat sat the cat ran"
        assert _dedup_repetitions(text) == text

    # ── case insensitivity ─────────────────────────────────────────────────

    def test_case_insensitive_matching_preserves_original_case(self):
        phrase = "Check My Calendar"
        repeated = " ".join([phrase] * 10)
        result = _dedup_repetitions(repeated)
        # The output should have the original casing, not lowercased.
        assert result == " ".join([phrase] * _MAX_CONSECUTIVE_PHRASE_REPEATS)

    def test_mixed_case_collapse(self):
        # Same phrase in different casing across repetitions.
        words_list = ["check my calendar"] * 5 + ["CHECK MY CALENDAR"] * 5
        repeated = " ".join(words_list)
        result = _dedup_repetitions(repeated)
        result_words = result.split()
        # After collapse the total word count should be
        # _MAX_CONSECUTIVE_PHRASE_REPEATS * 3 words (one copy per allowed repeat).
        assert len(result_words) == _MAX_CONSECUTIVE_PHRASE_REPEATS * 3
