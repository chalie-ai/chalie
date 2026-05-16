"""
Feature tests for the STT post-processing pure functions in api.voice.

Covers:
  * _strip_fillers — removes spoken filler words without damaging real words
  * _fix_contractions — restores apostrophes dropped by Moonshine

Both functions are pure and have no IO or collaborators, so they are
tested directly with real input/output assertions under @pytest.mark.unit.
"""

import pytest

from api.voice import _strip_fillers, _fix_contractions


@pytest.mark.unit
class TestStripFillers:
    """_strip_fillers removes spoken fillers and leaves real speech untouched."""

    @pytest.mark.parametrize("filler,expected", [
        # Leading filler stripped
        ("um check my calendar", "check my calendar"),
        ("uh right", "right"),
        ("ah yes", "yes"),
        ("erm continue", "continue"),
        # Mid-sentence fillers stripped, no double spaces left
        ("check um my uh calendar", "check my calendar"),
        ("yes um uh", "yes"),
        # Repeated/elongated filler forms stripped
        ("ummm", ""),
        ("uhhh please", "please"),
        # Case insensitive
        ("UM check", "check"),
    ])
    def test_filler_words_are_removed(self, filler, expected):
        assert _strip_fillers(filler) == expected

    @pytest.mark.parametrize("text", [
        "umbrella",   # "um" prefix but not a standalone filler
        "error",      # "er" prefix but not a standalone filler
        "ahead",      # "ah" prefix but not a standalone filler
        "check my calendar for tomorrow",
    ])
    def test_real_words_are_preserved(self, text):
        assert _strip_fillers(text) == text

    def test_empty_string_returns_empty(self):
        assert _strip_fillers("") == ""

    def test_no_double_spaces_after_removal(self):
        result = _strip_fillers("check um uh my calendar")
        assert "  " not in result


@pytest.mark.unit
class TestFixContractions:
    """_fix_contractions restores apostrophes that Moonshine drops."""

    @pytest.mark.parametrize("raw,expected", [
        ("I didnt say that", "I didn't say that"),
        ("cant wont dont", "can't won't don't"),
        ("you shouldnt do that", "you shouldn't do that"),
        ("I couldnt find it", "I couldn't find it"),
        ("im here", "I'm here"),
        ("ive seen it", "I've seen it"),
        ("youre right", "you're right"),
        ("theyre coming", "they're coming"),
        ("whos there", "who's there"),
        ("thats great", "that's great"),
    ])
    def test_contraction_is_restored(self, raw, expected):
        assert _fix_contractions(raw) == expected

    @pytest.mark.parametrize("raw,expected", [
        # Uppercase token → fully uppercased result
        ("DIDNT", "DIDN'T"),
        # Title-case token → title-case result
        ("Didnt", "Didn't"),
    ])
    def test_original_casing_is_preserved(self, raw, expected):
        assert _fix_contractions(raw) == expected

    @pytest.mark.parametrize("text", [
        # Ambiguous words whose bare form is independently valid must not change
        "they were here",  # "were" is past tense, not "we're"
        "well done",       # "well" is an adverb, not "we'll"
        "its colour",      # "its" as possessive, not "it's"
        "I checked the calendar",
    ])
    def test_ambiguous_words_are_not_rewritten(self, text):
        assert _fix_contractions(text) == text

    def test_empty_string_returns_empty(self):
        assert _fix_contractions("") == ""
