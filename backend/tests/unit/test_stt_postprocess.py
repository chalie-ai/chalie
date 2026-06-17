# Both functions are pure and have no IO or collaborators, so they are
# tested directly with real input/output assertions under @pytest.mark.unit.

import pytest

from api.voice import _strip_fillers, _fix_contractions


@pytest.mark.unit
class TestStripFillers:
    @pytest.mark.parametrize("filler,expected", [
        # Leading filler stripped
        ("um check my calendar", "check my calendar"),
        ("uh right", "right"),
        # Mid-sentence fillers stripped, no double spaces left
        ("check um my uh calendar", "check my calendar"),
        # Repeated/elongated filler forms stripped
        ("ummm", ""),
        # Case insensitive
        ("UM check", "check"),
    ])
    def test_filler_words_are_removed(self, filler, expected):
        assert _strip_fillers(filler) == expected

    @pytest.mark.parametrize("text", [
        "umbrella",   # "um" prefix but not a standalone filler
        "check my calendar for tomorrow",
    ])
    def test_real_words_are_preserved(self, text):
        assert _strip_fillers(text) == text

    def test_no_double_spaces_after_removal(self):
        result = _strip_fillers("check um uh my calendar")
        assert "  " not in result


@pytest.mark.unit
class TestFixContractions:
    @pytest.mark.parametrize("raw,expected", [
        ("I didnt say that", "I didn't say that"),
        ("cant wont dont", "can't won't don't"),
        ("im here", "I'm here"),
        ("youre right", "you're right"),
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
        "its colour",      # "its" as possessive, not "it's"
    ])
    def test_ambiguous_words_are_not_rewritten(self, text):
        assert _fix_contractions(text) == text

