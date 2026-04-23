"""Feature tests for PersonalityService.

All tests are @pytest.mark.unit — real SQLite via the ``db`` fixture,
real voices.jsonl corpus, zero mocks.

Service under test: services.personality.personality_service
"""

import itertools
import json
import os

import pytest

pytestmark = pytest.mark.unit

# ── Shared corpus helper ───────────────────────────────────────────────────────

_VOICES_PATH = os.path.join(
    os.path.dirname(__file__),
    '..', 'services', 'personality', 'voices.jsonl',
)


def _load_corpus() -> dict:
    """Read voices.jsonl once and return a dict keyed by tuple."""
    index = {}
    with open(_VOICES_PATH, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            index[tuple(obj['tuple'])] = obj['voice']
    return index


# ── TestVoiceLookup ────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestVoiceLookup:
    """get_voice() returns the exact corpus string for known tuples."""

    def test_loader_returns_correct_voice_for_neutral_tuple(self):
        """get_voice((0,0,0,0,0)) matches the corpus entry for that tuple."""
        corpus = _load_corpus()
        expected = corpus[(0, 0, 0, 0, 0)]

        from services.personality.personality_service import get_voice
        result = get_voice((0, 0, 0, 0, 0))

        assert result == expected

    def test_loader_returns_correct_voice_for_all_cool_tuple(self):
        """get_voice((-2,-2,-2,-2,-2)) matches the corpus entry for that tuple."""
        corpus = _load_corpus()
        expected = corpus[(-2, -2, -2, -2, -2)]

        from services.personality.personality_service import get_voice
        result = get_voice((-2, -2, -2, -2, -2))

        assert result == expected

    def test_loader_returns_correct_voice_for_all_warm_tuple(self):
        """get_voice((2,2,2,2,2)) matches the corpus entry for that tuple."""
        corpus = _load_corpus()
        expected = corpus[(2, 2, 2, 2, 2)]

        from services.personality.personality_service import get_voice
        result = get_voice((2, 2, 2, 2, 2))

        assert result == expected

    def test_loader_returns_correct_voice_for_mixed_tuple(self):
        """get_voice((1,-1,0,2,-2)) matches the corpus entry for that tuple."""
        corpus = _load_corpus()
        expected = corpus[(1, -1, 0, 2, -2)]

        from services.personality.personality_service import get_voice
        result = get_voice((1, -1, 0, 2, -2))

        assert result == expected

    def test_loader_all_3125_tuples_present(self):
        """Every valid 5^5 tuple is covered and returns a non-empty voice (≥60 chars)."""
        from services.personality.personality_service import get_voice

        for tup in itertools.product((-2, -1, 0, 1, 2), repeat=5):
            voice = get_voice(tup)
            assert isinstance(voice, str), f"Tuple {tup}: expected str, got {type(voice)}"
            assert len(voice) >= 60, (
                f"Tuple {tup}: voice too short ({len(voice)} chars) — looks like fallback: {voice!r}"
            )


# ── TestCurrentTupleDefaults ───────────────────────────────────────────────────


@pytest.mark.unit
class TestCurrentTupleDefaults:
    """get_current_tuple() and get_current_voice() with fresh DB (no setting stored)."""

    def test_get_current_tuple_defaults_to_neutral_when_unset(self, db):
        """With no 'personality' setting, get_current_tuple() returns (0,0,0,0,0)."""
        from services.personality.personality_service import (
            NEUTRAL_TUPLE,
            get_current_tuple,
        )

        result = get_current_tuple()
        assert result == NEUTRAL_TUPLE

    def test_get_current_voice_defaults_to_neutral_voice_when_unset(self, db):
        """With no 'personality' setting, get_current_voice() returns the neutral voice."""
        corpus = _load_corpus()
        expected_neutral_voice = corpus[(0, 0, 0, 0, 0)]

        from services.personality.personality_service import get_current_voice

        result = get_current_voice()
        assert result == expected_neutral_voice


# ── TestSetCurrentTuplePersistence ────────────────────────────────────────────


@pytest.mark.unit
class TestSetCurrentTuplePersistence:
    """set_current_tuple() persists to DB and subsequent reads reflect the change."""

    def test_set_current_tuple_persists_and_returns_voice(self, db):
        """set_current_tuple returns the correct voice; subsequent get_current_tuple returns the same tuple."""
        corpus = _load_corpus()
        target = (2, 1, 0, -1, -2)
        expected_voice = corpus[target]

        from services.personality.personality_service import (
            get_current_tuple,
            get_current_voice,
            set_current_tuple,
        )

        returned_voice = set_current_tuple(target)
        assert returned_voice == expected_voice, (
            "set_current_tuple() did not return the expected voice paragraph"
        )

        persisted_tuple = get_current_tuple()
        assert persisted_tuple == target, (
            f"get_current_tuple() returned {persisted_tuple!r} after set to {target!r}"
        )

        current_voice = get_current_voice()
        assert current_voice == expected_voice, (
            "get_current_voice() returned a different voice than what set_current_tuple returned"
        )


# ── TestSetCurrentTupleValidation ─────────────────────────────────────────────


@pytest.mark.unit
class TestSetCurrentTupleValidation:
    """set_current_tuple() raises ValueError for invalid inputs."""

    def test_set_current_tuple_rejects_positive_out_of_range(self, db):
        """Step value +3 is out of range — must raise ValueError."""
        from services.personality.personality_service import set_current_tuple

        with pytest.raises(ValueError):
            set_current_tuple((3, 0, 0, 0, 0))

    def test_set_current_tuple_rejects_negative_out_of_range(self, db):
        """Step value -3 is out of range — must raise ValueError."""
        from services.personality.personality_service import set_current_tuple

        with pytest.raises(ValueError):
            set_current_tuple((-3, 0, 0, 0, 0))

    def test_set_current_tuple_rejects_last_position_out_of_range(self, db):
        """Any position out of range raises ValueError, including the last."""
        from services.personality.personality_service import set_current_tuple

        with pytest.raises(ValueError):
            set_current_tuple((0, 0, 0, 0, 99))

    def test_set_current_tuple_rejects_too_short(self, db):
        """Tuple of length < 5 must raise ValueError."""
        from services.personality.personality_service import set_current_tuple

        with pytest.raises(ValueError):
            set_current_tuple((0, 0, 0, 0))

    def test_set_current_tuple_rejects_too_long(self, db):
        """Tuple of length > 5 must raise ValueError."""
        from services.personality.personality_service import set_current_tuple

        with pytest.raises(ValueError):
            set_current_tuple((0, 0, 0, 0, 0, 0))

    def test_set_current_tuple_rejects_empty(self, db):
        """Empty tuple must raise ValueError."""
        from services.personality.personality_service import set_current_tuple

        with pytest.raises(ValueError):
            set_current_tuple(())

    def test_set_current_tuple_rejects_bool_masquerading_as_int(self, db):
        """Booleans satisfy ``isinstance(x, int)`` in Python — reject them explicitly."""
        from services.personality.personality_service import set_current_tuple

        with pytest.raises(ValueError):
            set_current_tuple((True, False, 0, 0, 0))
