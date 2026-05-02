"""
Behavioural tests for salience_service — pure salience-scoring function.

compute_salience() takes valence, arousal, has_open_loop, novelty and returns
an integer score clamped to [1, 10].  Pure function with zero collaborators —
qualifies for unit tests per tester.md.
"""

import pytest

from services.salience_service import compute_salience

pytestmark = pytest.mark.unit


class TestComputeSalience:
    def test_maximal_inputs_yield_10(self):
        result = compute_salience(
            valence=1.0, arousal=1.0, has_open_loop=True, novelty=1.0,
        )
        assert result == 10

    def test_minimal_inputs_yield_1(self):
        result = compute_salience(
            valence=0.0, arousal=0.0, has_open_loop=False, novelty=0.0,
        )
        assert result == 1

    def test_negative_valence_uses_absolute_value(self):
        result_neg = compute_salience(
            valence=-1.0, arousal=0.5, has_open_loop=False, novelty=0.0,
        )
        result_pos = compute_salience(
            valence=1.0, arousal=0.5, has_open_loop=False, novelty=0.0,
        )
        assert result_neg == result_pos

    def test_open_loop_boosts_score(self):
        without = compute_salience(
            valence=0.0, arousal=0.0, has_open_loop=False, novelty=0.0,
        )
        with_ = compute_salience(
            valence=0.0, arousal=0.0, has_open_loop=True, novelty=0.0,
        )
        assert with_ > without

    def test_result_is_always_int_between_1_and_10(self):
        for v in (-1.0, -0.5, 0.0, 0.5, 1.0):
            for a in (0.0, 0.5, 1.0):
                for o in (False, True):
                    for n in (0.0, 0.5, 1.0):
                        s = compute_salience(v, a, o, n)
                        assert isinstance(s, int)
                        assert 1 <= s <= 10
