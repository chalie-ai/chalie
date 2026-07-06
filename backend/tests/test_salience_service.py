import pytest

from services.episodic_service import compute_salience

pytestmark = pytest.mark.unit


class TestComputeSalience:
    def test_maximal_inputs_yield_10(self) -> None:
        assert compute_salience(has_open_loop=True, novelty=1.0) == 10

    def test_minimal_inputs_yield_1(self) -> None:
        assert compute_salience(has_open_loop=False, novelty=0.0) == 1

    def test_novelty_boosts_score(self) -> None:
        low = compute_salience(has_open_loop=False, novelty=0.0)
        high = compute_salience(has_open_loop=False, novelty=1.0)
        assert high > low

    def test_open_loop_boosts_score(self) -> None:
        without = compute_salience(has_open_loop=False, novelty=0.0)
        with_ = compute_salience(has_open_loop=True, novelty=0.0)
        assert with_ > without

    def test_result_is_always_int_between_1_and_10(self) -> None:
        for o in (False, True):
            for n in (0.0, 0.5, 1.0):
                s = compute_salience(has_open_loop=o, novelty=n)
                assert isinstance(s, int)
                assert 1 <= s <= 10
