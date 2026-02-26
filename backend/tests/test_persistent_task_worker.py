"""Tests for persistent_task_worker — constants, jitter range, surfacing logic."""

import pytest

from workers.persistent_task_worker import (
    BASE_CYCLE_SECONDS,
    JITTER_FACTOR,
    FIRST_SURFACE_CYCLE,
    COVERAGE_JUMP_THRESHOLD,
)


pytestmark = pytest.mark.unit


# ── Constants ────────────────────────────────────────────────────────

class TestConstants:

    def test_base_cycle_seconds(self):
        assert BASE_CYCLE_SECONDS == 1800

    def test_jitter_factor(self):
        assert JITTER_FACTOR == 0.3

    def test_first_surface_cycle(self):
        assert FIRST_SURFACE_CYCLE == 2

    def test_coverage_jump_threshold(self):
        assert COVERAGE_JUMP_THRESHOLD == 0.15


# ── Jitter range ─────────────────────────────────────────────────────

class TestJitterRange:

    def test_minimum_sleep_time(self):
        """Minimum sleep = BASE * (1 - JITTER) = 1800 * 0.7 = 1260."""
        minimum = BASE_CYCLE_SECONDS * (1 - JITTER_FACTOR)
        assert minimum == pytest.approx(1260.0)

    def test_maximum_sleep_time(self):
        """Maximum sleep = BASE * (1 + JITTER) = 1800 * 1.3 = 2340."""
        maximum = BASE_CYCLE_SECONDS * (1 + JITTER_FACTOR)
        assert maximum == pytest.approx(2340.0)


# ── Surfacing logic ──────────────────────────────────────────────────

class TestSurfacingLogic:
    """
    should_surface = (
        cycles_completed == FIRST_SURFACE_CYCLE
        or coverage_jump > COVERAGE_JUMP_THRESHOLD
    )
    """

    @staticmethod
    def _should_surface(cycles_completed, coverage_jump):
        return (
            (cycles_completed == FIRST_SURFACE_CYCLE) or
            (coverage_jump > COVERAGE_JUMP_THRESHOLD)
        )

    def test_surfaces_at_first_surface_cycle(self):
        assert self._should_surface(cycles_completed=2, coverage_jump=0.0) is True

    def test_surfaces_when_coverage_jump_exceeds_threshold(self):
        assert self._should_surface(cycles_completed=5, coverage_jump=0.20) is True

    def test_no_surface_when_neither_condition_met(self):
        assert self._should_surface(cycles_completed=3, coverage_jump=0.05) is False

    def test_no_surface_on_first_cycle(self):
        assert self._should_surface(cycles_completed=1, coverage_jump=0.0) is False
