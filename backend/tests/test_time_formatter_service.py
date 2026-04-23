"""
Unit tests for TimeFormatterService.

Pure-function tests — no DB, no MemoryStore, no IO.
All boundaries and tiers are covered as specified in the world-state-rebuild plan.
"""

import pytest
from datetime import timedelta

from services.time_formatter_service import TimeFormatterService as T
from services.time_utils import utc_now


@pytest.mark.unit
class TestDurationBoundaries:
    """All tier boundary values per spec."""

    def test_0s(self):
        assert T.duration(0) == "0s"

    def test_59s(self):
        assert T.duration(59) == "59s"

    def test_60s(self):
        # Spec: 0–60s → '{X}s' (inclusive upper bound)
        assert T.duration(60) == "60s"

    def test_61s(self):
        # Spec: 61–3599s → '{X}m'
        assert T.duration(61) == "1m"

    def test_3599s(self):
        assert T.duration(3599) == "59m"

    def test_3600s(self):
        # Spec: 3600–86399s → '{X}h {Y}m'
        assert T.duration(3600) == "1h 0m"

    def test_3601s(self):
        assert T.duration(3601) == "1h 0m"

    def test_86399s(self):
        assert T.duration(86399) == "23h 59m"

    def test_86400s(self):
        # Spec: 86400+s → '{X}d {Y}h'
        assert T.duration(86400) == "1d 0h"

    def test_86461s(self):
        assert T.duration(86461) == "1d 0h"

    def test_172800s(self):
        assert T.duration(172800) == "2d 0h"


@pytest.mark.unit
class TestDurationFractional:
    """Fractional seconds are floored at each unit boundary."""

    def test_0_9_floors_to_0s(self):
        # 0.9 → int(0.9)=0 → '0s'
        assert T.duration(0.9) == "0s"

    def test_60_5_floors_to_60s(self):
        # 60.5 → int(60.5)=60 → '60s' (still in 0–60 tier)
        assert T.duration(60.5) == "60s"

    def test_3600_99_floors_to_1h_0m(self):
        # 3600.99 → int(3600.99)=3600 → '1h 0m'
        assert T.duration(3600.99) == "1h 0m"


@pytest.mark.unit
class TestDurationNegativeClamp:
    """Negative input clamps to zero — the contract is clamp-to-0, not abs().

    duration(-1) and duration(-3600) must both return '0s'.
    """

    def test_negative_1_clamps_to_zero(self):
        assert T.duration(-1) == "0s"

    def test_negative_3600_clamps_to_zero(self):
        assert T.duration(-3600) == "0s"


@pytest.mark.unit
class TestDurationDaysTier:
    """Days tier sanity checks from spec."""

    def test_1d_0h(self):
        assert T.duration(86400) == "1d 0h"

    def test_1d_1h(self):
        # 90061 = 86400 + 3661 → 1d, (3661//3600)=1h
        assert T.duration(90061) == "1d 1h"

    def test_3d_0h(self):
        assert T.duration(259200) == "3d 0h"


@pytest.mark.unit
class TestAgoWithDatetime:
    """ago() accepts past datetime; elapsed auto-computed vs utc_now()."""

    def test_recent_past_datetime(self):
        past = utc_now() - timedelta(seconds=90)
        result = T.ago(past)
        assert result == "1m ago"

    def test_further_past_datetime(self):
        past = utc_now() - timedelta(hours=2)
        result = T.ago(past)
        assert result == "2h 0m ago"

    def test_future_datetime_clamps_to_0s_ago(self):
        future = utc_now() + timedelta(seconds=60)
        assert T.ago(future) == "0s ago"

    def test_iso_string_past(self):
        past = utc_now() - timedelta(seconds=45)
        result = T.ago(past.isoformat())
        assert result == "45s ago"


@pytest.mark.unit
class TestAgoWithSeconds:
    """ago() accepts int/float elapsed seconds."""

    def test_int_seconds(self):
        assert T.ago(45) == "45s ago"

    def test_float_seconds(self):
        assert T.ago(61.0) == "1m ago"

    def test_zero_seconds(self):
        assert T.ago(0) == "0s ago"

    def test_negative_seconds_clamps_to_0s(self):
        # The ago() implementation uses max(0.0, ...) for int/float input.
        assert T.ago(-1) == "0s ago"

    def test_hours_seconds(self):
        assert T.ago(7200) == "2h 0m ago"
