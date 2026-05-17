"""
Unit tests for TimeFormatterService.

Pure-function tests — no DB, no MemoryStore, no IO.
All boundaries and tiers are covered as specified in the world-state-rebuild plan.
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo

from services.time_formatter_service import TimeFormatterService as T
from services.time_utils import utc_now


@pytest.mark.unit
class TestDurationBoundaries:
    """Tier boundary values: seconds, minutes, hours, days."""

    def test_0s(self):
        assert T.duration(0) == "0s"

    def test_60s_upper_boundary(self):
        assert T.duration(60) == "60s"

    def test_61s_minutes_tier(self):
        assert T.duration(61) == "1m"

    def test_3600s_hours_tier(self):
        assert T.duration(3600) == "1h 0m"

    def test_86400s_days_tier(self):
        assert T.duration(86400) == "1d 0h"

    def test_negative_clamps_to_zero(self):
        assert T.duration(-1) == "0s"

    def test_fractional_seconds_floored(self):
        assert T.duration(0.9) == "0s"

    def test_days_with_hours(self):
        assert T.duration(90061) == "1d 1h"


@pytest.mark.unit
class TestAgo:
    """ago() accepts datetime, ISO string, or int/float seconds."""

    def test_past_datetime(self):
        past = utc_now() - timedelta(seconds=90)
        assert T.ago(past) == "1m ago"

    def test_future_datetime_clamps(self):
        future = utc_now() + timedelta(seconds=60)
        assert T.ago(future) == "0s ago"

    def test_iso_string(self):
        past = utc_now() - timedelta(seconds=45)
        assert T.ago(past.isoformat()) == "45s ago"

    def test_int_seconds(self):
        assert T.ago(45) == "45s ago"

    def test_zero_seconds(self):
        assert T.ago(0) == "0s ago"

    def test_negative_clamps(self):
        assert T.ago(-1) == "0s ago"


@pytest.mark.unit
class TestLocalConvertsUtcToUserTimezone:
    """local() must render UTC inputs in the user's tz, never raw UTC."""

    def _patch_tz(self, tz_name: str):
        return patch(
            "services.locale_service.get_timezone",
            return_value=ZoneInfo(tz_name),
        )

    def test_aware_datetime_converted_to_local(self):
        dt = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        with self._patch_tz("Europe/Malta"):  # UTC+2 (DST)
            assert T.local(dt) == "2026-06-01 14:00"

    def test_iso_string_with_offset(self):
        with self._patch_tz("America/New_York"):  # UTC-4 (DST)
            assert T.local("2026-06-01T12:00:00+00:00") == "2026-06-01 08:00"

    def test_sqlite_naive_string(self):
        with self._patch_tz("Asia/Tokyo"):  # UTC+9
            assert T.local("2026-01-01 00:00:00") == "2026-01-01 09:00"

    def test_custom_format(self):
        dt = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        with self._patch_tz("Europe/Malta"):
            assert T.local(dt, fmt="%b %d %H:%M") == "Jun 01 14:00"

    def test_none_returns_none(self):
        assert T.local(None) is None

    def test_invalid_returns_none(self):
        assert T.local("") is None
        assert T.local("not-a-date") is None
        assert T.local("0001-01-01T00:00:00+00:00") is None
