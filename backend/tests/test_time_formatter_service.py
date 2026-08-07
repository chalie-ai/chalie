from contextlib import AbstractContextManager
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from services.time_formatter_service import TimeFormatterService as T
from services.time_utils import utc_now


@pytest.mark.unit
class TestDurationBoundaries:
    def test_61s_minutes_tier(self) -> None:
        assert T.duration(61) == "1m"

    def test_3600s_hours_tier(self) -> None:
        assert T.duration(3600) == "1h 0m"

    def test_86400s_days_tier(self) -> None:
        assert T.duration(86400) == "1d 0h"

    def test_days_with_hours(self) -> None:
        assert T.duration(90061) == "1d 1h"


@pytest.mark.unit
class TestAgo:
    def test_past_datetime(self) -> None:
        past = utc_now() - timedelta(seconds=90)
        assert T.ago(past) == "1m ago"

    def test_iso_string(self) -> None:
        past = utc_now() - timedelta(seconds=45)
        assert T.ago(past.isoformat()) == "45s ago"

    def test_int_seconds(self) -> None:
        assert T.ago(45) == "45s ago"




@pytest.mark.unit
class TestLocalConvertsUtcToUserTimezone:
    def _patch_tz(self, tz_name: str) -> AbstractContextManager[object]:
        return patch(
            "services.locale_service.LocaleService.get_timezone",
            return_value=ZoneInfo(tz_name),
        )

    def test_aware_datetime_converted_to_local(self) -> None:
        dt = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        with self._patch_tz("Europe/Malta"):  # UTC+2 (DST)
            assert T.local(dt) == "2026-06-01 14:00"

    def test_iso_string_with_offset(self) -> None:
        with self._patch_tz("America/New_York"):  # UTC-4 (DST)
            assert T.local("2026-06-01T12:00:00+00:00") == "2026-06-01 08:00"

    def test_sqlite_naive_string(self) -> None:
        with self._patch_tz("Asia/Tokyo"):  # UTC+9
            assert T.local("2026-01-01 00:00:00") == "2026-01-01 09:00"

    def test_custom_format(self) -> None:
        dt = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        with self._patch_tz("Europe/Malta"):
            assert T.local(dt, fmt="%b %d %H:%M") == "Jun 01 14:00"



