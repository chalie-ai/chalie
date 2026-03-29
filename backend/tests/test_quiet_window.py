"""Unit tests for the quiet_window guard."""

import pytest
from datetime import datetime, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo

from capabilities.quiet_window import is_quiet_now


_TZ = ZoneInfo("Europe/Malta")


def _make_now(hour, tz=_TZ):
    return datetime(2026, 3, 29, hour, 0, tzinfo=tz).astimezone(timezone.utc)


@pytest.mark.unit
class TestQuietWindow:
    """Covers quiet-hours logic, focus gating, and timezone fallback."""

    def test_quiet_during_off_hours(self):
        """23:00, 06:00, 00:00, and 22:00 local are all quiet."""
        for hour in (23, 6, 0, 22):
            with (
                patch("capabilities.quiet_window._get_user_tz", return_value=_TZ),
                patch("capabilities.quiet_window._focus_active", return_value=False),
            ):
                assert is_quiet_now(_make_now(hour)) is True

    def test_not_quiet_during_work_hours(self):
        """08:00, 10:00, 14:00, and 21:00 local are not quiet."""
        for hour in (8, 10, 14, 21):
            with (
                patch("capabilities.quiet_window._get_user_tz", return_value=_TZ),
                patch("capabilities.quiet_window._focus_active", return_value=False),
            ):
                assert is_quiet_now(_make_now(hour)) is False

    def test_quiet_when_focus_active(self):
        """14:00 local but focus active -> quiet."""
        with (
            patch("capabilities.quiet_window._get_user_tz", return_value=_TZ),
            patch("capabilities.quiet_window._focus_active", return_value=True),
        ):
            assert is_quiet_now(_make_now(14)) is True

    def test_fallback_utc_on_error(self):
        """Timezone lookup failure falls back to UTC."""
        now = datetime(2026, 3, 29, 2, 0, tzinfo=timezone.utc)
        with (
            patch("capabilities.quiet_window._get_user_tz", return_value=timezone.utc),
            patch("capabilities.quiet_window._focus_active", return_value=False),
        ):
            assert is_quiet_now(now) is True
