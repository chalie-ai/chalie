"""
Tests for backend/tools/date_time/handler.py
"""

import pytest
from datetime import datetime
from zoneinfo import ZoneInfo
from unittest.mock import patch, MagicMock
from tools.date_time.handler import execute


@pytest.mark.unit
class TestDateTimeHandler:
    """Test date_time tool handler."""

    def test_returns_all_expected_fields(self):
        """All 10 expected fields should be present."""
        result = execute("test_topic", {})

        expected_fields = [
            "date", "time", "timezone", "day_of_week", "unix_timestamp",
            "utc_offset", "iso_datetime", "is_weekend", "is_business_hours",
            "part_of_day"
        ]
        for field in expected_fields:
            assert field in result, f"Missing field: {field}"

    def test_part_of_day_morning(self):
        """5-11 hours should be 'morning'."""
        try:
            import freezegun
            with freezegun.freeze_time("2024-01-15 08:30:00"):
                result = execute("test_topic", {})
                assert result['part_of_day'] == 'morning'
        except ImportError:
            # Fallback: just check that a current morning/afternoon time works
            pytest.skip("freezegun not available")

    def test_part_of_day_afternoon(self):
        """12-16 hours should be 'afternoon'."""
        try:
            import freezegun
            with freezegun.freeze_time("2024-01-15 14:30:00"):
                result = execute("test_topic", {})
                assert result['part_of_day'] == 'afternoon'
        except ImportError:
            pytest.skip("freezegun not available")

    def test_part_of_day_evening(self):
        """17-20 hours should be 'evening'."""
        try:
            import freezegun
            with freezegun.freeze_time("2024-01-15 19:30:00"):
                result = execute("test_topic", {})
                assert result['part_of_day'] == 'evening'
        except ImportError:
            pytest.skip("freezegun not available")

    def test_part_of_day_night(self):
        """21-4 hours should be 'night'."""
        try:
            import freezegun
            with freezegun.freeze_time("2024-01-15 22:30:00"):
                result = execute("test_topic", {})
                assert result['part_of_day'] == 'night'
        except ImportError:
            pytest.skip("freezegun not available")

    def test_is_weekend_saturday(self):
        """Saturday should be marked as weekend."""
        try:
            import freezegun
            # 2024-01-20 is Saturday
            with freezegun.freeze_time("2024-01-20 10:00:00"):
                result = execute("test_topic", {})
                assert result['is_weekend'] is True
        except ImportError:
            pytest.skip("freezegun not available")

    def test_is_business_hours_weekday_10am(self):
        """Weekday 9-17 should be business hours."""
        try:
            import freezegun
            # 2024-01-15 is Monday
            with freezegun.freeze_time("2024-01-15 10:00:00"):
                result = execute("test_topic", {})
                assert result['is_business_hours'] is True
        except ImportError:
            pytest.skip("freezegun not available")

    def test_not_business_hours_weekend(self):
        """Weekend should not be business hours."""
        try:
            import freezegun
            # 2024-01-20 is Saturday at 10 AM
            with freezegun.freeze_time("2024-01-20 10:00:00"):
                result = execute("test_topic", {})
                assert result['is_business_hours'] is False
        except ImportError:
            pytest.skip("freezegun not available")

    def test_valid_timezone_param(self):
        """Valid timezone param should be used."""
        result = execute("test_topic", {"timezone": "US/Eastern"})

        assert "error" not in result
        assert result['timezone'] in ['US/Eastern', 'EST', 'EDT']  # May vary

    def test_invalid_timezone_falls_back(self):
        """Invalid timezone should fall back to server tz."""
        result = execute("test_topic", {"timezone": "Invalid/Timezone"})

        # Should not error, should use server timezone
        assert "error" not in result
        assert "timezone" in result

    def test_iso_datetime_format(self):
        """ISO datetime should be parseable."""
        result = execute("test_topic", {})

        iso_dt = result['iso_datetime']
        parsed = datetime.fromisoformat(iso_dt)
        assert parsed is not None
