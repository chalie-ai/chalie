"""
Tests for services.locale_service — the single localisation chokepoint.

Uses the real telemetry table via the `db` fixture. No mocks.
Tests the service's actual behavior: seed telemetry rows, call the service,
verify the output.
"""

import json
import pytest
from datetime import datetime, timezone
from zoneinfo import ZoneInfo


def _seed_telemetry(db, **kwargs):
    """Insert locale fields into the real telemetry table."""
    db.execute("DELETE FROM telemetry")
    for key, value in kwargs.items():
        db.execute(
            "INSERT INTO telemetry (key, value) VALUES (?, ?)",
            (key, json.dumps(value)),
        )
    db.commit()


@pytest.mark.unit
class TestGetTimezone:
    """locale_service reads timezone from telemetry and returns ZoneInfo."""

    def test_returns_stored_timezone(self, db):
        _seed_telemetry(db, timezone="Europe/Malta")
        from services.locale_service import get_timezone
        assert get_timezone().key == "Europe/Malta"

    def test_returns_utc_when_no_telemetry(self, db):
        db.execute("DELETE FROM telemetry")
        db.commit()
        from services.locale_service import get_timezone
        assert get_timezone().key == "UTC"

    def test_invalid_timezone_falls_back_to_utc(self, db):
        _seed_telemetry(db, timezone="Not/A/Zone")
        from services.locale_service import get_timezone
        assert get_timezone().key == "UTC"


@pytest.mark.unit
class TestGetLocale:

    def test_returns_stored_locale(self, db):
        _seed_telemetry(db, locale="en-MT")
        from services.locale_service import get_locale
        assert get_locale() == "en-MT"

    def test_defaults_to_en_us(self, db):
        db.execute("DELETE FROM telemetry")
        db.commit()
        from services.locale_service import get_locale
        assert get_locale() == "en-US"


@pytest.mark.unit
class TestGetCurrency:

    def test_returns_stored_currency(self, db):
        _seed_telemetry(db, currency="EUR")
        from services.locale_service import get_currency
        assert get_currency() == "EUR"

    def test_defaults_to_usd(self, db):
        db.execute("DELETE FROM telemetry")
        db.commit()
        from services.locale_service import get_currency
        assert get_currency() == "USD"


@pytest.mark.unit
class TestGetLocation:

    def test_returns_stored_location(self, db):
        _seed_telemetry(db, **{"location.lat": 35.899, "location.lon": 14.514, "location_name": "Valletta, Malta"})
        from services.locale_service import get_location
        loc = get_location()
        assert loc["lat"] == 35.899
        assert loc["lon"] == 14.514
        assert loc["name"] == "Valletta, Malta"

    def test_returns_nones_when_empty(self, db):
        db.execute("DELETE FROM telemetry")
        db.commit()
        from services.locale_service import get_location
        loc = get_location()
        assert loc["lat"] is None
        assert loc["lon"] is None
        assert loc["name"] is None


@pytest.mark.unit
class TestFormatDate:
    """format_date converts to user's local tz when for_ui=True, UTC when False."""

    def test_for_ui_converts_to_user_timezone(self, db):
        _seed_telemetry(db, timezone="Europe/Malta")
        from services.locale_service import format_date
        dt = datetime(2026, 5, 17, 14, 0, tzinfo=timezone.utc)
        # Europe/Malta is UTC+2 in May (CEST)
        assert format_date(dt, "%H:%M", for_ui=True) == "16:00"

    def test_for_db_keeps_utc(self, db):
        _seed_telemetry(db, timezone="Europe/Malta")
        from services.locale_service import format_date
        dt = datetime(2026, 5, 17, 14, 0, tzinfo=timezone.utc)
        assert format_date(dt, "%H:%M", for_ui=False) == "14:00"

    def test_string_input_parsed_as_utc(self, db):
        _seed_telemetry(db, timezone="Asia/Tokyo")
        from services.locale_service import format_date
        # Asia/Tokyo is UTC+9
        result = format_date("2026-05-17 10:00:00", "%H:%M", for_ui=True)
        assert result == "19:00"

    def test_naive_datetime_treated_as_utc(self, db):
        _seed_telemetry(db, timezone="America/New_York")
        from services.locale_service import format_date
        dt = datetime(2026, 5, 17, 20, 0)
        # New York is UTC-4 in May (EDT)
        assert format_date(dt, "%H:%M", for_ui=True) == "16:00"


@pytest.mark.unit
class TestToLocal:
    """to_local converts UTC datetimes to the user's timezone."""

    def test_utc_to_user_timezone(self, db):
        _seed_telemetry(db, timezone="Asia/Tokyo")
        from services.locale_service import to_local
        dt = datetime(2026, 5, 17, 14, 0, tzinfo=timezone.utc)
        result = to_local(dt)
        assert result.hour == 23  # UTC+9

    def test_preserves_date_across_day_boundary(self, db):
        _seed_telemetry(db, timezone="Pacific/Auckland")
        from services.locale_service import to_local
        # Auckland is UTC+12 in May
        dt = datetime(2026, 5, 17, 20, 0, tzinfo=timezone.utc)
        result = to_local(dt)
        assert result.day == 18
        assert result.hour == 8


@pytest.mark.unit
class TestLocalNow:
    """local_now returns current time in user's timezone."""

    def test_offset_matches_timezone(self, db):
        _seed_telemetry(db, timezone="Europe/Malta")
        from services.locale_service import local_now
        from services.time_utils import utc_now
        result = local_now()
        utc = utc_now()
        # Malta offset is +1 or +2 depending on DST
        diff_hours = (result.utcoffset().total_seconds()) / 3600
        assert diff_hours in (1.0, 2.0)


@pytest.mark.unit
class TestFormatCurrency:

    def test_formats_with_currency_code(self, db):
        _seed_telemetry(db, currency="EUR")
        from services.locale_service import format_currency
        assert format_currency(1234.5) == "EUR 1,234.50"

    def test_formats_without_symbol(self, db):
        _seed_telemetry(db, currency="GBP")
        from services.locale_service import format_currency
        assert format_currency(99.9, symbol=False) == "99.90"
