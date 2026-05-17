"""
Tests for services.locale_service — the single localisation chokepoint.

Uses the real telemetry table via the `db` fixture. No mocks.
Tests the service's actual behavior: seed telemetry rows, call the service,
verify the output.
"""

import json
import pytest
from datetime import datetime, timezone


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
class TestLocaleServiceReads:
    """Service reads from telemetry and falls back to defaults."""

    def test_reads_stored_values(self, db):
        _seed_telemetry(
            db, timezone="Europe/Malta", locale="en-MT",
            currency="EUR", **{"location.lat": 35.899, "location.lon": 14.514, "location_name": "Valletta"},
        )
        from services.locale_service import get_timezone, get_locale, get_currency, get_location
        assert get_timezone().key == "Europe/Malta"
        assert get_locale() == "en-MT"
        assert get_currency() == "EUR"
        assert get_location()["lat"] == 35.899

    def test_defaults_when_empty(self, db):
        db.execute("DELETE FROM telemetry")
        db.commit()
        from services.locale_service import get_timezone, get_locale, get_currency, get_location
        assert get_timezone().key == "UTC"
        assert get_locale() == "en-US"
        assert get_currency() == "USD"
        assert get_location()["lat"] is None

    def test_invalid_timezone_falls_back_to_utc(self, db):
        _seed_telemetry(db, timezone="Not/A/Zone")
        from services.locale_service import get_timezone
        assert get_timezone().key == "UTC"


@pytest.mark.unit
class TestFormatDate:
    """format_date is THE chokepoint — for_ui converts to local, else UTC."""

    def test_for_ui_converts_to_user_timezone(self, db):
        _seed_telemetry(db, timezone="Europe/Malta")
        from services.locale_service import format_date
        dt = datetime(2026, 5, 17, 14, 0, tzinfo=timezone.utc)
        assert format_date(dt, "%H:%M", for_ui=True) == "16:00"

    def test_for_db_keeps_utc(self, db):
        _seed_telemetry(db, timezone="Europe/Malta")
        from services.locale_service import format_date
        dt = datetime(2026, 5, 17, 14, 0, tzinfo=timezone.utc)
        assert format_date(dt, "%H:%M", for_ui=False) == "14:00"

    def test_none_and_empty_return_none(self, db):
        from services.locale_service import format_date
        assert format_date(None, "%H:%M") is None
        assert format_date("", "%H:%M") is None
        assert format_date("not-a-date", "%H:%M") is None


@pytest.mark.unit
class TestCalculateInterval:
    """calculate_interval returns (utc, local) tuple."""

    def test_returns_utc_and_local(self, db):
        _seed_telemetry(db, timezone="Europe/Malta")
        from services.locale_service import calculate_interval
        dt = datetime(2026, 5, 17, 10, 0, tzinfo=timezone.utc)
        utc_dt, local_dt = calculate_interval(dt, "day", 1)
        assert utc_dt == datetime(2026, 5, 18, 10, 0, tzinfo=timezone.utc)
        assert local_dt.hour == 12  # Malta UTC+2 in May

    def test_invalid_type_raises(self, db):
        from services.locale_service import calculate_interval
        with pytest.raises(ValueError):
            calculate_interval(datetime(2026, 1, 1, tzinfo=timezone.utc), "week", 1)
