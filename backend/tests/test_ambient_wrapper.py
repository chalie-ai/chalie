"""
Unit tests for AmbientWrapper.

All HTTP calls are mocked via unittest.mock.patch on urllib.request.urlopen.
Tests use in-memory SQLite and an isolated MemoryStore so they carry no
external dependencies.
"""

import json
import sqlite3
import threading
import time
import unittest
from io import BytesIO
from unittest.mock import MagicMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Ensure the backend directory is on the path for direct imports
# ---------------------------------------------------------------------------
import os
import sys

_backend_dir = os.path.join(os.path.dirname(__file__), "..")
if _backend_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(_backend_dir))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_http_response(data, status=200):
    """Return a mock urllib response context manager.

    Args:
        data: Python object to serialise as JSON.
        status: HTTP status code to report.

    Returns:
        MagicMock usable as a context manager (``with urlopen(...) as resp``).
    """
    body = json.dumps(data).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.status = status
    mock_resp.read.return_value = body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _make_memory_store():
    """Return a minimal thread-safe in-memory key-value store for tests."""
    from services.memory_store import MemoryStore
    return MemoryStore()


def _make_db():
    """Return an in-memory SQLite connection with the place_fingerprints table."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS place_fingerprints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fingerprint_hash TEXT UNIQUE NOT NULL,
            device_class TEXT NOT NULL,
            hour_bucket INTEGER NOT NULL,
            location_hash TEXT,
            connection_type TEXT,
            place_label TEXT NOT NULL,
            count INTEGER DEFAULT 1,
            last_seen_at TEXT DEFAULT (datetime('now'))
        )
        """
    )
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Geohash decoding tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGeohashDecode:
    """Tests for the pure-Python geohash decoder."""

    def test_known_geohash_u09t(self):
        """'u09t' decodes to near Paris (~48.8°N, 2.3°E)."""
        from wrappers.ambient_wrapper import geohash_decode
        lat, lon = geohash_decode("u09t")
        assert 48.0 <= lat <= 50.0
        assert 1.0 <= lon <= 4.0

    def test_known_geohash_9q8y(self):
        """'9q8y' decodes to near San Francisco (~37.7°N, -122.5°W)."""
        from wrappers.ambient_wrapper import geohash_decode
        lat, lon = geohash_decode("9q8y")
        assert 37.0 <= lat <= 38.5
        assert -123.5 <= lon <= -121.0

    def test_geohash_equator_prime_meridian(self):
        """'s' decodes to the north-east quadrant (lat>0, lon>0)."""
        from wrappers.ambient_wrapper import geohash_decode
        lat, lon = geohash_decode("s")
        assert lat > 0
        assert lon > 0

    def test_invalid_char_raises(self):
        """An invalid character should raise ValueError."""
        from wrappers.ambient_wrapper import geohash_decode
        with pytest.raises(ValueError, match="Invalid geohash character"):
            geohash_decode("u09t!")

    def test_roundtrip_precision(self):
        """Longer geohashes produce narrower bounding boxes."""
        from wrappers.ambient_wrapper import geohash_decode
        lat4, lon4 = geohash_decode("u09t")
        lat6, lon6 = geohash_decode("u09tvy")
        # 6-char hash should resolve closer to a specific point
        assert lat6 != lat4 or lon6 != lon4

    def test_southern_hemisphere(self):
        """Geohashes in the southern hemisphere should yield negative latitudes."""
        from wrappers.ambient_wrapper import geohash_decode
        # 'r' encodes the Australia / south-east Asia region
        lat, lon = geohash_decode("r")
        # The 'r' cell covers roughly lat [-90, 0] lon [90, 180] — lat is negative
        assert lat < 0


# ---------------------------------------------------------------------------
# AmbientWrapper unit tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAmbientWrapper:
    """Unit tests for AmbientWrapper poll methods and signal emission."""

    def _make_wrapper(self, store=None, db_conn=None):
        """Construct a wrapper instance with injected fakes.

        Args:
            store: MemoryStore-like object.  If None, a real MemoryStore is used.
            db_conn: sqlite3.Connection for place_fingerprints.  If None, an
                in-memory DB without fingerprints is used.

        Returns:
            Configured :class:`AmbientWrapper` instance (not started).
        """
        from wrappers.ambient_wrapper import AmbientWrapper

        wrapper = AmbientWrapper.__new__(AmbientWrapper)
        threading.Thread.__init__(wrapper, name="ambient-wrapper")
        wrapper.daemon = True
        wrapper.wrapper_id = "__ambient__"
        wrapper._stop_event = threading.Event()
        wrapper._config = {
            "enabled": True,
            "poll_intervals": {
                "weather_minutes": 30,
                "holidays_daily_hour": 6,
                "daylight_daily_hour": 5,
            },
            "location_source": "place_fingerprints",
            "severe_weather_bypass_focus": True,
        }
        wrapper._store = store or _make_memory_store()
        wrapper._db_conn = db_conn  # stashed for use in helpers below
        return wrapper

    def _inject_fingerprint(self, wrapper, geohash: str):
        """Insert a place fingerprint row into the wrapper's test DB.

        Args:
            wrapper: Wrapper instance created by :meth:`_make_wrapper`.
            geohash: Geohash string to store in ``location_hash``.
        """
        wrapper._db_conn.execute(
            """
            INSERT INTO place_fingerprints
                (fingerprint_hash, device_class, hour_bucket, location_hash, place_label)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("hash1", "desktop", 9, geohash, "home"),
        )
        wrapper._db_conn.commit()

    # ── No fingerprints — location-dependent polls skip gracefully ────────────

    def test_get_location_returns_none_when_no_fingerprints(self):
        """_get_location() returns None when place_fingerprints is empty."""
        db = _make_db()
        wrapper = self._make_wrapper(db_conn=db)

        with patch("wrappers.ambient_wrapper.get_shared_db_service") as mock_db_svc:
            mock_db_svc.return_value.connection.return_value.__enter__ = lambda s: db
            mock_db_svc.return_value.connection.return_value.__exit__ = MagicMock(
                return_value=False
            )
            # Use the real connection context manager behaviour
            from contextlib import contextmanager

            @contextmanager
            def _fake_conn():
                yield db

            mock_db_svc.return_value.connection = _fake_conn
            result = wrapper._get_location()

        assert result is None

    def test_weather_poll_skips_gracefully_without_location(self):
        """Weather poll stamps last_poll_weather but emits no signal when there is
        no location fingerprint."""
        wrapper = self._make_wrapper()
        emitted = []

        with patch("wrappers.ambient_wrapper.get_shared_db_service") as mock_db_svc, \
             patch("wrappers.ambient_wrapper.emit_reasoning_signal",
                   side_effect=lambda s: emitted.append(s) or True):

            from contextlib import contextmanager

            @contextmanager
            def _fake_conn():
                yield _make_db()  # empty DB — no rows

            mock_db_svc.return_value.connection = _fake_conn
            wrapper._poll_weather()

        assert len(emitted) == 0, "No signal should be emitted without a location"

    # ── Weather signal emission ───────────────────────────────────────────────

    def test_weather_emits_routine_signal(self):
        """A routine (unchanged) weather code emits activation_energy 0.2."""
        db = _make_db()
        wrapper = self._make_wrapper(db_conn=db)
        self._inject_fingerprint(wrapper, "u09t")

        weather_response = {
            "current": {"temperature_2m": 18.5, "weather_code": 1, "wind_speed_10m": 12},
            "daily": {
                "weather_code": [1, 1, 2],
                "temperature_2m_max": [20.0],
                "temperature_2m_min": [14.0],
            },
        }

        emitted = []
        mock_resp = _make_http_response(weather_response)

        from contextlib import contextmanager

        @contextmanager
        def _fake_conn():
            yield db

        with patch("wrappers.ambient_wrapper.get_shared_db_service") as mock_db_svc, \
             patch("urllib.request.urlopen", return_value=mock_resp), \
             patch("wrappers.ambient_wrapper.emit_reasoning_signal",
                   side_effect=lambda s: emitted.append(s) or True):

            mock_db_svc.return_value.connection = _fake_conn
            wrapper._poll_weather()

        assert len(emitted) == 1
        sig = emitted[0]
        assert sig.signal_type == "weather_update"
        assert sig.activation_energy == pytest.approx(0.2)
        assert sig.wrapper_id == "__ambient__"
        assert sig.metadata["weather_code"] == 1

    def test_weather_emits_significant_change_signal(self):
        """A weather code change emits activation_energy 0.6."""
        db = _make_db()
        wrapper = self._make_wrapper(db_conn=db)
        self._inject_fingerprint(wrapper, "u09t")

        # Pretend last code was clear sky (1), now it's overcast (3)
        wrapper._store.set("ambient_wrapper:last_weather", "1")

        weather_response = {
            "current": {"temperature_2m": 15.0, "weather_code": 3, "wind_speed_10m": 8},
            "daily": {
                "weather_code": [3, 3, 3],
                "temperature_2m_max": [17.0],
                "temperature_2m_min": [11.0],
            },
        }

        emitted = []
        mock_resp = _make_http_response(weather_response)

        from contextlib import contextmanager

        @contextmanager
        def _fake_conn():
            yield db

        with patch("wrappers.ambient_wrapper.get_shared_db_service") as mock_db_svc, \
             patch("urllib.request.urlopen", return_value=mock_resp), \
             patch("wrappers.ambient_wrapper.emit_reasoning_signal",
                   side_effect=lambda s: emitted.append(s) or True):

            mock_db_svc.return_value.connection = _fake_conn
            wrapper._poll_weather()

        assert len(emitted) == 1
        sig = emitted[0]
        assert sig.signal_type == "weather_update"
        assert sig.activation_energy == pytest.approx(0.6)

    def test_weather_emits_severe_weather_signal(self):
        """A severe WMO code (e.g. 95 — thunderstorm) emits activation_energy 0.9."""
        db = _make_db()
        wrapper = self._make_wrapper(db_conn=db)
        self._inject_fingerprint(wrapper, "u09t")

        weather_response = {
            "current": {
                "temperature_2m": 22.0,
                "weather_code": 95,  # Thunderstorm
                "wind_speed_10m": 55,
            },
            "daily": {
                "weather_code": [95],
                "temperature_2m_max": [25.0],
                "temperature_2m_min": [19.0],
            },
        }

        emitted = []
        mock_resp = _make_http_response(weather_response)

        from contextlib import contextmanager

        @contextmanager
        def _fake_conn():
            yield db

        with patch("wrappers.ambient_wrapper.get_shared_db_service") as mock_db_svc, \
             patch("urllib.request.urlopen", return_value=mock_resp), \
             patch("wrappers.ambient_wrapper.emit_reasoning_signal",
                   side_effect=lambda s: emitted.append(s) or True):

            mock_db_svc.return_value.connection = _fake_conn
            wrapper._poll_weather()

        assert len(emitted) == 1
        sig = emitted[0]
        assert sig.signal_type == "severe_weather"
        assert sig.activation_energy == pytest.approx(0.9)

    def test_weather_poll_survives_http_error(self):
        """A network error during weather fetch should not raise from _poll_weather."""
        db = _make_db()
        wrapper = self._make_wrapper(db_conn=db)
        self._inject_fingerprint(wrapper, "u09t")

        from contextlib import contextmanager

        @contextmanager
        def _fake_conn():
            yield db

        import urllib.error

        with patch("wrappers.ambient_wrapper.get_shared_db_service") as mock_db_svc, \
             patch("urllib.request.urlopen",
                   side_effect=urllib.error.URLError("timeout")):

            mock_db_svc.return_value.connection = _fake_conn
            # Should NOT raise
            wrapper._poll_weather()

    # ── Holiday signal emission ───────────────────────────────────────────────

    def test_holiday_emits_signal_when_approaching(self):
        """A holiday within 3 days emits a holiday_approaching signal."""
        from datetime import date, timedelta
        wrapper = self._make_wrapper()

        today = date.today()
        tomorrow = today + timedelta(days=1)

        holidays_response = [
            {
                "date": tomorrow.isoformat(),
                "name": "Test Holiday",
                "localName": "Test Holiday Local",
            }
        ]

        emitted = []
        mock_resp = _make_http_response(holidays_response)

        with patch("urllib.request.urlopen", return_value=mock_resp), \
             patch("wrappers.ambient_wrapper.emit_reasoning_signal",
                   side_effect=lambda s: emitted.append(s) or True), \
             patch.object(wrapper, "_get_country_code", return_value="US"):

            wrapper._poll_holidays()

        assert len(emitted) == 1
        sig = emitted[0]
        assert sig.signal_type == "holiday_approaching"
        assert sig.activation_energy == pytest.approx(0.5)
        assert sig.metadata["days_away"] == 1
        assert "Test Holiday" in sig.content

    def test_holiday_no_signal_when_far_away(self):
        """A holiday more than 3 days away should not trigger a signal."""
        from datetime import date, timedelta
        wrapper = self._make_wrapper()

        far_date = date.today() + timedelta(days=10)
        holidays_response = [
            {
                "date": far_date.isoformat(),
                "name": "Distant Holiday",
                "localName": "Distant Holiday",
            }
        ]

        emitted = []
        mock_resp = _make_http_response(holidays_response)

        with patch("urllib.request.urlopen", return_value=mock_resp), \
             patch("wrappers.ambient_wrapper.emit_reasoning_signal",
                   side_effect=lambda s: emitted.append(s) or True), \
             patch.object(wrapper, "_get_country_code", return_value="US"):

            wrapper._poll_holidays()

        assert len(emitted) == 0

    def test_holiday_skips_when_no_country_code(self):
        """Holidays poll skips gracefully when country code cannot be inferred."""
        wrapper = self._make_wrapper()
        emitted = []

        with patch.object(wrapper, "_get_country_code", return_value=None), \
             patch("wrappers.ambient_wrapper.emit_reasoning_signal",
                   side_effect=lambda s: emitted.append(s) or True):

            wrapper._poll_holidays()

        assert len(emitted) == 0

    def test_holiday_survives_http_error(self):
        """A network error during holidays fetch should not raise."""
        wrapper = self._make_wrapper()
        import urllib.error

        with patch.object(wrapper, "_get_country_code", return_value="US"), \
             patch("urllib.request.urlopen",
                   side_effect=urllib.error.URLError("network error")):

            # Should NOT raise
            wrapper._poll_holidays()

    # ── Daylight signal emission ──────────────────────────────────────────────

    def test_daylight_emits_signal(self):
        """Daylight poll emits a daylight_change signal with energy 0.1."""
        db = _make_db()
        wrapper = self._make_wrapper(db_conn=db)
        self._inject_fingerprint(wrapper, "u09t")

        daylight_response = {
            "results": {
                "sunrise": "2026-03-16T06:15:00+00:00",
                "sunset": "2026-03-16T18:30:00+00:00",
                "day_length": 44100,
            },
            "status": "OK",
        }

        emitted = []
        mock_resp = _make_http_response(daylight_response)

        from contextlib import contextmanager

        @contextmanager
        def _fake_conn():
            yield db

        with patch("wrappers.ambient_wrapper.get_shared_db_service") as mock_db_svc, \
             patch("urllib.request.urlopen", return_value=mock_resp), \
             patch("wrappers.ambient_wrapper.emit_reasoning_signal",
                   side_effect=lambda s: emitted.append(s) or True):

            mock_db_svc.return_value.connection = _fake_conn
            wrapper._poll_daylight()

        assert len(emitted) == 1
        sig = emitted[0]
        assert sig.signal_type == "daylight_change"
        assert sig.activation_energy == pytest.approx(0.1)
        assert sig.wrapper_id == "__ambient__"
        assert "sunrise" in sig.content.lower() or "Sunrise" in sig.content

    def test_daylight_skips_without_location(self):
        """Daylight poll emits no signal when location is unavailable."""
        wrapper = self._make_wrapper()
        emitted = []

        from contextlib import contextmanager

        @contextmanager
        def _fake_conn():
            yield _make_db()

        with patch("wrappers.ambient_wrapper.get_shared_db_service") as mock_db_svc, \
             patch("wrappers.ambient_wrapper.emit_reasoning_signal",
                   side_effect=lambda s: emitted.append(s) or True):

            mock_db_svc.return_value.connection = _fake_conn
            wrapper._poll_daylight()

        assert len(emitted) == 0

    def test_daylight_survives_http_error(self):
        """A network error during daylight fetch should not raise."""
        db = _make_db()
        wrapper = self._make_wrapper(db_conn=db)
        self._inject_fingerprint(wrapper, "u09t")

        import urllib.error
        from contextlib import contextmanager

        @contextmanager
        def _fake_conn():
            yield db

        with patch("wrappers.ambient_wrapper.get_shared_db_service") as mock_db_svc, \
             patch("urllib.request.urlopen",
                   side_effect=urllib.error.URLError("timeout")):

            mock_db_svc.return_value.connection = _fake_conn
            wrapper._poll_daylight()

    # ── Weather change detection ──────────────────────────────────────────────

    def test_last_weather_code_stored_after_emission(self):
        """After a successful weather poll, the WMO code is stored in MemoryStore."""
        db = _make_db()
        store = _make_memory_store()
        wrapper = self._make_wrapper(store=store, db_conn=db)
        self._inject_fingerprint(wrapper, "u09t")

        weather_response = {
            "current": {"temperature_2m": 10.0, "weather_code": 45, "wind_speed_10m": 5},
            "daily": {
                "weather_code": [45],
                "temperature_2m_max": [12.0],
                "temperature_2m_min": [8.0],
            },
        }

        mock_resp = _make_http_response(weather_response)

        from contextlib import contextmanager

        @contextmanager
        def _fake_conn():
            yield db

        with patch("wrappers.ambient_wrapper.get_shared_db_service") as mock_db_svc, \
             patch("urllib.request.urlopen", return_value=mock_resp), \
             patch("wrappers.ambient_wrapper.emit_reasoning_signal", return_value=True):

            mock_db_svc.return_value.connection = _fake_conn
            wrapper._poll_weather()

        stored = store.get("ambient_wrapper:last_weather")
        assert stored == "45"

    def test_no_last_code_means_first_poll_is_routine(self):
        """Without a previous code, the first poll uses routine energy (0.2)."""
        db = _make_db()
        wrapper = self._make_wrapper(db_conn=db)
        self._inject_fingerprint(wrapper, "u09t")

        # Ensure no previous code is stored
        assert wrapper._get_last_weather_code() is None

        weather_response = {
            "current": {"temperature_2m": 5.0, "weather_code": 2, "wind_speed_10m": 3},
            "daily": {
                "weather_code": [2],
                "temperature_2m_max": [7.0],
                "temperature_2m_min": [3.0],
            },
        }

        emitted = []
        mock_resp = _make_http_response(weather_response)

        from contextlib import contextmanager

        @contextmanager
        def _fake_conn():
            yield db

        with patch("wrappers.ambient_wrapper.get_shared_db_service") as mock_db_svc, \
             patch("urllib.request.urlopen", return_value=mock_resp), \
             patch("wrappers.ambient_wrapper.emit_reasoning_signal",
                   side_effect=lambda s: emitted.append(s) or True):

            mock_db_svc.return_value.connection = _fake_conn
            wrapper._poll_weather()

        assert len(emitted) == 1
        assert emitted[0].activation_energy == pytest.approx(0.2)
