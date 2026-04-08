"""
Ambient Wrapper — polls free public APIs and surfaces ambient context.

A daemon thread that polls free public APIs (weather, holidays, daylight) and
logs observations for ambient context.  It auto-registers itself with
WrapperAuthService on startup using the stable ID ``__ambient__``.

Poll schedule:
  - Open-Meteo (weather + UV):   every 30 minutes
  - Nager.Date (holidays):       daily at 06:00 local time
  - sunrise-sunset.org (daylight): daily at 05:00 local time

Location is derived from the most-recent ``place_fingerprints`` row, decoded
from a geohash to an approximate lat/lon (~1 km precision — never raw coords).
All location-dependent polls are silently skipped when no fingerprint exists.
"""

import json
import logging
import threading
import urllib.request
import urllib.error
from typing import Optional, Tuple

from services.config_service import ConfigService
from services.database_service import get_shared_db_service
from services.memory_client import MemoryClientService
from services.time_utils import utc_now

logger = logging.getLogger(__name__)

# ── MemoryStore keys ──────────────────────────────────────────────────────────
_LAST_WEATHER_KEY = "ambient_wrapper:last_weather"
_LAST_POLL_WEATHER_KEY = "ambient_wrapper:last_poll_weather"
_LAST_POLL_HOLIDAYS_KEY = "ambient_wrapper:last_poll_holidays"
_LAST_POLL_DAYLIGHT_KEY = "ambient_wrapper:last_poll_daylight"

# ── HTTP timeout ──────────────────────────────────────────────────────────────
_HTTP_TIMEOUT = 10

# ── Severe weather WMO codes (https://open-meteo.com/en/docs) ────────────────
# 55–57: freezing drizzle, 61–67: rain, 71–77: snow, 80–82: rain showers,
# 85–86: snow showers, 95–99: thunderstorm
_SEVERE_WEATHER_CODES = frozenset(
    list(range(55, 58)) +
    list(range(65, 68)) +
    list(range(71, 78)) +
    list(range(80, 83)) +
    list(range(85, 87)) +
    list(range(95, 100))
)


# ─────────────────────────────────────────────────────────────────────────────
# Geohash decoding (stdlib only — no external library)
# ─────────────────────────────────────────────────────────────────────────────

_GEOHASH_CHARS = "0123456789bcdefghjkmnpqrstuvwxyz"
_GEOHASH_DECODE_MAP = {ch: i for i, ch in enumerate(_GEOHASH_CHARS)}


def geohash_decode(geohash: str) -> Tuple[float, float]:
    """Decode a geohash string to an approximate (latitude, longitude) midpoint.

    The returned coordinates are accurate to approximately the geohash
    precision level (~1 km for 5-char hashes, ~150 m for 6-char hashes).

    Args:
        geohash: Lower-case geohash string (1–12 characters).

    Returns:
        ``(latitude, longitude)`` tuple as floats.

    Raises:
        ValueError: If the geohash contains an invalid character.
    """
    lat_min, lat_max = -90.0, 90.0
    lon_min, lon_max = -180.0, 180.0
    is_lon = True  # geohash alternates lon/lat bits starting with lon

    for char in geohash.lower():
        if char not in _GEOHASH_DECODE_MAP:
            raise ValueError(f"Invalid geohash character: {char!r}")
        value = _GEOHASH_DECODE_MAP[char]
        for bit_pos in range(4, -1, -1):  # 5 bits per character
            bit = (value >> bit_pos) & 1
            if is_lon:
                mid = (lon_min + lon_max) / 2
                if bit:
                    lon_min = mid
                else:
                    lon_max = mid
            else:
                mid = (lat_min + lat_max) / 2
                if bit:
                    lat_min = mid
                else:
                    lat_max = mid
            is_lon = not is_lon

    lat = (lat_min + lat_max) / 2
    lon = (lon_min + lon_max) / 2
    return lat, lon


# ─────────────────────────────────────────────────────────────────────────────
# AmbientWrapper
# ─────────────────────────────────────────────────────────────────────────────

class AmbientWrapper(threading.Thread):
    """Daemon thread that polls ambient public APIs and feeds signals to the
    reasoning loop.

    Starts automatically when registered in ``run.py`` via
    ``_try_register_ambient_wrapper()``.  The thread never crashes the process
    — all API errors are caught and logged.
    """

    daemon = True

    def __init__(self):
        """Initialise the wrapper thread and load config.

        Config is read from ``backend/configs/agents/ambient-wrapper.json``.
        Defaults are applied for any missing keys.
        """
        super().__init__(name="ambient-wrapper")
        self.wrapper_id = "__ambient__"
        self._stop_event = threading.Event()
        self._config = self._load_config()
        self._store = None  # lazy-initialised in run()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def run(self):
        """Main thread loop: register once, then poll on schedule."""
        logger.info("[AmbientWrapper] Starting")
        self._store = MemoryClientService.create_connection()
        self._register()
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception as exc:
                logger.warning("[AmbientWrapper] Unexpected error in tick: %s", exc)
            self._stop_event.wait(60)  # check every minute what polls are due
        logger.info("[AmbientWrapper] Stopped")

    def stop(self):
        """Signal the thread to exit after the current tick completes."""
        self._stop_event.set()

    # ── Registration ──────────────────────────────────────────────────────────

    def _register(self):
        """Auto-register with WrapperAuthService if not already registered.

        Uses the stable wrapper_id ``__ambient__`` so re-starts don't
        create duplicate registrations.  The bearer token is intentionally
        discarded — in-process callers do not need it.
        """
        try:
            from services.wrapper_auth_service import WrapperAuthService
            svc = WrapperAuthService()
            existing = svc.get_wrapper(self.wrapper_id)
            if existing is None:
                _token, _wid = svc.create_token(
                    name="Ambient Wrapper",
                    capabilities={
                        "signals": [
                            "weather_update",
                            "severe_weather",
                            "holiday_approaching",
                            "daylight_change",
                        ]
                    },
                    permissions={"broadcast": True},
                    metadata={"wrapper_id_override": self.wrapper_id},
                )
                logger.info("[AmbientWrapper] Registered with WrapperAuthService")
            else:
                logger.info("[AmbientWrapper] Already registered — skipping")
        except Exception as exc:
            logger.warning("[AmbientWrapper] Registration failed: %s", exc)

    # ── Tick ──────────────────────────────────────────────────────────────────

    def _tick(self):
        """Evaluate which polls are due and execute them.

        All polls are independent — a failure in one does not affect the
        others.
        """
        now = utc_now()
        local_hour = now.astimezone().hour  # system-local hour for daily polls

        if self._weather_due():
            self._poll_weather()

        if self._holidays_due(local_hour):
            self._poll_holidays()

        if self._daylight_due(local_hour):
            self._poll_daylight()

    # ── Due-check helpers ─────────────────────────────────────────────────────

    def _weather_due(self) -> bool:
        """Return True if the weather poll is overdue."""
        interval_minutes = self._config.get("poll_intervals", {}).get("weather_minutes", 30)
        return self._minutes_since(_LAST_POLL_WEATHER_KEY) >= interval_minutes

    def _holidays_due(self, local_hour: int) -> bool:
        """Return True if the holidays poll should fire this minute."""
        target_hour = self._config.get("poll_intervals", {}).get("holidays_daily_hour", 6)
        if local_hour != target_hour:
            return False
        return self._hours_since(_LAST_POLL_HOLIDAYS_KEY) >= 23

    def _daylight_due(self, local_hour: int) -> bool:
        """Return True if the daylight poll should fire this minute."""
        target_hour = self._config.get("poll_intervals", {}).get("daylight_daily_hour", 5)
        if local_hour != target_hour:
            return False
        return self._hours_since(_LAST_POLL_DAYLIGHT_KEY) >= 23

    def _minutes_since(self, key: str) -> float:
        """Return minutes elapsed since the timestamp stored at ``key``.

        Returns a large value (999) when the key is absent so a first run
        always triggers immediately.

        Args:
            key: MemoryStore key holding an ISO timestamp string.

        Returns:
            Elapsed time in minutes, or 999 if the key is unset.
        """
        raw = self._store.get(key)
        if raw is None:
            return 999.0
        try:
            from services.time_utils import parse_utc
            last = parse_utc(raw)
            elapsed = (utc_now() - last).total_seconds() / 60
            return elapsed
        except Exception:
            return 999.0

    def _hours_since(self, key: str) -> float:
        """Return hours elapsed since the timestamp stored at ``key``.

        Args:
            key: MemoryStore key holding an ISO timestamp string.

        Returns:
            Elapsed time in hours, or 999 if the key is unset.
        """
        return self._minutes_since(key) / 60

    def _stamp(self, key: str):
        """Write the current UTC time to ``key`` in the MemoryStore.

        Args:
            key: MemoryStore key to update.
        """
        self._store.set(key, utc_now().isoformat())

    # ── Location ──────────────────────────────────────────────────────────────

    def _get_location(self) -> Optional[Tuple[float, float]]:
        """Return an approximate (lat, lon) from the most-recent place fingerprint.

        Reads the ``location_hash`` column (a geohash) from the
        ``place_fingerprints`` table and decodes it.  Returns ``None`` when
        no fingerprint with a location_hash exists or when decoding fails.

        Returns:
            ``(latitude, longitude)`` tuple, or ``None``.
        """
        try:
            db = get_shared_db_service()
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT location_hash FROM place_fingerprints
                    WHERE location_hash IS NOT NULL
                    ORDER BY last_seen_at DESC
                    LIMIT 1
                    """
                )
                row = cursor.fetchone()
                cursor.close()

            if row is None or not row[0]:
                return None

            lat, lon = geohash_decode(row[0])
            return lat, lon
        except Exception as exc:
            logger.debug("[AmbientWrapper] Location lookup failed: %s", exc)
            return None

    # ── Country code helper ───────────────────────────────────────────────────

    def _get_country_code(self) -> Optional[str]:
        """Infer a 2-letter ISO country code from the system timezone.

        Uses a small, hand-coded mapping of common IANA timezone prefixes
        to country codes.  Falls back to None (holidays skipped) when the
        timezone is unknown.

        Returns:
            Two-letter uppercase country code (e.g. ``"US"``, ``"GB"``), or
            ``None``.
        """
        import time as _time
        tz_name = _time.tzname[0] if hasattr(_time, "tzname") else ""
        # Try via datetime timezone name
        try:
            import datetime
            tz_name = datetime.datetime.now().astimezone().tzname() or tz_name
        except Exception:
            pass

        # Try to get the IANA zone name via the TZ environment variable or
        # /etc/localtime symlink (best-effort, non-blocking)
        iana_zone = None
        try:
            import os
            iana_zone = os.environ.get("TZ")
            if not iana_zone:
                import subprocess
                result = subprocess.run(
                    ["timedatectl", "show", "--property=Timezone", "--value"],
                    capture_output=True, text=True, timeout=2
                )
                if result.returncode == 0:
                    iana_zone = result.stdout.strip()
        except Exception:
            pass

        # IANA continent/country prefix → ISO country code
        _TZ_COUNTRY_MAP = {
            "America/New_York": "US", "America/Chicago": "US",
            "America/Denver": "US", "America/Los_Angeles": "US",
            "America/Phoenix": "US", "America/Anchorage": "US",
            "America/Honolulu": "US", "America/Detroit": "US",
            "America/Toronto": "CA", "America/Vancouver": "CA",
            "America/Montreal": "CA", "America/Winnipeg": "CA",
            "Europe/London": "GB", "Europe/Dublin": "IE",
            "Europe/Paris": "FR", "Europe/Berlin": "DE",
            "Europe/Madrid": "ES", "Europe/Rome": "IT",
            "Europe/Amsterdam": "NL", "Europe/Brussels": "BE",
            "Europe/Zurich": "CH", "Europe/Vienna": "AT",
            "Europe/Warsaw": "PL", "Europe/Stockholm": "SE",
            "Europe/Oslo": "NO", "Europe/Copenhagen": "DK",
            "Europe/Helsinki": "FI", "Europe/Lisbon": "PT",
            "Europe/Athens": "GR", "Europe/Prague": "CZ",
            "Europe/Budapest": "HU", "Europe/Bucharest": "RO",
            "Europe/Sofia": "BG", "Europe/Zagreb": "HR",
            "Europe/Ljubljana": "SI", "Europe/Bratislava": "SK",
            "Europe/Tallinn": "EE", "Europe/Riga": "LV",
            "Europe/Vilnius": "LT", "Europe/Malta": "MT",
            "Asia/Tokyo": "JP", "Asia/Shanghai": "CN",
            "Asia/Seoul": "KR", "Asia/Kolkata": "IN",
            "Asia/Singapore": "SG", "Asia/Hong_Kong": "HK",
            "Asia/Dubai": "AE", "Asia/Jerusalem": "IL",
            "Asia/Karachi": "PK", "Asia/Dhaka": "BD",
            "Asia/Bangkok": "TH", "Asia/Jakarta": "ID",
            "Asia/Manila": "PH", "Asia/Taipei": "TW",
            "Asia/Kuala_Lumpur": "MY",
            "Australia/Sydney": "AU", "Australia/Melbourne": "AU",
            "Australia/Brisbane": "AU", "Australia/Perth": "AU",
            "Pacific/Auckland": "NZ",
            "America/Sao_Paulo": "BR", "America/Buenos_Aires": "AR",
            "America/Santiago": "CL", "America/Bogota": "CO",
            "America/Mexico_City": "MX", "America/Lima": "PE",
            "Africa/Cairo": "EG", "Africa/Johannesburg": "ZA",
            "Africa/Nairobi": "KE", "Africa/Lagos": "NG",
        }

        if iana_zone and iana_zone in _TZ_COUNTRY_MAP:
            return _TZ_COUNTRY_MAP[iana_zone]

        # Continent-prefix fallback
        if iana_zone:
            for prefix, code in _TZ_COUNTRY_MAP.items():
                if iana_zone == prefix:
                    return code

        return None

    # ── Weather poll ──────────────────────────────────────────────────────────

    def _poll_weather(self):
        """Fetch current weather from Open-Meteo and emit a reasoning signal.

        Compares the new weather code against the last emitted conditions to
        decide whether this is a routine update (activation_energy 0.2) or a
        significant change (activation_energy 0.6).  Severe weather codes
        always emit at 0.9 regardless.

        Silently skips when no location is available.
        """
        location = self._get_location()
        if location is None:
            logger.debug("[AmbientWrapper] Weather skipped — no location")
            self._stamp(_LAST_POLL_WEATHER_KEY)
            return

        lat, lon = location
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat:.4f}&longitude={lon:.4f}"
            f"&current=temperature_2m,weather_code,wind_speed_10m"
            f"&daily=weather_code,temperature_2m_max,temperature_2m_min"
            f"&timezone=auto"
        )

        try:
            data = self._fetch_json(url)
        except Exception as exc:
            logger.warning("[AmbientWrapper] Weather fetch failed: %s", exc)
            self._stamp(_LAST_POLL_WEATHER_KEY)
            return

        try:
            current = data.get("current", {})
            weather_code = current.get("weather_code", 0)
            temp = current.get("temperature_2m")
            wind = current.get("wind_speed_10m")
            daily = data.get("daily", {})
            forecast_codes = daily.get("weather_code", [])
            temp_max = daily.get("temperature_2m_max", [])
            temp_min = daily.get("temperature_2m_min", [])

            is_severe = int(weather_code) in _SEVERE_WEATHER_CODES
            last_code = self._get_last_weather_code()
            changed_significantly = (last_code is not None and last_code != int(weather_code))

            if is_severe:
                activation_energy = 0.9
                signal_type = "severe_weather"
                summary = f"Severe weather detected — WMO code {weather_code}"
                if temp is not None:
                    summary += f", {temp}°C"
            elif changed_significantly:
                activation_energy = 0.6
                signal_type = "weather_update"
                summary = f"Weather changed — now WMO {weather_code}"
                if temp is not None:
                    summary += f", {temp}°C"
            else:
                activation_energy = 0.2
                signal_type = "weather_update"
                summary = f"Current weather WMO {weather_code}"
                if temp is not None:
                    summary += f", {temp}°C"
                if wind is not None:
                    summary += f", wind {wind} km/h"

            _ = {
                "weather_code": weather_code,
                "temperature_c": temp,
                "wind_speed_kmh": wind,
                "forecast_codes": forecast_codes[:3],
                "temp_max_forecast": temp_max[:1],
                "temp_min_forecast": temp_min[:1],
                "lat": round(lat, 2),
                "lon": round(lon, 2),
            }

            self._store_last_weather_code(int(weather_code))
            logger.info(
                "[AmbientWrapper] Processed %s (code=%s, energy=%.1f)",
                signal_type, weather_code, activation_energy
            )

        except Exception as exc:
            logger.warning("[AmbientWrapper] Weather signal construction failed: %s", exc)

        self._stamp(_LAST_POLL_WEATHER_KEY)

    def _get_last_weather_code(self) -> Optional[int]:
        """Return the weather code from the last successful weather signal, or None.

        Returns:
            Integer WMO weather code or ``None`` if no previous observation.
        """
        raw = self._store.get(_LAST_WEATHER_KEY)
        if raw is None:
            return None
        try:
            return int(raw)
        except (ValueError, TypeError):
            return None

    def _store_last_weather_code(self, code: int):
        """Persist the weather code for change-detection on the next cycle.

        Args:
            code: Integer WMO weather code.
        """
        self._store.set(_LAST_WEATHER_KEY, str(code))

    # ── Holidays poll ─────────────────────────────────────────────────────────

    def _poll_holidays(self):
        """Fetch upcoming public holidays from Nager.Date and emit signals.

        Emits a ``holiday_approaching`` signal for any holiday within the
        next 3 days (activation_energy 0.5).  Skips gracefully when the
        country code cannot be inferred.
        """
        country_code = self._get_country_code()
        if country_code is None:
            logger.debug("[AmbientWrapper] Holidays skipped — no country code")
            self._stamp(_LAST_POLL_HOLIDAYS_KEY)
            return

        url = f"https://date.nager.at/api/v3/NextPublicHolidays/{country_code}"

        try:
            holidays = self._fetch_json(url)
        except Exception as exc:
            logger.warning("[AmbientWrapper] Holidays fetch failed: %s", exc)
            self._stamp(_LAST_POLL_HOLIDAYS_KEY)
            return

        try:
            if not isinstance(holidays, list):
                self._stamp(_LAST_POLL_HOLIDAYS_KEY)
                return

            today = utc_now().date()
            for holiday in holidays:
                date_str = holiday.get("date", "")
                name = holiday.get("name", "Unknown holiday")
                try:
                    from datetime import date as _date
                    holiday_date = _date.fromisoformat(date_str)
                    days_away = (holiday_date - today).days
                except Exception:
                    continue

                if 0 <= days_away <= 3:
                    logger.info(
                        "[AmbientWrapper] Holiday noted: %s in %d day(s)",
                        name, days_away
                    )
                    break  # process at most one holiday per daily poll

        except Exception as exc:
            logger.warning("[AmbientWrapper] Holiday signal construction failed: %s", exc)

        self._stamp(_LAST_POLL_HOLIDAYS_KEY)

    # ── Daylight poll ─────────────────────────────────────────────────────────

    def _poll_daylight(self):
        """Fetch sunrise/sunset times from sunrise-sunset.org and emit a signal.

        Emits a ``daylight_change`` signal with low activation energy (0.1) —
        intended as a world-model update rather than an action trigger.

        Silently skips when no location is available.
        """
        location = self._get_location()
        if location is None:
            logger.debug("[AmbientWrapper] Daylight skipped — no location")
            self._stamp(_LAST_POLL_DAYLIGHT_KEY)
            return

        lat, lon = location
        url = (
            f"https://api.sunrise-sunset.org/json"
            f"?lat={lat:.4f}&lng={lon:.4f}&formatted=0"
        )

        try:
            data = self._fetch_json(url)
        except Exception as exc:
            logger.warning("[AmbientWrapper] Daylight fetch failed: %s", exc)
            self._stamp(_LAST_POLL_DAYLIGHT_KEY)
            return

        try:
            results = data.get("results", {})
            sunrise = results.get("sunrise")
            sunset = results.get("sunset")
            day_length = results.get("day_length")  # seconds

            if not sunrise or not sunset:
                self._stamp(_LAST_POLL_DAYLIGHT_KEY)
                return

            day_hours = round(day_length / 3600, 1) if day_length else None
            summary = f"Sunrise {sunrise}, sunset {sunset}"
            if day_hours is not None:
                summary += f" ({day_hours}h daylight)"

            logger.info("[AmbientWrapper] Daylight: %s", summary)

        except Exception as exc:
            logger.warning("[AmbientWrapper] Daylight signal construction failed: %s", exc)

        self._stamp(_LAST_POLL_DAYLIGHT_KEY)

    # ── HTTP helper ───────────────────────────────────────────────────────────

    @staticmethod
    def _fetch_json(url: str) -> dict:
        """Perform a GET request and return the parsed JSON body.

        Args:
            url: Full URL to fetch.

        Returns:
            Parsed JSON as a Python object (dict or list).

        Raises:
            urllib.error.URLError: On network errors.
            ValueError: On non-200 status or invalid JSON.
        """
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Chalie-AmbientWrapper/1.0"},
        )
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
            if resp.status != 200:
                raise ValueError(f"HTTP {resp.status} from {url}")
            body = resp.read().decode("utf-8")
        return json.loads(body)

    # ── Config ────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_config() -> dict:
        """Load config from ``backend/configs/agents/ambient-wrapper.json``.

        Returns a default config dict if the file is missing or unparseable.

        Returns:
            Config dict with at minimum ``enabled`` and ``poll_intervals`` keys.
        """
        defaults = {
            "enabled": True,
            "poll_intervals": {
                "weather_minutes": 30,
                "holidays_daily_hour": 6,
                "daylight_daily_hour": 5,
            },
            "location_source": "place_fingerprints",
            "severe_weather_bypass_focus": True,
        }
        try:
            cfg = ConfigService.resolve_agent_config("ambient-wrapper")
            # Merge — config file wins over defaults
            merged = {**defaults, **cfg}
            merged["poll_intervals"] = {
                **defaults["poll_intervals"],
                **cfg.get("poll_intervals", {}),
            }
            return merged
        except Exception:
            return defaults

