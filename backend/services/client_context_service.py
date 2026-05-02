"""
Client Context Service — Stores and retrieves client timezone, location, device info,
behavioral signals, and system info.

The raw heartbeat payload (whatever the frontend sends) is persisted to the
``telemetry`` table as flat key/value rows, e.g. ``device.name`` → ``"iPhone"``.
The frontend (heartbeat.js) is the single source of truth for which keys are
collected; this service blindly persists what it receives and reconstructs the
nested dict on read so existing consumers (time_utils, scheduler_skill,
…) see the same shape they always did.

Side concerns that stay in MemoryStore (NOT telemetry):
  - location-history ring buffer (mobility inference)
  - session re-entry / place-transition flags (ephemeral inference state)
  - culture-seeding cookie

Side concerns persisted elsewhere:
  - timezone → settings table (survives restarts, used by scheduler/CalDAV)
  - demographic traits → data_graph
"""

import json
import time
import logging
import requests
from services.memory_client import MemoryClientService
from services.database_service import get_shared_db_service


HISTORY_KEY = "client_context:history"
HISTORY_MAX = 12  # ~1hr at 5min intervals
TTL = 3600  # 1 hour (used by ephemeral MemoryStore keys, not telemetry)

# Session re-entry: user returned after extended absence
REENTRY_KEY = "ambient:session_reentry"
REENTRY_THRESHOLD = 1800  # 30 min
REENTRY_TTL = 300  # 5 min flag

# Demographic seeding
CULTURE_SEED_KEY = "ambient:culture_seeded"

# Locale → culture region mapping (Possible tier, source: inferred)
LOCALE_CULTURE_MAP = {
    "mt": "mediterranean_european",
    "it": "mediterranean_european",
    "es": "mediterranean_european",
    "pt": "mediterranean_european",
    "el": "mediterranean_european",
    "fr": "western_european",
    "de": "western_european",
    "nl": "western_european",
    "da": "northern_european",
    "sv": "northern_european",
    "no": "northern_european",
    "fi": "northern_european",
    "ja": "east_asian",
    "zh": "east_asian",
    "ko": "east_asian",
    "hi": "south_asian",
    "bn": "south_asian",
    "ar": "middle_eastern",
    "he": "middle_eastern",
    "tr": "middle_eastern",
    "ru": "eastern_european",
    "pl": "eastern_european",
    "uk": "eastern_european",
    "cs": "eastern_european",
}

# Region-specific locale overrides (language-country combos)
LOCALE_REGION_OVERRIDES = {
    "pt-BR": "latin_american",
    "es-MX": "latin_american",
    "es-AR": "latin_american",
    "es-CO": "latin_american",
    "es-CL": "latin_american",
    "en-IN": "south_asian",
    "en-MT": "mediterranean_european",
    "en-ZA": "sub_saharan_african",
    "en-NG": "sub_saharan_african",
    "en-AU": "oceanian",
    "en-NZ": "oceanian",
}


class ClientContextService:
    """Manages client context (timezone, location, device, behavioral signals).

    Telemetry is persisted to the ``telemetry`` table (flat key/value).  The
    MemoryStore connection is retained for ephemeral inference flags
    (place-transition, session-reentry, culture-seed) and the location-history
    ring buffer — not for telemetry itself.
    """

    def __init__(self):
        """Initialize the service and open a MemoryStore connection."""
        self._store = MemoryClientService.create_connection()

    # ── Telemetry table helpers ────────────────────────────────────────

    @staticmethod
    def _flatten(ctx: dict, prefix: str = "") -> dict[str, str]:
        """Flatten a nested dict into ``{"a.b.c": json_str_value}``.

        Leaf values (anything that isn't a non-empty dict) are JSON-encoded
        so type fidelity round-trips through the TEXT column.  Empty dicts
        are dropped — they carry no information once persisted as rows.
        """
        out: dict[str, str] = {}
        for key, value in ctx.items():
            full_key = f"{prefix}{key}"
            if isinstance(value, dict) and value:
                out.update(ClientContextService._flatten(value, prefix=f"{full_key}."))
            else:
                out[full_key] = json.dumps(value)
        return out

    @staticmethod
    def _unflatten(rows: dict[str, str]) -> dict:
        """Rebuild the nested dict from the flat ``{key: json_str}`` rows."""
        out: dict = {}
        for flat_key, raw_value in rows.items():
            try:
                value = json.loads(raw_value)
            except (TypeError, ValueError):
                value = raw_value
            parts = flat_key.split(".")
            cursor = out
            for part in parts[:-1]:
                existing = cursor.get(part)
                if not isinstance(existing, dict):
                    existing = {}
                    cursor[part] = existing
                cursor = existing
            cursor[parts[-1]] = value
        return out

    def _resolve_location_name(self, lat: float, lon: float) -> str | None:
        """Resolve a human-readable city/country name from coordinates.

        Calls the Nominatim reverse-geocoding API (OpenStreetMap). Prefers
        city → town → municipality → county → state_district as the locality
        label, combined with the country name.

        Args:
            lat: Latitude in decimal degrees.
            lon: Longitude in decimal degrees.

        Returns:
            A string such as ``"Valletta, Malta"`` on success, or ``None`` if
            the API call fails or returns an unusable address.
        """
        try:
            url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json&accept-language=en"
            headers = {"User-Agent": "Chalie/1.0"}
            response = requests.get(url, headers=headers, timeout=3)
            response.raise_for_status()
            data = response.json()
            address = data.get("address", {})
            city = (address.get("city") or address.get("town") or
                    address.get("municipality") or address.get("county") or
                    address.get("state_district") or "")
            country = address.get("country", "")
            if city and country:
                return f"{city}, {country}"
            if country:
                return country
        except (requests.RequestException, KeyError, ValueError) as e:
            logging.debug(f"[CLIENT CONTEXT] Failed to resolve location: {e}")
        return None

    def save(self, ctx: dict):
        """
        Save client context to MemoryStore with extended processing.

        Handles: location resolution, behavioral data merging, location history,
        session re-entry, and demographic seeding.
        """
        cached_ctx = self.get()

        # Merge behavioral data: don't overwrite if new heartbeat lacks it
        if "behavioral" not in ctx and "behavioral" in cached_ctx:
            ctx["behavioral"] = cached_ctx["behavioral"]

        # Resolve location name if location changed significantly
        if location := ctx.get("location"):
            cached_location = cached_ctx.get("location", {})
            lat_changed = abs(location.get("lat", 0) - cached_location.get("lat", 0)) > 0.05
            lon_changed = abs(location.get("lon", 0) - cached_location.get("lon", 0)) > 0.05

            no_cached_name = "location_name" not in cached_ctx
            cached_stale = cached_ctx.get("_location_name_stale", False)

            if lat_changed or lon_changed or no_cached_name or cached_stale:
                location_name = self._resolve_location_name(location["lat"], location["lon"])
                if location_name:
                    ctx["location_name"] = location_name
                    ctx.pop("_location_name_stale", None)
                    logging.debug(f"[CLIENT CONTEXT] Resolved location: {location_name}")
                else:
                    if "location_name" in cached_ctx:
                        ctx["location_name"] = cached_ctx["location_name"]
                    ctx["_location_name_stale"] = True
                    logging.debug("[CLIENT CONTEXT] Location resolve failed, marked stale for retry")
            else:
                if "location_name" in cached_ctx:
                    ctx["location_name"] = cached_ctx["location_name"]

        # Session re-entry detection
        self._check_session_reentry(cached_ctx)

        # Persist timezone to settings (survives restarts, enables CalDAV/scheduler)
        if tz_name := ctx.get("timezone"):
            self._persist_timezone(tz_name, cached_ctx.get("timezone"))

        # Persist the heartbeat to the telemetry table — replace-all so deleted
        # FE keys disappear from the next render.
        ctx["saved_at"] = time.time()
        flat = self._flatten(ctx)
        db = get_shared_db_service()
        with db.connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("DELETE FROM telemetry")
                cursor.executemany(
                    "INSERT INTO telemetry (key, value) VALUES (?, ?)",
                    list(flat.items()),
                )
                conn.commit()
            finally:
                cursor.close()

        # Location history ring buffer (for mobility inference)
        self._push_history(ctx)

        # Demographic trait seeding (once per session)
        self._seed_demographic_traits(ctx)

        logging.debug(f"[CLIENT CONTEXT] Saved context with timezone={ctx.get('timezone')}, "
                     f"device={ctx.get('device', {}).get('class')}")

    def get(self) -> dict:
        """Retrieve client context from the telemetry table as a nested dict."""
        db = get_shared_db_service()
        with db.connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT key, value FROM telemetry")
                rows = dict(cursor.fetchall())
            finally:
                cursor.close()
        return self._unflatten(rows) if rows else {}

    def is_stale(self, max_age_seconds: int = 600) -> bool:
        """Check whether the stored client context is older than the allowed age.

        Args:
            max_age_seconds: Maximum acceptable age in seconds. Defaults to
                600 (10 minutes).

        Returns:
            ``True`` if the stored context is missing or its ``saved_at``
            timestamp is older than ``max_age_seconds``, ``False`` otherwise.
        """
        ctx = self.get()
        saved_at = ctx.get("saved_at", 0)
        is_stale = (time.time() - saved_at) > max_age_seconds
        if is_stale and ctx:
            age = time.time() - saved_at
            logging.debug(f"[CLIENT CONTEXT] Context is stale (age={age:.0f}s, max={max_age_seconds}s)")
        return is_stale

    # ── Location History ───────────────────────────────────────────────

    def _push_history(self, ctx: dict):
        """Push current context snapshot to location history ring buffer."""
        entry = {"saved_at": ctx.get("saved_at")}
        if location := ctx.get("location"):
            entry["location"] = location
        if connection := ctx.get("connection"):
            entry["connection"] = connection
        if network := ctx.get("network"):
            entry["network"] = network

        try:
            self._store.lpush(HISTORY_KEY, json.dumps(entry))
            self._store.ltrim(HISTORY_KEY, 0, HISTORY_MAX - 1)
            self._store.expire(HISTORY_KEY, TTL)
        except Exception as e:
            logging.debug(f"[CLIENT CONTEXT] Failed to push history: {e}")

    # ── Session Re-entry Detection ─────────────────────────────────────

    def _check_session_reentry(self, cached_ctx: dict):
        """Detect if user returned after extended absence (>30min)."""
        if not cached_ctx:
            return

        saved_at = cached_ctx.get("saved_at", 0)
        if not saved_at:
            return

        age = time.time() - saved_at
        if age > REENTRY_THRESHOLD:
            try:
                self._store.setex(REENTRY_KEY, REENTRY_TTL, json.dumps({
                    "absent_seconds": int(age),
                    "returned_at": time.time(),
                }))
                logging.debug(f"[CLIENT CONTEXT] Session re-entry detected (absent {age:.0f}s)")
            except Exception as e:
                logging.debug(f"[CLIENT CONTEXT] Re-entry flag failed: {e}")

    def _persist_timezone(self, tz_name: str, previous_tz: str | None):
        """Write user_timezone to settings if it changed.

        Only writes on actual change to avoid unnecessary DB churn on every
        heartbeat (sent every 5 minutes).
        """
        if tz_name == previous_tz:
            return
        try:
            from zoneinfo import ZoneInfo
            ZoneInfo(tz_name)  # validate IANA name
        except Exception:
            logging.debug(f"[CLIENT CONTEXT] Invalid timezone '{tz_name}', not persisting")
            return
        try:
            from services.database_service import get_shared_db_service
            from services.settings_service import SettingsService
            SettingsService(get_shared_db_service()).set(
                "user_timezone", tz_name, "string",
                "User IANA timezone (auto-detected from client heartbeat)"
            )
            logging.debug(f"[CLIENT CONTEXT] Persisted user_timezone={tz_name}")
        except Exception as e:
            logging.debug(f"[CLIENT CONTEXT] Failed to persist timezone: {e}")

    def is_session_reentry(self) -> bool:
        """Check if the user just returned from an extended absence."""
        return bool(self._store.get(REENTRY_KEY))

    def get_timezone_offset(self) -> int | None:
        """Return the user's UTC offset in minutes, derived from the stored IANA timezone.

        Positive values mean east of UTC (e.g. UTC+2 → 120).
        Negative values mean west of UTC (e.g. UTC-5 → -300).
        Returns None if no timezone is stored or the ZoneInfo lookup fails.
        """
        ctx = self.get()
        timezone = ctx.get("timezone")
        if not timezone:
            return None
        try:
            from zoneinfo import ZoneInfo
            from datetime import datetime, timezone as dt_timezone
            # Use the current wall-clock moment so DST is accounted for correctly.
            now_utc = datetime.now(dt_timezone.utc)
            tz = ZoneInfo(timezone)
            local_dt = now_utc.astimezone(tz)
            # utcoffset() returns a timedelta; convert to whole minutes.
            offset = local_dt.utcoffset()
            if offset is not None:
                return int(offset.total_seconds() // 60)
        except Exception as e:
            logging.debug(f"[CLIENT CONTEXT] get_timezone_offset failed: {e}")
        return None

    # ── Demographic Trait Seeding ──────────────────────────────────────

    def _seed_demographic_traits(self, ctx: dict):
        """
        Seed culture-region trait from locale/location (Possible tier).
        Runs once — subsequent reinforcement comes from conversation.
        Religion, gender, and age are NEVER telemetry-seeded.
        """
        # Only seed once
        if self._store.get(CULTURE_SEED_KEY):
            return

        locale = ctx.get("locale", "")
        language = ctx.get("language", "")

        # Try region-specific overrides first (e.g., pt-BR → latin_american)
        culture = None
        for locale_key in [locale, language]:
            if locale_key in LOCALE_REGION_OVERRIDES:
                culture = LOCALE_REGION_OVERRIDES[locale_key]
                break

        # Fall back to language-only mapping
        if not culture:
            for locale_key in [locale, language]:
                lang_code = locale_key.split("-")[0].lower() if locale_key else ""
                if lang_code in LOCALE_CULTURE_MAP:
                    culture = LOCALE_CULTURE_MAP[lang_code]
                    break

        if not culture:
            return

        try:
            from services.data_graph_service import get_data_graph_service
            dgs = get_data_graph_service()
            dgs.store(kind='user_specific', key='culture_region', value=culture,
                      source='demographic_seeding')

            if language:
                dgs.store(kind='user_specific', key='language_preference', value=language,
                          source='demographic_seeding')

            self._store.setex(CULTURE_SEED_KEY, 86400 * 30, "1")  # Don't re-seed for 30 days
            logging.debug(f"[CLIENT CONTEXT] Seeded culture_region={culture} from locale={locale}")
        except Exception as e:
            logging.debug(f"[CLIENT CONTEXT] Demographic seeding failed: {e}")


