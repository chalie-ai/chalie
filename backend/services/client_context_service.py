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
  - demographic traits → data_graph

Locale fields (timezone, locale, language, currency, location) are read
exclusively via ``services.locale_service`` — never directly from this service.
"""

import json
import logging
import time
from services.memory_client import MemoryClientService

_NOMINATIM_USER_AGENT = "Chalie/1.0"
_NOMINATIM_TIMEOUT_S = 3
_nominatim = None


def _get_nominatim():
    """Return a lazily-initialised Nominatim geocoder singleton."""
    global _nominatim
    if _nominatim is None:
        from geopy.geocoders import Nominatim
        _nominatim = Nominatim(
            user_agent=_NOMINATIM_USER_AGENT, timeout=_NOMINATIM_TIMEOUT_S,
        )
    return _nominatim


HISTORY_KEY = "client_context:history"
HISTORY_MAX = 12  # ~1hr at 5min intervals
TTL = 3600  # 1 hour (used by ephemeral MemoryStore keys, not telemetry)

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

    Telemetry persistence is delegated to ``TelemetryCacheService``.
    MemoryStore is retained for ephemeral inference flags (place-transition,
    session-reentry, culture-seed) and the location-history ring buffer.
    """

    def __init__(self):
        """Initialize the service and open a MemoryStore connection."""
        self._store = MemoryClientService.create_connection()

    def _resolve_location_name(self, lat: float, lon: float) -> str | None:
        """Resolve a human-readable city/country name from coordinates.

        Uses the geopy Nominatim geocoder (OpenStreetMap). Prefers
        city → town → municipality → county → state_district as the locality
        label, combined with the country name.

        Args:
            lat: Latitude in decimal degrees.
            lon: Longitude in decimal degrees.

        Returns:
            A string such as ``"Valletta, Malta"`` on success, or ``None`` if
            the geocoder call fails or returns an unusable address.
        """
        try:
            geocoder = _get_nominatim()
            location = geocoder.reverse((lat, lon), language="en", exactly_one=True)
            if location is None:
                return None
            address = location.raw.get("address", {})
            city = (address.get("city") or address.get("town") or
                    address.get("municipality") or address.get("county") or
                    address.get("state_district") or "")
            country = address.get("country", "")
            if city and country:
                return f"{city}, {country}"
            if country:
                return country
        except (KeyError, ValueError, AttributeError) as e:
            logging.debug(f"[CLIENT CONTEXT] Failed to resolve location: {e}")
        except Exception as e:
            logging.warning(f"[CLIENT CONTEXT] Unexpected error resolving location: {e}")
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
            # ``cached_ctx.get("X", {})`` returns ``None`` when the key exists
            # with value ``None`` — which it will, because ``_flatten`` JSON-
            # encodes ``None`` as ``"null"`` and ``_unflatten`` decodes it back
            # to ``None``. Coerce explicitly so ``.get()`` chaining is safe.
            cached_location = cached_ctx.get("location") or {}
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


        from services.heartbeat_service import heartbeat_service
        heartbeat_service.write(ctx)

        # Location history ring buffer (for mobility inference)
        self._push_history(ctx)

        # Demographic trait seeding (once per session)
        self._seed_demographic_traits(ctx)

        logging.debug(f"[CLIENT CONTEXT] Saved context with timezone={ctx.get('timezone')}, "
                     f"device={(ctx.get('device') or {}).get('class')}")

    def get(self) -> dict:
        """Retrieve client context from the heartbeat cache."""
        from services.heartbeat_service import heartbeat_service
        return heartbeat_service.read()

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


