"""
Client Context Service — Stores and retrieves client timezone, location, device info,
behavioral signals, and system info from Redis.

This provides a single source of truth for the user's context, accessible
by all services (frontal cortex, scheduler, date_time tool, weather tool, etc.).

Extended with:
- Location history ring buffer for mobility inference
- Place transition detection
- Session re-entry detection
- Demographic trait seeding from locale/location
- Circadian hourly interaction counts
- Rich format_for_prompt with ambient inference
"""

import json
import time
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
import requests
from services.redis_client import RedisClientService


REDIS_KEY = "client_context:primary"
HISTORY_KEY = "client_context:history"
HISTORY_MAX = 12  # ~1hr at 5min intervals
TTL = 3600  # 1 hour

# Session re-entry: user returned after extended absence
REENTRY_KEY = "ambient:session_reentry"
REENTRY_THRESHOLD = 1800  # 30 min
REENTRY_TTL = 300  # 5 min flag

# Place transition
PLACE_TRANSITION_KEY = "ambient:place_transition"
PLACE_TRANSITION_TTL = 300  # 5 min

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
    """Manages client context (timezone, location, device, behavioral signals) in Redis."""

    def __init__(self):
        self._redis = RedisClientService.create_connection()

    def _resolve_location_name(self, lat: float, lon: float) -> str | None:
        """Resolve location name from lat/lon coordinates."""
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
        Save client context to Redis with extended processing.

        Handles: location resolution, behavioral data merging, location history,
        place transition detection, session re-entry, demographic seeding,
        and circadian data collection.
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

            if lat_changed or lon_changed or "location_name" not in cached_ctx:
                location_name = self._resolve_location_name(location["lat"], location["lon"])
                if location_name:
                    ctx["location_name"] = location_name
                    logging.debug(f"[CLIENT CONTEXT] Resolved location: {location_name}")
            else:
                if "location_name" in cached_ctx:
                    ctx["location_name"] = cached_ctx["location_name"]

        # Session re-entry detection
        self._check_session_reentry(cached_ctx)

        # Save primary context
        ctx["saved_at"] = time.time()
        self._redis.set(REDIS_KEY, json.dumps(ctx), ex=TTL)

        # Location history ring buffer (for mobility inference)
        self._push_history(ctx)

        # Place transition detection
        self._detect_place_transition(cached_ctx, ctx)

        # Demographic trait seeding (once per session)
        self._seed_demographic_traits(ctx)

        # Circadian data collection
        self._record_circadian(ctx)

        # Record place fingerprint for learning (after all context is saved)
        self._record_place_fingerprint(ctx)

        logging.debug(f"[CLIENT CONTEXT] Saved context with timezone={ctx.get('timezone')}, "
                     f"device={ctx.get('device', {}).get('class')}")

    def get(self) -> dict:
        """Retrieve client context from Redis."""
        raw = self._redis.get(REDIS_KEY)
        return json.loads(raw) if raw else {}

    def is_stale(self, max_age_seconds: int = 600) -> bool:
        """Check if client context is stale (no update for max_age_seconds)."""
        ctx = self.get()
        saved_at = ctx.get("saved_at", 0)
        is_stale = (time.time() - saved_at) > max_age_seconds
        if is_stale and ctx:
            age = time.time() - saved_at
            logging.debug(f"[CLIENT CONTEXT] Context is stale (age={age:.0f}s, max={max_age_seconds}s)")
        return is_stale

    def format_for_prompt(self) -> str:
        """
        Format client context as human-readable prompt string with ambient inference.

        Uses hedging language for confidence framing:
        "likely" / "probably" / "seems like" — never assertive surveillance language.
        """
        ctx = self.get()
        if not ctx:
            return ""

        parts = []

        # Format time using live server clock in user's timezone
        if timezone := ctx.get("timezone"):
            try:
                user_dt = datetime.now(ZoneInfo(timezone))
                time_str = user_dt.strftime("%I:%M %p, %A %d %B %Y").lstrip("0")
                parts.append(f"Current time: {time_str}")
            except Exception as e:
                logging.debug(f"[CLIENT CONTEXT] Failed to compute time: {e}")

        # Location (never expose raw coordinates to LLM)
        if location_name := ctx.get("location_name"):
            parts.append(f"Location: {location_name}")

        # Device class
        device = ctx.get("device", {})
        if device_class := device.get("class"):
            parts.append(f"Device: {device_class}")

        # Ambient inferences (with confidence framing)
        try:
            from services.ambient_inference_service import AmbientInferenceService
            from services.place_learning_service import PlaceLearningService
            from services.database_service import DatabaseService

            db = DatabaseService()
            place_learning = PlaceLearningService(db)
            inference = AmbientInferenceService(place_learning_service=place_learning)
            inferences = inference.infer(ctx)

            attention = inferences.get("attention")
            place = inferences.get("place")
            energy = inferences.get("energy")

            if attention:
                attention_labels = {
                    "deep_focus": "likely in a focused session",
                    "casual": "seems to be casually browsing",
                    "distracted": "appears to be multitasking",
                    "away": "seems to be away",
                }
                if label := attention_labels.get(attention):
                    parts.append(f"State: {label}")

            if place:
                place_labels = {
                    "home": "probably at home",
                    "work": "probably at work",
                    "transit": "likely in transit",
                    "out": "likely out and about",
                }
                if label := place_labels.get(place):
                    parts.append(f"Context: {label}")

            if energy:
                energy_labels = {
                    "high": "high",
                    "moderate": "moderate",
                    "low": "low",
                }
                if label := energy_labels.get(energy):
                    parts.append(f"Energy: {label}")

        except Exception as e:
            logging.debug(f"[CLIENT CONTEXT] Inference failed: {e}")

        return " | ".join(parts)

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
            self._redis.lpush(HISTORY_KEY, json.dumps(entry))
            self._redis.ltrim(HISTORY_KEY, 0, HISTORY_MAX - 1)
            self._redis.expire(HISTORY_KEY, TTL)
        except Exception as e:
            logging.debug(f"[CLIENT CONTEXT] Failed to push history: {e}")

    # ── Place Transition Detection ─────────────────────────────────────

    def _detect_place_transition(self, old_ctx: dict, new_ctx: dict):
        """Detect place transitions and set Redis flag for downstream consumers."""
        if not old_ctx:
            return

        try:
            from services.ambient_inference_service import AmbientInferenceService
            from services.place_learning_service import PlaceLearningService
            from services.database_service import DatabaseService

            db = DatabaseService()
            place_learning = PlaceLearningService(db)
            inference = AmbientInferenceService(place_learning_service=place_learning)
            old_place = inference._infer_place(old_ctx)
            new_place = inference._infer_place(new_ctx)

            if old_place and new_place and old_place != new_place:
                transition = json.dumps({
                    "from": old_place,
                    "to": new_place,
                    "at": time.time(),
                })
                self._redis.setex(PLACE_TRANSITION_KEY, PLACE_TRANSITION_TTL, transition)
                logging.debug(f"[CLIENT CONTEXT] Place transition: {old_place} → {new_place}")
        except Exception as e:
            logging.debug(f"[CLIENT CONTEXT] Place transition detection failed: {e}")

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
                self._redis.setex(REENTRY_KEY, REENTRY_TTL, json.dumps({
                    "absent_seconds": int(age),
                    "returned_at": time.time(),
                }))
                logging.debug(f"[CLIENT CONTEXT] Session re-entry detected (absent {age:.0f}s)")
            except Exception as e:
                logging.debug(f"[CLIENT CONTEXT] Re-entry flag failed: {e}")

    def is_session_reentry(self) -> bool:
        """Check if the user just returned from an extended absence."""
        return bool(self._redis.get(REENTRY_KEY))

    # ── Demographic Trait Seeding ──────────────────────────────────────

    def _seed_demographic_traits(self, ctx: dict):
        """
        Seed culture-region trait from locale/location (Possible tier).
        Runs once — subsequent reinforcement comes from conversation.
        Religion, gender, and age are NEVER telemetry-seeded.
        """
        # Only seed once
        if self._redis.get(CULTURE_SEED_KEY):
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
            from services.user_trait_service import UserTraitService
            from services.database_service import DatabaseService

            db = DatabaseService()
            trait_service = UserTraitService(db)
            trait_service.store_trait(
                trait_key="culture_region",
                trait_value=culture,
                confidence=0.3,  # Possible tier
                category="core",
                source="inferred",
                is_literal=True,
            )

            # Also seed language preference
            if language:
                trait_service.store_trait(
                    trait_key="language_preference",
                    trait_value=language,
                    confidence=0.5,
                    category="core",
                    source="inferred",
                    is_literal=True,
                )

            self._redis.setex(CULTURE_SEED_KEY, 86400 * 30, "1")  # Don't re-seed for 30 days
            logging.debug(f"[CLIENT CONTEXT] Seeded culture_region={culture} from locale={locale}")
        except Exception as e:
            logging.debug(f"[CLIENT CONTEXT] Demographic seeding failed: {e}")

    # ── Circadian Data Collection ──────────────────────────────────────

    def _record_circadian(self, ctx: dict):
        """
        Store hourly interaction count in Redis for future circadian analysis.
        Passive — no inference yet.
        """
        timezone = ctx.get("timezone")
        if not timezone:
            return

        try:
            user_dt = datetime.now(ZoneInfo(timezone))
            day_of_week = user_dt.weekday()  # 0=Monday
            hour = user_dt.hour
            key = f"ambient:circadian:{day_of_week}:{hour}"
            self._redis.incr(key)
            # 7-day rolling window
            self._redis.expire(key, 86400 * 7)
        except Exception as e:
            logging.debug(f"[CLIENT CONTEXT] Circadian recording failed: {e}")

    # ── Place Fingerprint Recording ────────────────────────────────────

    def _record_place_fingerprint(self, ctx: dict):
        """Record place observation for place learning service."""
        try:
            from services.ambient_inference_service import AmbientInferenceService
            from services.place_learning_service import PlaceLearningService
            from services.database_service import DatabaseService

            db = DatabaseService()
            place_learning = PlaceLearningService(db)
            inference = AmbientInferenceService(place_learning_service=place_learning)
            place = inference._infer_place(ctx)
            if place:
                place_learning.record(ctx, place)
        except Exception as e:
            logging.debug(f"[CLIENT CONTEXT] Place fingerprint recording failed: {e}")
