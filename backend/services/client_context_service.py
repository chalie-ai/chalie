"""Stores and retrieves client timezone, location, device info, behavioral
signals, and system info.

The raw heartbeat payload (whatever the frontend sends) is persisted to the
``telemetry`` table as flat key/value rows, e.g. ``device.name`` →
``"iPhone"``. The frontend (heartbeat.js) is the single source of truth for
which keys are collected; this service blindly persists what it receives
and reconstructs the nested dict on read so existing consumers (time_utils,
scheduler_skill, …) see the same shape they always did.

Side concerns that stay in MemoryStore (NOT telemetry): the location-history
ring buffer (mobility inference) and session re-entry / place-transition
flags (ephemeral inference state).

Locale fields (timezone, locale, language, currency, location) are read
exclusively via ``services.locale_service`` — never directly from this
service.
"""

import json
import logging
import time
from typing import TYPE_CHECKING, Optional, cast

from services.memory_client import MemoryClientService

if TYPE_CHECKING:
    from typing import Protocol

    class _Location(Protocol):
        @property
        def raw(self) -> dict[str, object]:
            ...

    class _Geocoder(Protocol):
        def reverse(self, query: object, language: str = ..., exactly_one: bool = ...) -> "_Location | None":
            ...

_NOMINATIM_USER_AGENT = "Chalie/1.0"
_NOMINATIM_TIMEOUT_S = 3
_nominatim: "Optional[_Geocoder]" = None


def _get_nominatim() -> "_Geocoder":
    """Return a lazily-initialised Nominatim geocoder singleton."""
    global _nominatim
    if _nominatim is None:
        from geopy.geocoders import Nominatim
        _nominatim = cast("_Geocoder", Nominatim(
            user_agent=_NOMINATIM_USER_AGENT, timeout=_NOMINATIM_TIMEOUT_S,
        ))
    return _nominatim


HISTORY_KEY = "client_context:history"
HISTORY_MAX = 12  # ~1hr at 5min intervals
TTL = 3600  # 1 hour (used by ephemeral MemoryStore keys, not telemetry)


class ClientContextService:
    """Manages client context (timezone, location, device, behavioral signals).

    Telemetry persistence is delegated to ``TelemetryCacheService``.
    MemoryStore is retained for ephemeral inference flags (place-transition,
    session-reentry) and the location-history ring buffer.
    """

    def __init__(self) -> None:
        self._store = MemoryClientService.create_connection()

    def _resolve_location_name(self, lat: float, lon: float) -> str | None:
        """Uses the geopy Nominatim geocoder (OpenStreetMap). Prefers
        city → town → municipality → county → state_district as the
        locality label, combined with the country name. Returns ``None``
        on geocoder failure or unusable address."""
        try:
            geocoder = _get_nominatim()
            location = geocoder.reverse((lat, lon), language="en", exactly_one=True)
            if location is None:
                return None
            address = cast(dict[str, object], location.raw.get("address", {}))
            city = cast(str, address.get("city") or address.get("town") or
                    address.get("municipality") or address.get("county") or
                    address.get("state_district") or "")
            country = cast(str, address.get("country", ""))
            if city and country:
                return f"{city}, {country}"
            if country:
                return country
        except (KeyError, ValueError, AttributeError) as e:
            logging.debug(f"[CLIENT CONTEXT] Failed to resolve location: {e}")
        except Exception as e:
            logging.warning(f"[CLIENT CONTEXT] Unexpected error resolving location: {e}")
        return None

    def save(self, ctx: dict[str, object]) -> None:
        """Handles location resolution, behavioral-data merging, location
        history, and session re-entry."""
        cached_ctx = self.get()

        # Merge behavioral data: don't overwrite if new heartbeat lacks it
        if "behavioral" not in ctx and "behavioral" in cached_ctx:
            ctx["behavioral"] = cached_ctx["behavioral"]

        # Resolve location name if location changed significantly
        if location := cast(dict[str, object], ctx.get("location")):
            # ``cached_ctx.get("X", {})`` returns ``None`` when the key exists
            # with value ``None`` — which it will, because ``_flatten`` JSON-
            # encodes ``None`` as ``"null"`` and ``_unflatten`` decodes it back
            # to ``None``. Coerce explicitly so ``.get()`` chaining is safe.
            cached_location = cast(dict[str, object], cached_ctx.get("location") or {})
            lat_changed = abs(cast(float, location.get("lat", 0)) - cast(float, cached_location.get("lat", 0))) > 0.05
            lon_changed = abs(cast(float, location.get("lon", 0)) - cast(float, cached_location.get("lon", 0))) > 0.05

            no_cached_name = "location_name" not in cached_ctx
            cached_stale = cached_ctx.get("_location_name_stale", False)

            if lat_changed or lon_changed or no_cached_name or cached_stale:
                location_name = self._resolve_location_name(cast(float, location["lat"]), cast(float, location["lon"]))
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

        logging.debug(f"[CLIENT CONTEXT] Saved context with timezone={ctx.get('timezone')}, "
                     f"device={cast(dict[str, object], ctx.get('device') or {}).get('class')}")

    def get(self) -> dict[str, object]:
        """Retrieve client context from the heartbeat cache."""
        from services.heartbeat_service import heartbeat_service
        return heartbeat_service.read()

    def is_stale(self, max_age_seconds: int = 600) -> bool:
        """``True`` if the stored context is missing or its ``saved_at`` is
        older than ``max_age_seconds`` (default 600 = 10 min)."""
        ctx = self.get()
        saved_at = cast(float, ctx.get("saved_at", 0))
        is_stale = (time.time() - saved_at) > max_age_seconds
        if is_stale and ctx:
            age = time.time() - saved_at
            logging.debug(f"[CLIENT CONTEXT] Context is stale (age={age:.0f}s, max={max_age_seconds}s)")
        return is_stale

    # ── Location History ───────────────────────────────────────────────

    def _push_history(self, ctx: dict[str, object]) -> None:
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



