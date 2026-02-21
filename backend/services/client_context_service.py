"""
Client Context Service — Stores and retrieves client timezone, location, and system info from Redis.

This provides a single source of truth for the user's timezone and location, accessible
by all services (frontal cortex, scheduler, date_time tool, weather tool, etc.).
"""

import json
import time
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
import requests
from services.redis_client import RedisClientService


REDIS_KEY = "client_context:primary"
TTL = 3600  # 1 hour


class ClientContextService:
    """Manages client context (timezone, location, locale) in Redis."""

    def __init__(self):
        self._redis = RedisClientService.create_connection()

    def _resolve_location_name(self, lat: float, lon: float) -> str | None:
        """
        Resolve location name from lat/lon coordinates using wttr.in API.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            str: "City, Country" format, or None on failure
        """
        try:
            url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json&accept-language=en"
            headers = {"User-Agent": "Chalie/1.0"}
            response = requests.get(url, headers=headers, timeout=3)
            response.raise_for_status()
            data = response.json()
            address = data.get("address", {})
            # Prefer city/town/municipality over finer sub-localities to avoid
            # obscure hamlet names (e.g. Maltese sub-village names with special chars)
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
        Save client context to Redis.

        Args:
            ctx: Dictionary with keys like:
                - timezone: str (e.g., "Europe/Malta")
                - locale: str (e.g., "en-MT")
                - language: str (e.g., "en-GB")
                - local_time: str (ISO format)
                - connection: str (e.g., "4g", "3g")
                - location: dict with lat, lon
        """
        # Resolve location name if location changed significantly
        if location := ctx.get("location"):
            cached_ctx = self.get()
            cached_location = cached_ctx.get("location", {})

            # Check if location changed by more than 0.05 degrees
            lat_changed = abs(location.get("lat", 0) - cached_location.get("lat", 0)) > 0.05
            lon_changed = abs(location.get("lon", 0) - cached_location.get("lon", 0)) > 0.05

            if lat_changed or lon_changed or "location_name" not in cached_ctx:
                location_name = self._resolve_location_name(location["lat"], location["lon"])
                if location_name:
                    ctx["location_name"] = location_name
                    logging.debug(f"[CLIENT CONTEXT] Resolved location: {location_name}")
            else:
                # Preserve existing location_name if coordinates didn't change significantly
                if "location_name" in cached_ctx:
                    ctx["location_name"] = cached_ctx["location_name"]

        ctx["saved_at"] = time.time()
        self._redis.set(REDIS_KEY, json.dumps(ctx), ex=TTL)
        logging.debug(f"[CLIENT CONTEXT] Saved context with timezone={ctx.get('timezone')}, "
                     f"location={ctx.get('location')}")

    def get(self) -> dict:
        """
        Retrieve client context from Redis.

        Returns:
            dict: Client context (empty dict if not found or stale)
        """
        raw = self._redis.get(REDIS_KEY)
        return json.loads(raw) if raw else {}

    def is_stale(self, max_age_seconds: int = 600) -> bool:
        """
        Check if client context is stale (no update for max_age_seconds).

        Args:
            max_age_seconds: Maximum age before considering stale (default 600s = 10 min)

        Returns:
            bool: True if context is missing or older than max_age_seconds
        """
        ctx = self.get()
        saved_at = ctx.get("saved_at", 0)
        is_stale = (time.time() - saved_at) > max_age_seconds
        if is_stale and ctx:
            age = time.time() - saved_at
            logging.debug(f"[CLIENT CONTEXT] Context is stale (age={age:.0f}s, max={max_age_seconds}s)")
        return is_stale

    def format_for_prompt(self) -> str:
        """
        Format client context as human-readable prompt string.

        Returns:
            str: Formatted context (e.g., "Current time: 03:45 PM, Thursday 20 February 2026 | Location: London, United Kingdom")
                 or empty string if not available
        """
        ctx = self.get()
        if not ctx:
            return ""

        parts = []

        # Format time using live server clock in user's timezone (avoids stale heartbeat)
        if timezone := ctx.get("timezone"):
            try:
                user_dt = datetime.now(ZoneInfo(timezone))
                time_str = user_dt.strftime("%I:%M %p, %A %d %B %Y").lstrip("0")
                parts.append(f"Current time: {time_str}")
            except Exception as e:
                logging.debug(f"[CLIENT CONTEXT] Failed to compute time: {e}")

        # Format location: only include if resolved name is available (never expose raw coordinates to LLM)
        if location_name := ctx.get("location_name"):
            parts.append(f"Location: {location_name}")

        return " | ".join(parts)
