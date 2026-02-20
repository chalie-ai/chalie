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
            url = f"https://wttr.in/{lat},{lon}?format=j1"
            response = requests.get(url, timeout=3)
            response.raise_for_status()
            data = response.json()
            nearest_area = data.get("nearest_area", [])
            if nearest_area:
                area = nearest_area[0]
                city = area.get("areaName", [{}])[0].get("value", "Unknown")
                country = area.get("country", [{}])[0].get("value", "Unknown")
                return f"{city}, {country}"
        except (requests.RequestException, KeyError, ValueError, IndexError) as e:
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

        # Format time in user's timezone
        if local_time := ctx.get("local_time"):
            if timezone := ctx.get("timezone"):
                try:
                    # Parse ISO format time and convert to user's timezone
                    dt = datetime.fromisoformat(local_time.replace("Z", "+00:00"))
                    user_tz = ZoneInfo(timezone)
                    user_dt = dt.astimezone(user_tz)
                    # Format as "03:45 PM, Thursday 20 February 2026"
                    time_str = user_dt.strftime("%I:%M %p, %A %d %B %Y").lstrip("0")
                    parts.append(f"Current time: {time_str}")
                except (ValueError, KeyError) as e:
                    logging.debug(f"[CLIENT CONTEXT] Failed to format time: {e}")
                    parts.append(f"Current time: {local_time}")

        # Format location: only include if resolved name is available (never expose raw coordinates to LLM)
        if location_name := ctx.get("location_name"):
            parts.append(f"Location: {location_name}")

        return " | ".join(parts)
