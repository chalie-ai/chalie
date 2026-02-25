"""
Place Learning Service — Accumulates place fingerprints over time so
inference improves beyond heuristics.

Uses geohash (~1km precision) for privacy: raw coordinates are never stored
in PostgreSQL.
"""

import hashlib
import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)
LOG_PREFIX = "[PLACE LEARNING]"

LEARNED_THRESHOLD = 20  # minimum observations before learned label overrides heuristic


class PlaceLearningService:
    """Accumulates and looks up learned place fingerprints in PostgreSQL."""

    def __init__(self, database_service):
        self.db = database_service

    def record(self, ctx: dict, place_label: str):
        """
        Record a place observation by upserting a fingerprint.

        Args:
            ctx: Client context dict (contains device, local_time, location, connection, network)
            place_label: Inferred place label (home/work/transit/out)
        """
        fp_hash = self._build_fingerprint(ctx)
        if not fp_hash:
            return

        device_class = ctx.get("device", {}).get("class", "unknown")
        hour_bucket = self._hour_to_bucket(ctx)
        location_hash = self._geohash(ctx)
        connection_type = ctx.get("connection") or ctx.get("network", {}).get("effective_type", "")

        try:
            self.db.execute(
                """
                INSERT INTO place_fingerprints
                    (fingerprint_hash, device_class, hour_bucket, location_hash,
                     connection_type, place_label, count, last_seen_at)
                VALUES (%s, %s, %s, %s, %s, %s, 1, NOW())
                ON CONFLICT (fingerprint_hash)
                DO UPDATE SET
                    count = place_fingerprints.count + 1,
                    place_label = CASE
                        WHEN place_fingerprints.count + 1 > place_fingerprints.count
                        THEN %s
                        ELSE place_fingerprints.place_label
                    END,
                    last_seen_at = NOW()
                """,
                (fp_hash, device_class, hour_bucket, location_hash,
                 connection_type, place_label, place_label)
            )
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Failed to record fingerprint: {e}")

    def lookup(self, ctx: dict) -> Optional[str]:
        """
        Look up a learned place label for the current context.

        Returns:
            str: Learned place label if count >= threshold, else None.
        """
        fp_hash = self._build_fingerprint(ctx)
        if not fp_hash:
            return None

        try:
            rows = self.db.fetch_all(
                """
                SELECT place_label, count
                FROM place_fingerprints
                WHERE fingerprint_hash = %s AND count >= %s
                LIMIT 1
                """,
                (fp_hash, LEARNED_THRESHOLD)
            )
            if rows:
                return rows[0]["place_label"]
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} Failed to lookup fingerprint: {e}")

        return None

    def _build_fingerprint(self, ctx: dict) -> Optional[str]:
        """
        Build a deterministic fingerprint hash from context signals.

        Components: device_class + hour_bucket + location_hash + connection_type
        """
        device_class = ctx.get("device", {}).get("class", "")
        if not device_class:
            return None

        hour_bucket = self._hour_to_bucket(ctx)
        if hour_bucket is None:
            return None

        location_hash = self._geohash(ctx) or "none"
        connection_type = ctx.get("connection") or ctx.get("network", {}).get("effective_type", "none")

        key = f"{device_class}:{hour_bucket}:{location_hash}:{connection_type}"
        return hashlib.sha256(key.encode()).hexdigest()[:32]

    def _hour_to_bucket(self, ctx: dict) -> Optional[int]:
        """Convert local hour to 3-hour bucket (0-7)."""
        local_time = ctx.get("local_time", "")
        if not local_time:
            return None
        try:
            time_part = local_time.split("T")[1] if "T" in local_time else ""
            hour = int(time_part.split(":")[0])
            return hour // 3
        except (IndexError, ValueError):
            return None

    def _geohash(self, ctx: dict) -> Optional[str]:
        """
        Quantize lat/lon to ~1km precision geohash.
        Raw coordinates are NEVER stored — only this quantized hash.
        """
        location = ctx.get("location")
        if not location or "lat" not in location or "lon" not in location:
            return None

        lat = location["lat"]
        lon = location["lon"]

        # Quantize to ~0.01 degree (~1km precision)
        qlat = round(lat, 2)
        qlon = round(lon, 2)
        raw = f"{qlat:.2f},{qlon:.2f}"
        return hashlib.md5(raw.encode()).hexdigest()[:8]
