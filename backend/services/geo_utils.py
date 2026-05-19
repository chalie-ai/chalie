"""Geo utilities — distance, travel time, and speed estimation."""

import logging
import statistics
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_geodesic = None


def _get_geodesic():
    """Return the lazily-imported geodesic distance function."""
    global _geodesic
    if _geodesic is None:
        from geopy.distance import geodesic
        _geodesic = geodesic
    return _geodesic

DEFAULT_SPEED_KMH = 30  # conservative fallback (urban mixed traffic)
_MIN_SPEED_KMH = 1.0   # guard against GPS noise yielding absurd speeds
_MAX_SPEED_KMH = 200.0  # guard against GPS jumps

_PARSE_UTC_SENTINEL = datetime.min.replace(tzinfo=timezone.utc)


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Geodesic distance in kilometres between two points."""
    return _get_geodesic()((lat1, lon1), (lat2, lon2)).km


def estimate_travel_minutes(dist_km: float, speed_kmh: float = DEFAULT_SPEED_KMH) -> float:
    """Estimated travel time in minutes at given speed."""
    if speed_kmh <= 0:
        return 0.0
    return (dist_km / speed_kmh) * 60.0


def estimate_speed_from_history(history_entries: list) -> float | None:
    """Derive average speed in km/h from timestamped location snapshots.

    Each entry is a dict with nested 'location' (lat/lon) and 'saved_at'
    (ISO timestamp string). Returns the median speed across consecutive
    pairs in km/h, or None when fewer than two valid pairs can be derived.
    """
    from services.time_utils import parse_utc

    valid = []
    for entry in history_entries:
        loc = entry.get("location") or {}
        lat = loc.get("lat")
        lon = loc.get("lon")
        ts_raw = (entry.get("saved_at") or "").strip()
        if lat is None or lon is None or not ts_raw:
            continue
        try:
            lat_f = float(lat)
            lon_f = float(lon)
        except (TypeError, ValueError):
            continue
        ts = parse_utc(ts_raw)
        if ts == _PARSE_UTC_SENTINEL:
            continue
        valid.append((ts, lat_f, lon_f))

    if len(valid) < 2:
        return None

    valid.sort(key=lambda t: t[0])
    speeds = []
    for i in range(1, len(valid)):
        prev_ts, prev_lat, prev_lon = valid[i - 1]
        curr_ts, curr_lat, curr_lon = valid[i]
        elapsed_hours = (curr_ts - prev_ts).total_seconds() / 3600.0
        if elapsed_hours <= 0:
            continue
        dist = distance_km(prev_lat, prev_lon, curr_lat, curr_lon)
        spd = dist / elapsed_hours
        if _MIN_SPEED_KMH <= spd <= _MAX_SPEED_KMH:
            speeds.append(spd)

    if not speeds:
        return None

    return statistics.median(speeds)
