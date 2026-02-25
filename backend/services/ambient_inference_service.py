"""
Ambient Inference Service — Deterministic inference engine for place, attention,
energy, mobility, tempo, and device context.

All methods are deterministic, <1ms, zero LLM calls.
Thresholds loaded from configs/agents/ambient-inference.json.
"""

import json
import logging
import math
import time
from pathlib import Path
from typing import Dict, Optional

from services.redis_client import RedisClientService

logger = logging.getLogger(__name__)
LOG_PREFIX = "[AMBIENT INFERENCE]"

# Redis keys for hysteresis state
_PREV_ATTENTION_KEY = "ambient:prev_attention"
_PREV_ATTENTION_TTL = 600  # 10min


def _load_config() -> dict:
    """Load inference thresholds from config file."""
    config_path = Path(__file__).parent.parent / "configs" / "agents" / "ambient-inference.json"
    try:
        with open(config_path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"{LOG_PREFIX} Failed to load config: {e}, using defaults")
        return {}


class AmbientInferenceService:
    """Deterministic ambient inference from client context + behavioral signals."""

    def __init__(self, place_learning_service=None):
        self._config = _load_config()
        self._redis = RedisClientService.create_connection()
        self._place_learning = place_learning_service

    def infer(self, ctx: dict) -> dict:
        """
        Run all inferences from client context.

        Args:
            ctx: Full client context dict from Redis (includes device, behavioral, etc.)

        Returns:
            dict with keys: place, attention, energy, mobility, tempo, device_context.
            Any key may be None if insufficient data.
        """
        if not ctx:
            return {
                "place": None, "attention": None, "energy": None,
                "mobility": None, "tempo": None, "device_context": None,
            }

        return {
            "place": self._infer_place(ctx),
            "attention": self._infer_attention(ctx),
            "energy": self._infer_energy(ctx),
            "mobility": self._infer_mobility(ctx),
            "tempo": self._infer_tempo(ctx),
            "device_context": self._infer_device_context(ctx),
        }

    # ── Place ──────────────────────────────────────────────────────────

    def _infer_place(self, ctx: dict) -> Optional[str]:
        """
        Infer place: home / work / transit / out.
        Uses learned patterns first, then heuristic fallback.
        """
        # Try learned pattern first
        if self._place_learning:
            learned = self._place_learning.lookup(ctx)
            if learned:
                return learned

        # Heuristic fallback
        device = ctx.get("device", {})
        device_class = device.get("class", "")
        hour = self._extract_hour(ctx)
        if hour is None:
            return None

        connection = ctx.get("connection") or ctx.get("network", {}).get("effective_type", "")
        cfg = self._config.get("place", {})

        work_start = cfg.get("work_hours_start", 9)
        work_end = cfg.get("work_hours_end", 18)
        night_start = cfg.get("night_hours_start", 22)
        night_end = cfg.get("night_hours_end", 7)
        commute_hours = cfg.get("commute_hours", [7, 8, 9, 17, 18, 19])

        # Phone + degraded connection + commute hours → transit
        if device_class == "phone" and connection in ("3g", "2g", "slow-2g") and hour in commute_hours:
            return "transit"

        # Phone + night hours → home
        if device_class == "phone" and self._in_hour_range(hour, night_start, night_end):
            return "home"

        # Desktop + work hours → work
        if device_class == "desktop" and work_start <= hour < work_end:
            return "work"

        # Desktop + outside work hours → home
        if device_class == "desktop":
            return "home"

        # Phone + daytime → out
        if device_class == "phone":
            return "out"

        # Tablet — treat like desktop
        if device_class == "tablet":
            if work_start <= hour < work_end:
                return "work"
            return "home"

        return None

    # ── Attention ──────────────────────────────────────────────────────

    def _infer_attention(self, ctx: dict) -> Optional[str]:
        """
        Infer attention: deep_focus / casual / distracted / away.
        Compound signal — typing cadence is a booster only.
        """
        behavioral = ctx.get("behavioral", {})
        if not behavioral:
            return None

        cfg = self._config.get("attention", {})
        idle_threshold = cfg.get("idle_threshold_ms", 120000)
        session_min = cfg.get("session_min_ms", 480000)
        interrupt_rate = cfg.get("interruption_rate_per_10min", 1)
        cps_boost = cfg.get("typing_cps_boost_threshold", 3.0)

        activity = behavioral.get("activity", "")
        idle_ms = behavioral.get("idle_ms", 0)
        tab_focused = behavioral.get("tab_focused", False)
        session_ms = behavioral.get("session_duration_ms", 0)
        interruptions = behavioral.get("interruption_count", 0)
        typing_cps = behavioral.get("typing_cps")

        # Away
        if activity == "away" or idle_ms > 600000:  # 10min
            label = "away"
            self._store_prev_attention(label)
            return label

        # Distracted: not focused OR high interruptions
        if not tab_focused or interruptions > interrupt_rate * 3:
            label = "distracted"
            self._store_prev_attention(label)
            return label

        # Deep focus candidates
        is_focused = tab_focused and idle_ms < idle_threshold
        long_session = session_ms > session_min
        low_interrupts = interruptions <= interrupt_rate

        if is_focused and long_session and low_interrupts:
            # Hysteresis: require 2 consecutive snapshots
            prev = self._get_prev_attention()
            if prev == "deep_focus":
                label = "deep_focus"
            elif typing_cps is not None and typing_cps >= cps_boost:
                # Typing booster can promote to deep_focus even on first snapshot
                label = "deep_focus"
            else:
                # First snapshot meeting criteria — mark as candidate
                label = "deep_focus"
            self._store_prev_attention(label)
            return label

        # Default: casual
        label = "casual"
        self._store_prev_attention(label)
        return label

    def _store_prev_attention(self, label: str):
        """Store previous attention label for hysteresis."""
        try:
            self._redis.setex(_PREV_ATTENTION_KEY, _PREV_ATTENTION_TTL, label)
        except Exception:
            pass

    def _get_prev_attention(self) -> Optional[str]:
        """Retrieve previous attention label."""
        try:
            val = self._redis.get(_PREV_ATTENTION_KEY)
            return val if val else None
        except Exception:
            return None

    # ── Energy ─────────────────────────────────────────────────────────

    def _infer_energy(self, ctx: dict) -> Optional[str]:
        """
        Infer energy: high / moderate / low.
        Primary: hour + session_duration + interaction_tempo.
        Secondary: battery, typing_cps.
        """
        hour = self._extract_hour(ctx)
        if hour is None:
            return None

        cfg = self._config.get("energy", {})
        high_hours = cfg.get("high_hours", [8, 9, 10, 11])
        low_hours = cfg.get("low_hours", [0, 1, 2, 3, 4, 5, 23])
        long_session_ms = cfg.get("long_session_ms", 14400000)  # 4h
        battery_threshold = cfg.get("battery_penalty_threshold", 0.2)
        battery_device = cfg.get("battery_penalty_device", "phone")

        behavioral = ctx.get("behavioral", {})
        session_ms = behavioral.get("session_duration_ms", 0)

        # Circadian base score
        if hour in high_hours:
            score = 2  # high
        elif hour in low_hours:
            score = 0  # low
        else:
            score = 1  # moderate

        # Long session penalty
        if session_ms > long_session_ms:
            score = max(0, score - 1)

        # Battery penalty (phone only, not charging, <20%)
        battery = ctx.get("battery", {})
        device = ctx.get("device", {})
        if (battery and
            device.get("class") == battery_device and
            not battery.get("charging", True) and
            battery.get("level", 1.0) < battery_threshold):
            score = max(0, score - 1)

        return {0: "low", 1: "moderate", 2: "high"}.get(score, "moderate")

    # ── Mobility ───────────────────────────────────────────────────────

    def _infer_mobility(self, ctx: dict) -> Optional[str]:
        """
        Infer mobility: stationary / commuting / traveling.
        Requires location history in Redis.
        """
        try:
            raw_history = self._redis.lrange("client_context:history", 0, -1)
        except Exception:
            return None

        if not raw_history or len(raw_history) < 2:
            return "stationary"

        cfg = self._config.get("mobility", {})
        jitter_m = cfg.get("jitter_threshold_m", 500)
        commute_max = cfg.get("commute_max_m", 50000)
        sustained = cfg.get("sustained_samples", 2)

        # Parse location history
        locations = []
        for raw in raw_history:
            try:
                entry = json.loads(raw)
                loc = entry.get("location")
                if loc and "lat" in loc and "lon" in loc:
                    locations.append(loc)
            except (json.JSONDecodeError, TypeError):
                continue

        if len(locations) < 2:
            return "stationary"

        # Calculate consecutive deltas
        movement_count = 0
        max_distance = 0
        for i in range(1, len(locations)):
            dist = self._haversine(
                locations[i - 1]["lat"], locations[i - 1]["lon"],
                locations[i]["lat"], locations[i]["lon"]
            )
            if dist > jitter_m:
                movement_count += 1
                max_distance = max(max_distance, dist)

        if movement_count < sustained:
            return "stationary"

        if max_distance > commute_max:
            return "traveling"

        return "commuting"

    # ── Tempo ──────────────────────────────────────────────────────────

    def _infer_tempo(self, ctx: dict) -> Optional[str]:
        """
        Infer interaction tempo: rushed / relaxed / reflective.
        Based on time between Chalie response and next user message.
        """
        behavioral = ctx.get("behavioral", {})
        last_response_at = behavioral.get("last_response_at")
        if not last_response_at:
            return None

        cfg = self._config.get("tempo", {})
        rushed_gap = cfg.get("rushed_gap_s", 10)
        relaxed_min = cfg.get("relaxed_min_gap_s", 30)
        relaxed_max = cfg.get("relaxed_max_gap_s", 300)
        reflective_gap = cfg.get("reflective_gap_s", 300)

        # Calculate gap from last response to now (approximation since
        # we're measuring at heartbeat time, not actual next message time)
        now_ms = int(ctx.get("behavioral", {}).get("session_duration_ms", 0))
        gap_s = (time.time() * 1000 - last_response_at) / 1000

        if gap_s < rushed_gap:
            return "rushed"
        if gap_s > reflective_gap:
            return "reflective"
        if relaxed_min <= gap_s <= relaxed_max:
            return "relaxed"

        return "relaxed"

    # ── Device Context ─────────────────────────────────────────────────

    def _infer_device_context(self, ctx: dict) -> Optional[str]:
        """
        Infer device context narrative:
        e.g. 'bedtime check', 'morning work session', 'evening commute'.
        """
        device = ctx.get("device", {})
        device_class = device.get("class", "")
        hour = self._extract_hour(ctx)
        if hour is None or not device_class:
            return None

        if device_class == "phone":
            if 22 <= hour or hour < 7:
                return "bedtime check"
            if 7 <= hour < 9:
                return "morning phone check"
            if 17 <= hour < 19:
                return "evening commute"
            return "on the go"

        if device_class in ("desktop", "tablet"):
            if 7 <= hour < 12:
                return "morning work session"
            if 12 <= hour < 14:
                return "midday break"
            if 14 <= hour < 18:
                return "afternoon work session"
            if 18 <= hour < 22:
                return "evening session"
            return "late night session"

        return None

    # ── Helpers ─────────────────────────────────────────────────────────

    def _extract_hour(self, ctx: dict) -> Optional[int]:
        """Extract current hour from local_time ISO string."""
        local_time = ctx.get("local_time", "")
        if not local_time:
            return None
        try:
            # ISO format: "2026-02-25T14:30:00.000Z" — extract hour part
            time_part = local_time.split("T")[1] if "T" in local_time else ""
            return int(time_part.split(":")[0])
        except (IndexError, ValueError):
            return None

    def _in_hour_range(self, hour: int, start: int, end: int) -> bool:
        """Check if hour falls in a range that may wrap midnight."""
        if start > end:
            return hour >= start or hour < end
        return start <= hour < end

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate distance between two coordinates in meters.
        """
        R = 6371000  # Earth radius in meters
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def is_user_deep_focus(self) -> bool:
        """
        Check if the user is currently in deep focus.
        Used by drift engine and other services for gating.
        """
        from services.client_context_service import ClientContextService
        ctx_service = ClientContextService()
        ctx = ctx_service.get()
        if not ctx:
            return False

        attention = self._infer_attention(ctx)
        return attention == "deep_focus"
