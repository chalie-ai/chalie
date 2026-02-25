"""Unit tests for AmbientInferenceService."""
import json
import pytest
from unittest.mock import MagicMock, patch

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_redis():
    """Create a mock Redis for hysteresis state."""
    store = {}

    class FakeRedis:
        def get(self, key):
            return store.get(key)

        def setex(self, key, ttl, value):
            store[key] = value

        def set(self, key, value, **kwargs):
            store[key] = value

        def lrange(self, key, start, end):
            return store.get(key, [])

    fake = FakeRedis()
    fake._store = store
    return fake


@pytest.fixture
def inference_service(mock_redis):
    """Create AmbientInferenceService with mocked Redis."""
    with patch("services.ambient_inference_service.RedisClientService") as mock_rcs:
        mock_rcs.create_connection.return_value = mock_redis
        from services.ambient_inference_service import AmbientInferenceService
        service = AmbientInferenceService(place_learning_service=None)
        service._redis = mock_redis
        return service


# ── Place inference ──────────────────────────────────────────────

class TestPlaceInference:
    def test_desktop_morning_is_work(self, inference_service):
        ctx = {
            "device": {"class": "desktop"},
            "local_time": "2026-02-25T10:30:00.000Z",
        }
        assert inference_service._infer_place(ctx) == "work"

    def test_phone_night_is_home(self, inference_service):
        ctx = {
            "device": {"class": "phone"},
            "local_time": "2026-02-25T23:30:00.000Z",
        }
        assert inference_service._infer_place(ctx) == "home"

    def test_phone_3g_commute_hour_is_transit(self, inference_service):
        ctx = {
            "device": {"class": "phone"},
            "local_time": "2026-02-25T08:15:00.000Z",
            "connection": "3g",
        }
        assert inference_service._infer_place(ctx) == "transit"

    def test_phone_daytime_is_out(self, inference_service):
        ctx = {
            "device": {"class": "phone"},
            "local_time": "2026-02-25T14:00:00.000Z",
            "connection": "4g",
        }
        assert inference_service._infer_place(ctx) == "out"

    def test_desktop_evening_is_home(self, inference_service):
        ctx = {
            "device": {"class": "desktop"},
            "local_time": "2026-02-25T20:00:00.000Z",
        }
        assert inference_service._infer_place(ctx) == "home"

    def test_no_device_returns_none(self, inference_service):
        ctx = {"local_time": "2026-02-25T10:00:00.000Z"}
        assert inference_service._infer_place(ctx) is None

    def test_no_local_time_returns_none(self, inference_service):
        ctx = {"device": {"class": "desktop"}}
        assert inference_service._infer_place(ctx) is None

    def test_tablet_work_hours_is_work(self, inference_service):
        ctx = {
            "device": {"class": "tablet"},
            "local_time": "2026-02-25T11:00:00.000Z",
        }
        assert inference_service._infer_place(ctx) == "work"

    def test_learned_pattern_overrides_heuristic(self, inference_service):
        mock_place_learning = MagicMock()
        mock_place_learning.lookup.return_value = "home"
        inference_service._place_learning = mock_place_learning

        ctx = {
            "device": {"class": "desktop"},
            "local_time": "2026-02-25T10:00:00.000Z",
        }
        # Heuristic would say "work", but learned says "home"
        assert inference_service._infer_place(ctx) == "home"


# ── Attention inference ──────────────────────────────────────────

class TestAttentionInference:
    def test_deep_focus_compound(self, inference_service):
        """Focused + low idle + session >8min + few interruptions → deep_focus."""
        ctx = {
            "behavioral": {
                "activity": "active",
                "idle_ms": 5000,
                "tab_focused": True,
                "session_duration_ms": 600000,  # 10 min
                "interruption_count": 0,
                "typing_cps": 4.0,
            }
        }
        result = inference_service._infer_attention(ctx)
        assert result == "deep_focus"

    def test_fast_typing_alone_not_deep_focus(self, inference_service):
        """Fast typing but tab unfocused → distracted, not deep_focus."""
        ctx = {
            "behavioral": {
                "activity": "active",
                "idle_ms": 1000,
                "tab_focused": False,
                "session_duration_ms": 600000,
                "interruption_count": 0,
                "typing_cps": 8.0,
            }
        }
        result = inference_service._infer_attention(ctx)
        assert result == "distracted"

    def test_away_state(self, inference_service):
        ctx = {
            "behavioral": {
                "activity": "away",
                "idle_ms": 700000,
                "tab_focused": True,
                "session_duration_ms": 600000,
                "interruption_count": 0,
            }
        }
        assert inference_service._infer_attention(ctx) == "away"

    def test_distracted_high_interruptions(self, inference_service):
        ctx = {
            "behavioral": {
                "activity": "active",
                "idle_ms": 1000,
                "tab_focused": True,
                "session_duration_ms": 600000,
                "interruption_count": 10,
            }
        }
        assert inference_service._infer_attention(ctx) == "distracted"

    def test_casual_default(self, inference_service):
        """Focused but short session → casual."""
        ctx = {
            "behavioral": {
                "activity": "active",
                "idle_ms": 5000,
                "tab_focused": True,
                "session_duration_ms": 120000,  # 2 min (below 8min threshold)
                "interruption_count": 0,
            }
        }
        assert inference_service._infer_attention(ctx) == "casual"

    def test_no_behavioral_returns_none(self, inference_service):
        assert inference_service._infer_attention({}) is None


# ── Energy inference ─────────────────────────────────────────────

class TestEnergyInference:
    def test_morning_is_high(self, inference_service):
        ctx = {"local_time": "2026-02-25T09:00:00.000Z"}
        assert inference_service._infer_energy(ctx) == "high"

    def test_late_night_is_low(self, inference_service):
        ctx = {"local_time": "2026-02-25T02:00:00.000Z"}
        assert inference_service._infer_energy(ctx) == "low"

    def test_afternoon_is_moderate(self, inference_service):
        ctx = {"local_time": "2026-02-25T15:00:00.000Z"}
        assert inference_service._infer_energy(ctx) == "moderate"

    def test_late_night_long_session_is_low(self, inference_service):
        ctx = {
            "local_time": "2026-02-25T23:00:00.000Z",
            "behavioral": {"session_duration_ms": 18000000},  # 5h
        }
        assert inference_service._infer_energy(ctx) == "low"

    def test_low_battery_phone_reduces_energy(self, inference_service):
        ctx = {
            "local_time": "2026-02-25T15:00:00.000Z",  # moderate baseline
            "battery": {"level": 0.1, "charging": False},
            "device": {"class": "phone"},
            "behavioral": {},
        }
        assert inference_service._infer_energy(ctx) == "low"

    def test_low_battery_desktop_no_penalty(self, inference_service):
        ctx = {
            "local_time": "2026-02-25T15:00:00.000Z",  # moderate baseline
            "battery": {"level": 0.1, "charging": False},
            "device": {"class": "desktop"},
            "behavioral": {},
        }
        assert inference_service._infer_energy(ctx) == "moderate"

    def test_no_time_returns_none(self, inference_service):
        assert inference_service._infer_energy({}) is None


# ── Mobility inference ───────────────────────────────────────────

class TestMobilityInference:
    def test_jitter_stays_stationary(self, inference_service, mock_redis):
        """Movement <500m → still stationary."""
        # Two locations ~100m apart
        mock_redis._store["client_context:history"] = [
            json.dumps({"location": {"lat": 35.9000, "lon": 14.5000}}),
            json.dumps({"location": {"lat": 35.9008, "lon": 14.5008}}),
        ]
        assert inference_service._infer_mobility({}) == "stationary"

    def test_sustained_movement_is_commuting(self, inference_service, mock_redis):
        """Sustained >2km over 2+ samples → commuting."""
        mock_redis._store["client_context:history"] = [
            json.dumps({"location": {"lat": 35.90, "lon": 14.50}}),
            json.dumps({"location": {"lat": 35.92, "lon": 14.52}}),
            json.dumps({"location": {"lat": 35.95, "lon": 14.55}}),
        ]
        result = inference_service._infer_mobility({})
        assert result == "commuting"

    def test_no_history_is_stationary(self, inference_service, mock_redis):
        assert inference_service._infer_mobility({}) == "stationary"

    def test_large_distance_is_traveling(self, inference_service, mock_redis):
        """Movement >50km → traveling."""
        mock_redis._store["client_context:history"] = [
            json.dumps({"location": {"lat": 35.90, "lon": 14.50}}),
            json.dumps({"location": {"lat": 36.50, "lon": 15.10}}),
            json.dumps({"location": {"lat": 37.00, "lon": 15.50}}),
        ]
        result = inference_service._infer_mobility({})
        assert result == "traveling"


# ── Tempo inference ──────────────────────────────────────────────

class TestTempoInference:
    def test_no_response_returns_none(self, inference_service):
        ctx = {"behavioral": {}}
        assert inference_service._infer_tempo(ctx) is None

    def test_recent_response_rushed(self, inference_service):
        import time
        ctx = {
            "behavioral": {
                "last_response_at": time.time() * 1000 - 3000,  # 3s ago
                "session_duration_ms": 600000,
            }
        }
        assert inference_service._infer_tempo(ctx) == "rushed"

    def test_old_response_reflective(self, inference_service):
        import time
        ctx = {
            "behavioral": {
                "last_response_at": time.time() * 1000 - 600000,  # 10min ago
                "session_duration_ms": 600000,
            }
        }
        assert inference_service._infer_tempo(ctx) == "reflective"


# ── Device context inference ─────────────────────────────────────

class TestDeviceContextInference:
    def test_phone_late_night_bedtime(self, inference_service):
        ctx = {
            "device": {"class": "phone"},
            "local_time": "2026-02-25T23:30:00.000Z",
        }
        assert inference_service._infer_device_context(ctx) == "bedtime check"

    def test_desktop_morning_work(self, inference_service):
        ctx = {
            "device": {"class": "desktop"},
            "local_time": "2026-02-25T09:00:00.000Z",
        }
        assert inference_service._infer_device_context(ctx) == "morning work session"


# ── Full inference ───────────────────────────────────────────────

class TestFullInference:
    def test_empty_context_all_none(self, inference_service):
        result = inference_service.infer({})
        assert result["place"] is None
        assert result["attention"] is None
        assert result["energy"] is None
        assert result["mobility"] is None
        assert result["tempo"] is None
        assert result["device_context"] is None

    def test_full_context_returns_all(self, inference_service, mock_redis):
        import time
        ctx = {
            "device": {"class": "desktop"},
            "local_time": "2026-02-25T10:00:00.000Z",
            "connection": "4g",
            "behavioral": {
                "activity": "active",
                "idle_ms": 3000,
                "tab_focused": True,
                "session_duration_ms": 600000,
                "interruption_count": 0,
                "typing_cps": 4.0,
                "last_response_at": time.time() * 1000 - 60000,
            },
        }
        result = inference_service.infer(ctx)
        assert result["place"] == "work"
        assert result["attention"] == "deep_focus"
        assert result["energy"] == "high"
        assert result["device_context"] == "morning work session"


# ── Session re-entry (via client_context_service) ────────────────

class TestSessionReentry:
    def test_stale_context_sets_reentry(self):
        import time
        store = {}

        class FakeRedis:
            def get(self, key):
                raw = store.get(key)
                return raw

            def set(self, key, value, **kwargs):
                store[key] = value

            def setex(self, key, ttl, value):
                store[key] = value

            def incr(self, key):
                store[key] = int(store.get(key, 0)) + 1

            def expire(self, key, ttl):
                pass

            def lpush(self, key, *values):
                if key not in store:
                    store[key] = []
                for v in values:
                    store[key].insert(0, v)

            def ltrim(self, key, start, end):
                if key in store:
                    store[key] = store[key][start:end + 1]

        with patch("services.client_context_service.RedisClientService") as mock_rcs:
            mock_rcs.create_connection.return_value = FakeRedis()
            from services.client_context_service import ClientContextService
            service = ClientContextService()
            service._redis = FakeRedis()

            # Simulate stale context (saved 40min ago)
            old_ctx = {"saved_at": time.time() - 2400, "timezone": "UTC"}
            store["client_context:primary"] = json.dumps(old_ctx)

            # Save new context — should detect re-entry
            service.save({"timezone": "UTC", "local_time": "2026-02-25T10:00:00.000Z"})

            # Check re-entry flag was set
            assert "ambient:session_reentry" in store


# ── Graceful degradation ─────────────────────────────────────────

class TestGracefulDegradation:
    def test_missing_behavioral_returns_none_attention(self, inference_service):
        ctx = {"device": {"class": "desktop"}}
        assert inference_service._infer_attention(ctx) is None

    def test_missing_time_returns_none_energy(self, inference_service):
        ctx = {"device": {"class": "desktop"}}
        assert inference_service._infer_energy(ctx) is None

    def test_malformed_time_returns_none(self, inference_service):
        ctx = {"local_time": "not-a-time", "device": {"class": "desktop"}}
        assert inference_service._infer_place(ctx) is None
