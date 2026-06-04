"""Feature tests for the TKT-557 geo-tagging components.

Three services are covered:

* geo_utils          — pure distance / travel / speed functions
* GeoPatternProcessor — LLM processor that extracts location-based patterns
* PlaceAbility        — save/list/get/delete named places via data_graph

All tests run against real production code with the real DB (via the `db`
fixture from conftest.py which uses SchemaConvergenceService). No mocks of
in-process code.

geopy is required. Tests are skipped if the package is absent so the CI
baseline is not broken in environments that haven't installed it yet.
"""

from datetime import datetime, timedelta, timezone

import pytest

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# geopy availability guard
# ---------------------------------------------------------------------------

try:
    from geopy.distance import geodesic as _geopy_check  # noqa: F401
    _GEOPY_AVAILABLE = True
except ImportError:
    _GEOPY_AVAILABLE = False

_requires_geopy = pytest.mark.skipif(
    not _GEOPY_AVAILABLE,
    reason="geopy not installed — install it to run geo_utils tests",
)


# ---------------------------------------------------------------------------
# Test 1: geo_utils.distance_km — real geodesic distance
# ---------------------------------------------------------------------------

class TestDistanceKm:
    """distance_km returns the geodesic distance between two real coordinates."""

    @_requires_geopy
    def test_valletta_to_sliema_approx_2km(self):
        """Valletta (35.8989, 14.5146) → Sliema (35.9121, 14.5013) ≈ 1.5–3 km."""
        from services.geo_utils import distance_km

        dist = distance_km(35.8989, 14.5146, 35.9121, 14.5013)

        assert 1.5 <= dist <= 3.0, (
            f"Expected Valletta→Sliema distance between 1.5 and 3.0 km, got {dist:.4f} km"
        )

    @_requires_geopy
    def test_same_point_is_zero(self):
        """Identical coordinates return 0."""
        from services.geo_utils import distance_km

        dist = distance_km(35.8989, 14.5146, 35.8989, 14.5146)

        assert dist == 0.0


# ---------------------------------------------------------------------------
# Test 2: geo_utils.estimate_travel_minutes
# ---------------------------------------------------------------------------

class TestEstimateTravelMinutes:
    """estimate_travel_minutes converts distance and speed to minutes."""

    @_requires_geopy
    def test_zero_speed_returns_zero(self):
        """Zero speed guard returns 0.0 rather than raising ZeroDivisionError."""
        from services.geo_utils import estimate_travel_minutes

        minutes = estimate_travel_minutes(10.0, speed_kmh=0.0)

        assert minutes == 0.0


# ---------------------------------------------------------------------------
# Test 3: geo_utils.estimate_speed_from_history
# ---------------------------------------------------------------------------

class TestEstimateSpeedFromHistory:
    """estimate_speed_from_history derives median speed from location snapshots."""

    def _entry(self, lat, lon, ts):
        return {"location": {"lat": lat, "lon": lon}, "saved_at": ts.isoformat()}

    @_requires_geopy
    def test_valid_entries_return_plausible_speed(self):
        """Two points ~1.9 km apart with 4-minute gap → speed near 28–29 km/h."""
        from services.geo_utils import estimate_speed_from_history

        t0 = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        t1 = t0 + timedelta(minutes=4)

        speed = estimate_speed_from_history([
            self._entry(35.8989, 14.5146, t0),
            self._entry(35.9121, 14.5013, t1),
        ])

        assert speed is not None
        # Valletta–Sliema at ~1.9 km in 4 min ≈ 28.5 km/h
        assert 20.0 <= speed <= 45.0, f"Expected speed 20–45 km/h, got {speed:.2f}"

    @_requires_geopy
    @pytest.mark.parametrize("entries_fn,reason", [
        (lambda s: [], "empty list"),
        (lambda s: [s._entry(35.8989, 14.5146, datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc))], "single entry"),
        (lambda s: [s._entry(0.0, 0.0, datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)),
                    s._entry(1.0, 0.0, datetime(2026, 1, 1, 10, 0, 1, tzinfo=timezone.utc))], "gps noise above max speed"),
        (lambda s: [s._entry(35.8989, 14.5146, datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)),
                    s._entry(35.9121, 14.5013, datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc))], "identical timestamps"),
    ], ids=["empty", "single", "gps_noise", "same_ts"])
    def test_degenerate_inputs_return_none(self, entries_fn, reason):
        """Degenerate history inputs return None."""
        from services.geo_utils import estimate_speed_from_history

        result = estimate_speed_from_history(entries_fn(self))
        assert result is None, f"Expected None for {reason}, got {result}"


# ---------------------------------------------------------------------------
# Test 4: geo-pattern channel config (GeoConfig — flat path)
# ---------------------------------------------------------------------------

class TestGeoPatternConfig:
    """GeoConfig produces the expected channel config and prompt content."""

    def test_config_sets_expected_attributes(self, db):
        """GeoConfig sets channel, role, skip_transcript, max_iterations,
        always_available, and discoverable."""
        from configs.channels import GeoConfig

        config = GeoConfig(window_start=0, window_end=100)

        assert config.channel == "geo_pattern"
        assert config.role == "geo_pattern"
        assert config.skip_transcript is True
        assert config.max_iterations == 30
        assert "save_pattern" in config.always_available
        assert "save_graph" in config.always_available
        assert config.discoverable == []

    def test_system_prompt_contains_geo_keywords(self, db):
        """get_system_prompt() references location-tagging, save_pattern, save_graph,
        and geo-spatial — the four pillars of the geo-pattern task description."""
        from configs.channels import GeoConfig

        config = GeoConfig(window_start=0, window_end=100)
        prompt = config.get_system_prompt(None)

        assert "location-tagged" in prompt, "System prompt must mention 'location-tagged'"
        assert "save_pattern" in prompt, "System prompt must reference the save_pattern tool"
        assert "save_graph" in prompt, "System prompt must reference the save_graph tool"
        assert "geo" in prompt.lower(), "System prompt must reference geo-spatial context"

    def test_existing_patterns_block_empty_db_returns_none_yet(self, db):
        """_pattern_existing_patterns_block() with no behavioral_pattern rows returns '(none yet)'."""
        from configs.channels import _pattern_existing_patterns_block

        block = _pattern_existing_patterns_block()

        assert block == "(none yet)", (
            f"Expected '(none yet)' on empty DB, got: {block!r}"
        )


# ---------------------------------------------------------------------------
# Test 5: PlaceAbility — schema and metadata
