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
    def test_default_speed_30kmh(self):
        """At default 30 km/h, 30 km takes 60 minutes."""
        from services.geo_utils import estimate_travel_minutes

        minutes = estimate_travel_minutes(30.0)

        assert minutes == pytest.approx(60.0)

    @_requires_geopy
    def test_custom_speed_60kmh(self):
        """At 60 km/h, 30 km takes 30 minutes."""
        from services.geo_utils import estimate_travel_minutes

        minutes = estimate_travel_minutes(30.0, speed_kmh=60.0)

        assert minutes == pytest.approx(30.0)

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
    def test_fewer_than_two_entries_returns_none(self):
        """A single location snapshot cannot yield a speed — returns None."""
        from services.geo_utils import estimate_speed_from_history

        t0 = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

        result = estimate_speed_from_history([self._entry(35.8989, 14.5146, t0)])

        assert result is None

    @_requires_geopy
    def test_empty_list_returns_none(self):
        """Empty history returns None."""
        from services.geo_utils import estimate_speed_from_history

        assert estimate_speed_from_history([]) is None

    @_requires_geopy
    def test_gps_noise_above_max_speed_filtered_returns_none(self):
        """Pairs whose implied speed exceeds 200 km/h (GPS jump) are filtered out.

        If no valid pairs survive, None is returned.
        """
        from services.geo_utils import estimate_speed_from_history

        # ~111 km in 1 second = ~400,000 km/h — far above the 200 km/h cap.
        t0 = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        t1 = t0 + timedelta(seconds=1)

        result = estimate_speed_from_history([
            self._entry(0.0, 0.0, t0),
            self._entry(1.0, 0.0, t1),
        ])

        assert result is None, (
            "GPS-noise pair (absurdly high speed) should be filtered; "
            f"expected None, got {result}"
        )

    @_requires_geopy
    def test_identical_timestamps_returns_none(self):
        """Two entries with the same timestamp → elapsed=0 → pair skipped → None."""
        from services.geo_utils import estimate_speed_from_history

        t0 = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

        result = estimate_speed_from_history([
            self._entry(35.8989, 14.5146, t0),
            self._entry(35.9121, 14.5013, t0),
        ])

        assert result is None, (
            "Identical timestamps yield zero elapsed time; "
            f"expected None, got {result}"
        )


# ---------------------------------------------------------------------------
# Test 4: GeoPatternProcessor — constructor and static methods
# ---------------------------------------------------------------------------

class TestGeoPatternProcessor:
    """GeoPatternProcessor sets expected class attributes and prompt content."""

    def test_constructor_sets_expected_attributes(self, db):
        """GeoPatternProcessor constructor sets CHANNEL, ROLE, SKIP_TRANSCRIPT_WRITE,
        MAX_ITERATIONS, ALWAYS_AVAILABLE, and DISCOVERABLE."""
        from services.geo_pattern_processor import GeoPatternProcessor

        proc = GeoPatternProcessor(window_start=0, window_end=100)

        assert proc.CHANNEL == "geo_pattern"
        assert proc.ROLE == "background"
        assert proc.SKIP_TRANSCRIPT_WRITE is True
        assert proc.MAX_ITERATIONS == 30
        assert "save_pattern" in proc.ALWAYS_AVAILABLE
        assert "save_graph" in proc.ALWAYS_AVAILABLE
        assert proc.DISCOVERABLE == []

    def test_get_user_definition_returns_empty_string(self, db):
        """get_user_definition() returns empty string — GPP has no user persona."""
        from services.geo_pattern_processor import GeoPatternProcessor

        proc = GeoPatternProcessor(window_start=0, window_end=100)

        assert proc.get_user_definition() == ""

    def test_get_system_prompt_contains_geo_keywords(self, db):
        """get_system_prompt() references location-tagging, save_pattern, save_graph,
        and geo-spatial — the four pillars of the GPP's task description."""
        from services.geo_pattern_processor import GeoPatternProcessor

        proc = GeoPatternProcessor(window_start=0, window_end=100)
        prompt = proc.get_system_prompt()

        assert "location-tagged" in prompt, "System prompt must mention 'location-tagged'"
        assert "save_pattern" in prompt, "System prompt must reference the save_pattern tool"
        assert "save_graph" in prompt, "System prompt must reference the save_graph tool"
        assert "geo" in prompt.lower(), "System prompt must reference geo-spatial context"

    def test_existing_patterns_block_empty_db_returns_none_yet(self, db):
        """_existing_patterns_block() with no behavioral_pattern rows returns '(none yet)'."""
        from services.geo_pattern_processor import GeoPatternProcessor

        proc = GeoPatternProcessor(window_start=0, window_end=100)
        block = proc._existing_patterns_block()

        assert block == "(none yet)", (
            f"Expected '(none yet)' on empty DB, got: {block!r}"
        )


# ---------------------------------------------------------------------------
# Test 5: PlaceAbility — schema and metadata
# ---------------------------------------------------------------------------

class TestPlaceAbility:
    """PlaceAbility declares correct NAME, SUMMARY, EXAMPLES, INPUT_SCHEMA, TIMEOUT."""

    def test_name_and_timeout(self):
        """NAME is 'place'; TIMEOUT is set (non-zero)."""
        from abilities.place import PlaceAbility

        ability = PlaceAbility()

        assert ability.NAME == "place"
        assert ability.TIMEOUT > 0

    def test_summary_describes_place_management(self):
        """SUMMARY mentions saving and listing named places."""
        from abilities.place import PlaceAbility

        ability = PlaceAbility()

        assert isinstance(ability.SUMMARY, str)
        assert len(ability.SUMMARY) > 0
        # SUMMARY must help routing — it should mention save/list
        lower = ability.SUMMARY.lower()
        assert "save" in lower or "named place" in lower, (
            "SUMMARY must reference saving places for correct find_tools routing"
        )

    def test_input_schema_has_expected_actions(self):
        """INPUT_SCHEMA enumerates save, list, get, delete actions."""
        from abilities.place import PlaceAbility

        ability = PlaceAbility()
        schema = ability.INPUT_SCHEMA

        assert schema.get("type") == "object"
        action_enum = schema["properties"]["action"]["enum"]
        assert set(action_enum) == {"save", "list", "get", "delete"}, (
            f"Expected actions {{save, list, get, delete}}, got {set(action_enum)}"
        )
        assert "action" in schema.get("required", []), (
            "'action' must be a required field"
        )

    def test_examples_are_non_empty(self):
        """EXAMPLES list is non-empty (needed for find_tools centroid accuracy)."""
        from abilities.place import PlaceAbility

        ability = PlaceAbility()

        assert isinstance(ability.EXAMPLES, list)
        assert len(ability.EXAMPLES) > 0, (
            "EXAMPLES must be populated so find_tools can route 'save location as home' queries"
        )
