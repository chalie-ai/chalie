"""Feature tests for the geo-tagging components.

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

import sqlite3
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import cast

import pytest

from configs.enums.channels import Channel

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
    @_requires_geopy
    def test_valletta_to_sliema_approx_2km(self) -> None:
        from services.geo_utils import distance_km

        dist = distance_km(35.8989, 14.5146, 35.9121, 14.5013)

        assert 1.5 <= dist <= 3.0, (
            f"Expected Valletta→Sliema distance between 1.5 and 3.0 km, got {dist:.4f} km"
        )

    @_requires_geopy
    def test_same_point_is_zero(self) -> None:
        from services.geo_utils import distance_km

        dist = distance_km(35.8989, 14.5146, 35.8989, 14.5146)

        assert dist == 0.0


# ---------------------------------------------------------------------------
# Test 2: geo_utils.estimate_travel_minutes
# ---------------------------------------------------------------------------

class TestEstimateTravelMinutes:
    @_requires_geopy
    def test_zero_speed_returns_zero(self) -> None:
        """Zero speed guard returns 0.0 rather than raising ZeroDivisionError."""
        from services.geo_utils import estimate_travel_minutes

        minutes = estimate_travel_minutes(10.0, speed_kmh=0.0)

        assert minutes == 0.0


# ---------------------------------------------------------------------------
# Test 3: geo_utils.estimate_speed_from_history
# ---------------------------------------------------------------------------

class TestEstimateSpeedFromHistory:

    def _entry(self, lat: float, lon: float, ts: datetime) -> dict[str, object]:
        return {"location": {"lat": lat, "lon": lon}, "saved_at": ts.isoformat()}

    @_requires_geopy
    def test_valid_entries_return_plausible_speed(self) -> None:
        from services.geo_utils import estimate_speed_from_history

        t0 = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        t1 = t0 + timedelta(minutes=4)

        speed = estimate_speed_from_history(cast(list[object], [
            self._entry(35.8989, 14.5146, t0),
            self._entry(35.9121, 14.5013, t1),
        ]))

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
    def test_degenerate_inputs_return_none(
        self,
        entries_fn: Callable[["TestEstimateSpeedFromHistory"], list[dict[str, object]]],
        reason: str,
    ) -> None:
        from services.geo_utils import estimate_speed_from_history

        result = estimate_speed_from_history(cast(list[object], entries_fn(self)))
        assert result is None, f"Expected None for {reason}, got {result}"


# ---------------------------------------------------------------------------
# Test 4: geo-pattern window honours the per-source channel allowlist
# ---------------------------------------------------------------------------

def _seed_located_row(db: sqlite3.Connection, *, channel: str, role: str = "user", content: str) -> None:
    db.execute(
        "INSERT INTO transcript (channel, role, content, created_at, "
        "location_lat, location_lon, location_name) "
        "VALUES (?, ?, ?, datetime('now'), 35.9, 14.5, 'Valletta')",
        (channel, role, content),
    )
    db.commit()


class TestGeoPatternWindowChannelFilter:
    """The geo-pattern window feeds the model ONLY rows whose source profile
    marks them as user geo-activity. A located row on a muted/non-user channel
    must never reach the model — the prior behaviour fed every located row
    regardless of channel, so a delegate's coordinates could masquerade as the
    user being somewhere.

    Driven through the real ``_geo_pattern_load_transcript_block`` (the exact
    builder GeoConfig.get_user_prompt calls) over mixed-channel fixtures.
    """

    def test_only_user_geoactivity_rows_enter_the_window(self, db: sqlite3.Connection) -> None:
        from configs.channels.geo_pattern import _geo_pattern_load_transcript_block

        # One row per representative channel, all located, all in the window.
        _seed_located_row(db, channel=Channel.USER.value, content="USER at the gym")
        _seed_located_row(db, channel=Channel.DMN.value, content="DMN reflection located")
        _seed_located_row(db, channel="delegate:research", content="DELEGATE located")
        _seed_located_row(db, channel=Channel.for_external_agent("bob"), content="EXT located")

        last_id = db.execute("SELECT MAX(id) FROM transcript").fetchone()[0]
        block = _geo_pattern_load_transcript_block(0, last_id)

        # Only the user geo-activity row is admitted.
        assert "USER at the gym" in block
        # dmn is HEAVY for episodes but NOT user geo-activity (geo_is_user=False).
        assert "DMN reflection located" not in block
        # Muted / non-user channels are excluded.
        assert "DELEGATE located" not in block
        assert "EXT located" not in block

    def test_existing_patterns_block_empty_db_returns_no_patterns(self, db: sqlite3.Connection) -> None:
        """The behavioural-pattern lane feeding the existing-patterns block is
        empty on a fresh DB. The '(none yet)' prose now lives in
        PromptService._pattern_existing_patterns_block; the data-level surface it
        renders from is BehavioralPatternService.top_patterns(). The read path
        (top_patterns -> patterns -> BehavioralPattern.recent()) runs over the
        class-bound connection and never touches ``mp``."""
        from services.behavioral_pattern_service import BehavioralPatternService

        assert BehavioralPatternService(None).top_patterns() == {}
