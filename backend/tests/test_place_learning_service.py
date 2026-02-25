"""Unit tests for PlaceLearningService."""
import pytest
from unittest.mock import MagicMock

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_db():
    """Create a mock DatabaseService."""
    db = MagicMock()
    db.execute = MagicMock()
    db.fetch_all = MagicMock(return_value=[])
    return db


@pytest.fixture
def place_service(mock_db):
    from services.place_learning_service import PlaceLearningService
    return PlaceLearningService(mock_db)


def _make_ctx(device_class="desktop", hour="10", lat=35.90, lon=14.50, connection="4g"):
    ctx = {
        "device": {"class": device_class},
        "local_time": f"2026-02-25T{hour}:30:00.000Z",
        "connection": connection,
    }
    if lat is not None and lon is not None:
        ctx["location"] = {"lat": lat, "lon": lon}
    return ctx


# ── Cold Start ───────────────────────────────────────────────────

class TestColdStart:
    def test_lookup_below_threshold_returns_none(self, place_service, mock_db):
        """<20 observations → returns None (falls back to heuristics)."""
        mock_db.fetch_all.return_value = []
        ctx = _make_ctx()
        assert place_service.lookup(ctx) is None

    def test_lookup_no_device_returns_none(self, place_service):
        """Missing device info → can't build fingerprint → None."""
        ctx = {"local_time": "2026-02-25T10:00:00.000Z"}
        assert place_service.lookup(ctx) is None

    def test_lookup_no_time_returns_none(self, place_service):
        """Missing time → can't build fingerprint → None."""
        ctx = {"device": {"class": "desktop"}}
        assert place_service.lookup(ctx) is None


# ── Learned Patterns ─────────────────────────────────────────────

class TestLearnedPatterns:
    def test_lookup_above_threshold_returns_label(self, place_service, mock_db):
        """>=20 observations → returns majority label."""
        mock_db.fetch_all.return_value = [{"place_label": "work", "count": 25}]
        ctx = _make_ctx()
        result = place_service.lookup(ctx)
        assert result == "work"

    def test_record_calls_execute(self, place_service, mock_db):
        """Record should call db.execute with upsert SQL."""
        ctx = _make_ctx()
        place_service.record(ctx, "work")
        mock_db.execute.assert_called_once()
        call_args = mock_db.execute.call_args
        sql = call_args[0][0]
        assert "INSERT INTO place_fingerprints" in sql
        assert "ON CONFLICT" in sql


# ── Geohash Privacy ──────────────────────────────────────────────

class TestGeohashPrivacy:
    def test_raw_coords_not_stored(self, place_service):
        """Geohash returns a hash, not raw coordinates."""
        ctx = _make_ctx(lat=35.8989, lon=14.5134)
        gh = place_service._geohash(ctx)
        assert gh is not None
        # Should be a short hex hash, not contain raw floats
        assert "35.89" not in gh
        assert "14.51" not in gh
        assert len(gh) == 8

    def test_no_location_returns_none(self, place_service):
        """No location → geohash returns None."""
        ctx = {"device": {"class": "desktop"}, "local_time": "2026-02-25T10:00:00.000Z"}
        assert place_service._geohash(ctx) is None

    def test_nearby_locations_same_hash(self, place_service):
        """Locations within ~1km quantize to the same hash."""
        ctx1 = _make_ctx(lat=35.9001, lon=14.5001)
        ctx2 = _make_ctx(lat=35.9005, lon=14.5005)
        assert place_service._geohash(ctx1) == place_service._geohash(ctx2)

    def test_distant_locations_different_hash(self, place_service):
        """Locations >1km apart should produce different hashes."""
        ctx1 = _make_ctx(lat=35.90, lon=14.50)
        ctx2 = _make_ctx(lat=35.95, lon=14.55)
        assert place_service._geohash(ctx1) != place_service._geohash(ctx2)


# ── Fingerprint Consistency ──────────────────────────────────────

class TestFingerprintConsistency:
    def test_same_context_same_fingerprint(self, place_service):
        """Same context → same fingerprint hash."""
        ctx = _make_ctx()
        fp1 = place_service._build_fingerprint(ctx)
        fp2 = place_service._build_fingerprint(ctx)
        assert fp1 == fp2
        assert fp1 is not None

    def test_different_device_different_fingerprint(self, place_service):
        """Different device → different fingerprint."""
        ctx1 = _make_ctx(device_class="desktop")
        ctx2 = _make_ctx(device_class="phone")
        assert place_service._build_fingerprint(ctx1) != place_service._build_fingerprint(ctx2)

    def test_different_hour_bucket_different_fingerprint(self, place_service):
        """Different 3-hour bucket → different fingerprint."""
        ctx1 = _make_ctx(hour="09")  # bucket 3
        ctx2 = _make_ctx(hour="15")  # bucket 5
        assert place_service._build_fingerprint(ctx1) != place_service._build_fingerprint(ctx2)

    def test_hour_to_bucket(self, place_service):
        """Hour to 3-hour bucket mapping."""
        assert place_service._hour_to_bucket({"local_time": "2026-02-25T00:00:00Z"}) == 0
        assert place_service._hour_to_bucket({"local_time": "2026-02-25T05:00:00Z"}) == 1
        assert place_service._hour_to_bucket({"local_time": "2026-02-25T09:00:00Z"}) == 3
        assert place_service._hour_to_bucket({"local_time": "2026-02-25T23:00:00Z"}) == 7


# ── Graceful Degradation ─────────────────────────────────────────

class TestGracefulDegradation:
    def test_record_no_device_noop(self, place_service, mock_db):
        """Missing device → record does nothing."""
        ctx = {"local_time": "2026-02-25T10:00:00.000Z"}
        place_service.record(ctx, "work")
        mock_db.execute.assert_not_called()

    def test_record_db_failure_no_crash(self, place_service, mock_db):
        """DB error → logged, no crash."""
        mock_db.execute.side_effect = Exception("DB down")
        ctx = _make_ctx()
        place_service.record(ctx, "work")  # Should not raise

    def test_lookup_db_failure_returns_none(self, place_service, mock_db):
        """DB error → returns None."""
        mock_db.fetch_all.side_effect = Exception("DB down")
        ctx = _make_ctx()
        assert place_service.lookup(ctx) is None
