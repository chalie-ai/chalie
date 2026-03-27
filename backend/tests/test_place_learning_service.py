"""Unit tests for PlaceLearningService."""
import pytest
from unittest.mock import patch

pytestmark = pytest.mark.unit


@pytest.fixture
def place_service(db):
    """PlaceLearningService backed by the real test database."""
    from services.database_service import get_shared_db_service
    from services.place_learning_service import PlaceLearningService
    return PlaceLearningService(get_shared_db_service())


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
    def test_lookup_below_threshold_returns_none(self, place_service, db):
        """<20 observations -> returns None (falls back to heuristics)."""
        ctx = _make_ctx()
        assert place_service.lookup(ctx) is None

    def test_lookup_no_device_returns_none(self, place_service):
        """Missing device info -> can't build fingerprint -> None."""
        ctx = {"local_time": "2026-02-25T10:00:00.000Z"}
        assert place_service.lookup(ctx) is None

    def test_lookup_no_time_returns_none(self, place_service):
        """Missing time -> can't build fingerprint -> None."""
        ctx = {"device": {"class": "desktop"}}
        assert place_service.lookup(ctx) is None


# ── Learned Patterns ─────────────────────────────────────────────

class TestLearnedPatterns:
    def test_lookup_above_threshold_returns_label(self, place_service, db):
        """>=20 observations -> returns majority label."""
        # Build the fingerprint to know what hash to seed
        ctx = _make_ctx()
        fp_hash = place_service._build_fingerprint(ctx)
        # Seed a fingerprint row with count >= 20
        db.execute(
            "INSERT INTO place_fingerprints "
            "(fingerprint_hash, device_class, hour_bucket, place_label, count) "
            "VALUES (?, 'desktop', 3, 'work', 25)",
            (fp_hash,),
        )
        db.commit()

        result = place_service.lookup(ctx)
        assert result == "work"

    def test_record_inserts_fingerprint(self, place_service, db):
        """Record should insert a fingerprint row into the database."""
        ctx = _make_ctx()
        place_service.record(ctx, "work")

        fp_hash = place_service._build_fingerprint(ctx)
        row = db.execute(
            "SELECT place_label, count FROM place_fingerprints WHERE fingerprint_hash = ?",
            (fp_hash,),
        ).fetchone()
        assert row is not None
        assert row["place_label"] == "work"
        assert row["count"] == 1

    def test_record_upserts_on_conflict(self, place_service, db):
        """Second record for the same fingerprint should increment count."""
        ctx = _make_ctx()
        place_service.record(ctx, "work")
        place_service.record(ctx, "work")

        fp_hash = place_service._build_fingerprint(ctx)
        row = db.execute(
            "SELECT count FROM place_fingerprints WHERE fingerprint_hash = ?",
            (fp_hash,),
        ).fetchone()
        assert row["count"] == 2


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
        """No location -> geohash returns None."""
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
        """Same context -> same fingerprint hash."""
        ctx = _make_ctx()
        fp1 = place_service._build_fingerprint(ctx)
        fp2 = place_service._build_fingerprint(ctx)
        assert fp1 == fp2
        assert fp1 is not None

    def test_different_device_different_fingerprint(self, place_service):
        """Different device -> different fingerprint."""
        ctx1 = _make_ctx(device_class="desktop")
        ctx2 = _make_ctx(device_class="phone")
        assert place_service._build_fingerprint(ctx1) != place_service._build_fingerprint(ctx2)

    def test_different_hour_bucket_different_fingerprint(self, place_service):
        """Different 3-hour bucket -> different fingerprint."""
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
    def test_record_no_device_noop(self, place_service, db):
        """Missing device -> record does nothing."""
        ctx = {"local_time": "2026-02-25T10:00:00.000Z"}
        place_service.record(ctx, "work")
        row = db.execute("SELECT count(*) as cnt FROM place_fingerprints").fetchone()
        assert row["cnt"] == 0

    def test_record_db_failure_no_crash(self, db):
        """DB error -> logged, no crash."""
        from services.place_learning_service import PlaceLearningService
        from services.database_service import get_shared_db_service
        svc = PlaceLearningService(get_shared_db_service())
        # Break the execute method to simulate DB failure
        with patch.object(svc.db, 'execute', side_effect=Exception("DB down")):
            ctx = _make_ctx()
            svc.record(ctx, "work")  # Should not raise

    def test_lookup_db_failure_returns_none(self, db):
        """DB error -> returns None."""
        from services.place_learning_service import PlaceLearningService
        from services.database_service import get_shared_db_service
        svc = PlaceLearningService(get_shared_db_service())
        with patch.object(svc.db, 'fetch_all', side_effect=Exception("DB down")):
            ctx = _make_ctx()
            assert svc.lookup(ctx) is None
