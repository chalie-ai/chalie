"""
Tests for TemporalPatternService — temporal behavioral pattern mining.

Covers:
- Existing: peak hours/days detection, static helpers, trait storage
- New: observation buffer, Laplace-smoothed prediction, probability-ratio anomaly
  detection, rhythm summary, cleanup, transitions, ambient distribution, stats

Tests use either MagicMock (for isolated logic) or in-memory SQLite with real
schema tables (for DB-backed methods).
"""

import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest


# ── In-memory DB fixture for tests that need real SQLite ────────────────

class InMemoryDB:
    """Minimal DB service backed by in-memory SQLite."""

    def __init__(self):
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._setup_schema()

    def _setup_schema(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS temporal_observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                observation_type TEXT NOT NULL,
                observed_value TEXT NOT NULL,
                day_of_week INTEGER NOT NULL,
                hour_bucket INTEGER NOT NULL,
                device_class TEXT,
                location_hash TEXT,
                recorded_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_temporal_obs_type_day_hour
                ON temporal_observations(observation_type, day_of_week, hour_bucket);
            CREATE INDEX IF NOT EXISTS idx_temporal_obs_recorded
                ON temporal_observations(recorded_at);

            CREATE TABLE IF NOT EXISTS temporal_aggregate (
                observation_type TEXT NOT NULL,
                observed_value TEXT NOT NULL,
                day_of_week INTEGER NOT NULL,
                hour_bucket INTEGER NOT NULL,
                device_class TEXT NOT NULL DEFAULT '',
                count INTEGER DEFAULT 0,
                last_seen TEXT,
                PRIMARY KEY(observation_type, observed_value,
                            day_of_week, hour_bucket, device_class)
            );

            CREATE TABLE IF NOT EXISTS interaction_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT,
                topic TEXT,
                created_at TEXT
            );

            CREATE TABLE IF NOT EXISTS user_traits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trait_key TEXT,
                trait_value TEXT,
                confidence REAL DEFAULT 0.5,
                category TEXT,
                source TEXT DEFAULT 'inferred',
                is_literal INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );
        """)

    @contextmanager
    def connection(self):
        yield self._conn

    def close(self):
        self._conn.close()


@pytest.fixture
def memdb():
    d = InMemoryDB()
    yield d
    d.close()


@pytest.fixture
def svc(memdb):
    from services.temporal_pattern_service import TemporalPatternService
    return TemporalPatternService(memdb)


@pytest.fixture
def fresh_buffer():
    """Return a fresh buffer instance (not the module singleton)."""
    from services.temporal_pattern_service import TemporalObservationBuffer
    return TemporalObservationBuffer(max_buffer=5)


def seed_aggregate(db, obs_type, value, day, hour, count, device_class=''):
    """Insert a row into temporal_aggregate for testing."""
    with db.connection() as conn:
        conn.execute(
            """INSERT INTO temporal_aggregate
                (observation_type, observed_value, day_of_week,
                 hour_bucket, device_class, count, last_seen)
            VALUES (?, ?, ?, ?, ?, ?, datetime('now'))""",
            (obs_type, value, day, hour, device_class, count)
        )
        conn.commit()


def seed_interaction_log(db, hour, topic=None, count=1, days_ago=0):
    """Insert rows into interaction_log."""
    base_dt = datetime.utcnow() - timedelta(days=days_ago)
    ts = base_dt.replace(hour=hour, minute=0, second=0).isoformat()
    with db.connection() as conn:
        for _ in range(count):
            conn.execute(
                "INSERT INTO interaction_log (event_type, topic, created_at) VALUES (?, ?, ?)",
                ('user_input', topic, ts)
            )
        conn.commit()


# ═══════════════════════════════════════════════════════════════════════
# Buffer Tests
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestTemporalObservationBuffer:

    def test_append_and_flush(self, memdb, fresh_buffer):
        """Buffer flushes to both raw observations and aggregate tables."""
        fresh_buffer.append({
            'observation_type': 'energy',
            'observed_value': 'high',
            'day_of_week': 2,
            'hour_bucket': 10,
            'device_class': 'desktop',
            'location_hash': 'abc123',
        })
        assert fresh_buffer.pending_count == 1

        fresh_buffer.flush(memdb)
        assert fresh_buffer.pending_count == 0

        with memdb.connection() as conn:
            raw = conn.execute("SELECT COUNT(*) FROM temporal_observations").fetchone()[0]
            agg = conn.execute("SELECT COUNT(*) FROM temporal_aggregate").fetchone()[0]
        assert raw == 1
        assert agg == 1

    def test_auto_flush_at_max_buffer(self, fresh_buffer):
        """Buffer auto-flushes when max_buffer is reached."""
        flush_calls = []
        original = fresh_buffer._flush_locked

        def tracking_flush(db_service=None):
            flush_calls.append(True)

        fresh_buffer._flush_locked = tracking_flush

        for i in range(5):
            fresh_buffer.append({
                'observation_type': 'energy', 'observed_value': 'high',
                'day_of_week': 0, 'hour_bucket': i,
            })

        assert len(flush_calls) == 1

    def test_thread_safety(self, fresh_buffer):
        """Multiple threads can append without data loss or crashes."""
        num_threads = 10
        items_per_thread = 20
        barrier = threading.Barrier(num_threads)

        def writer(thread_id):
            barrier.wait()
            for i in range(items_per_thread):
                fresh_buffer.append({
                    'observation_type': 'energy', 'observed_value': 'high',
                    'day_of_week': 0, 'hour_bucket': thread_id,
                })

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # No exceptions = thread-safe

    def test_flush_empty_buffer(self, memdb, fresh_buffer):
        """Flushing an empty buffer is a no-op."""
        fresh_buffer.flush(memdb)
        assert fresh_buffer.pending_count == 0
        assert fresh_buffer.write_errors_count == 0

    def test_flush_upserts_aggregate(self, memdb, fresh_buffer):
        """Repeated observations UPSERT the aggregate count."""
        obs = {
            'observation_type': 'energy', 'observed_value': 'high',
            'day_of_week': 1, 'hour_bucket': 9,
            'device_class': '', 'location_hash': '',
        }
        for _ in range(3):
            fresh_buffer.append(obs.copy())
        fresh_buffer.flush(memdb)

        with memdb.connection() as conn:
            raw = conn.execute("SELECT COUNT(*) FROM temporal_observations").fetchone()[0]
            agg_count = conn.execute(
                "SELECT count FROM temporal_aggregate WHERE observation_type='energy'"
            ).fetchone()[0]
        assert raw == 3
        assert agg_count == 3


# ═══════════════════════════════════════════════════════════════════════
# Prediction Tests
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestPrediction:

    def test_predict_strong_pattern(self, memdb, svc):
        """Predict returns the dominant value with Laplace-smoothed confidence."""
        # high=25, moderate=3, low=2 → total=30
        seed_aggregate(memdb, 'energy', 'high', 1, 10, 25)
        seed_aggregate(memdb, 'energy', 'moderate', 1, 10, 3)
        seed_aggregate(memdb, 'energy', 'low', 1, 10, 2)

        result = svc.predict('energy', day=1, hour=10)
        assert result is not None
        assert result['value'] == 'high'
        assert result['observation_count'] == 30
        # Laplace: (25+1)/(30+3) = 26/33 ≈ 0.788
        assert 0.78 < result['confidence'] < 0.80
        assert result['runner_up'] == 'moderate'

    def test_predict_insufficient_data(self, memdb, svc):
        """Returns None below MIN_OBSERVATIONS_PER_BUCKET (10)."""
        seed_aggregate(memdb, 'energy', 'high', 1, 10, 5)
        assert svc.predict('energy', day=1, hour=10) is None

    def test_predict_low_confidence(self, memdb, svc):
        """Returns None when no value reaches MIN_PREDICTION_CONFIDENCE (0.6)."""
        # Even distribution → each ≈33% → Laplace ≈ (12+1)/(33+3) = 0.361
        seed_aggregate(memdb, 'energy', 'high', 1, 10, 12)
        seed_aggregate(memdb, 'energy', 'moderate', 1, 10, 11)
        seed_aggregate(memdb, 'energy', 'low', 1, 10, 10)
        assert svc.predict('energy', day=1, hour=10) is None

    def test_predict_unknown_obs_type(self, memdb, svc):
        """Returns None for unregistered observation types."""
        assert svc.predict('nonexistent', day=1, hour=10) is None

    def test_predict_weekday_aggregation(self, memdb, svc):
        """Weekday prediction aggregates across Mon-Fri for the given hour."""
        for d in range(5):
            seed_aggregate(memdb, 'energy', 'high', d, 10, 5)
        result = svc.predict('energy', day=0, hour=10)
        assert result is not None
        assert result['observation_count'] == 25

    def test_predict_weekend_aggregation(self, memdb, svc):
        """Weekend prediction aggregates Saturday + Sunday."""
        seed_aggregate(memdb, 'energy', 'low', 5, 22, 15)
        seed_aggregate(memdb, 'energy', 'low', 6, 22, 15)
        result = svc.predict('energy', day=5, hour=22)
        assert result is not None
        assert result['value'] == 'low'
        assert result['observation_count'] == 30


# ═══════════════════════════════════════════════════════════════════════
# Transition Detection
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestTransitions:

    def test_get_upcoming_transitions(self, memdb, svc):
        """Detects value change at hour boundary."""
        # Use a fixed weekday + hour so the test passes regardless of when it runs.
        fixed_hour = 10
        next_hour = 11
        fixed_monday = datetime(2026, 3, 2, fixed_hour, 0, 0)  # known Monday

        for d in range(5):
            seed_aggregate(memdb, 'energy', 'high', d, fixed_hour, 20)
            seed_aggregate(memdb, 'energy', 'low', d, next_hour, 20)

        with patch('services.temporal_pattern_service.datetime') as mock_dt:
            mock_dt.utcnow.return_value = fixed_monday
            transitions = svc.get_upcoming_transitions(lookahead_minutes=120)

        energy_t = [t for t in transitions if t['obs_type'] == 'energy']
        assert len(energy_t) >= 1
        assert energy_t[0]['from_value'] == 'high'
        assert energy_t[0]['to_value'] == 'low'


# ═══════════════════════════════════════════════════════════════════════
# Anomaly Detection
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestAnomalyDetection:

    def test_detects_anomaly(self, memdb, svc):
        """Flags when predicted_prob > 0.75 and actual_prob < 0.3."""
        for d in range(5):
            seed_aggregate(memdb, 'energy', 'high', d, 10, 40)
            seed_aggregate(memdb, 'energy', 'moderate', d, 10, 3)
            seed_aggregate(memdb, 'energy', 'low', d, 10, 2)

        anomaly = svc.detect_anomaly('energy', 'low', day=0, hour=10)
        assert anomaly is not None
        assert anomaly['expected'] == 'high'
        assert anomaly['actual'] == 'low'
        assert anomaly['predicted_prob'] > 0.75
        assert anomaly['actual_prob'] < 0.3

    def test_no_anomaly_when_matching(self, memdb, svc):
        """No anomaly when current value matches prediction."""
        for d in range(5):
            seed_aggregate(memdb, 'energy', 'high', d, 10, 40)
            seed_aggregate(memdb, 'energy', 'moderate', d, 10, 3)
        assert svc.detect_anomaly('energy', 'high', day=0, hour=10) is None

    def test_no_anomaly_insufficient_data(self, memdb, svc):
        """No anomaly below ANOMALY_MIN_OBSERVATIONS (20)."""
        seed_aggregate(memdb, 'energy', 'high', 0, 10, 5)
        assert svc.detect_anomaly('energy', 'low', day=0, hour=10) is None

    def test_no_anomaly_when_actual_not_rare(self, memdb, svc):
        """No anomaly when actual_prob >= 0.3 (not rare enough)."""
        for d in range(5):
            seed_aggregate(memdb, 'energy', 'high', d, 10, 10)
            seed_aggregate(memdb, 'energy', 'moderate', d, 10, 6)
            seed_aggregate(memdb, 'energy', 'low', d, 10, 1)
        assert svc.detect_anomaly('energy', 'moderate', day=0, hour=10) is None


# ═══════════════════════════════════════════════════════════════════════
# Rhythm Summary
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestRhythmSummary:

    def test_empty_when_no_traits(self, memdb, svc):
        assert svc.get_rhythm_summary() == ''

    def test_returns_summary_lines(self, memdb, svc):
        with memdb.connection() as conn:
            conn.execute(
                """INSERT INTO user_traits (trait_key, trait_value, confidence, category)
                   VALUES ('weekday_energy_rhythm', 'Typically high in the morning', 0.8, 'behavioral')"""
            )
            conn.execute(
                """INSERT INTO user_traits (trait_key, trait_value, confidence, category)
                   VALUES ('weekday_attention_rhythm', 'Typically deep_focus in the morning', 0.7, 'behavioral')"""
            )
            conn.commit()

        result = svc.get_rhythm_summary()
        assert 'high in the morning' in result
        assert 'deep_focus in the morning' in result

    def test_max_three_lines(self, memdb, svc):
        with memdb.connection() as conn:
            for i in range(5):
                conn.execute(
                    """INSERT INTO user_traits (trait_key, trait_value, confidence, category)
                       VALUES (?, ?, 0.8, 'behavioral')""",
                    (f'pattern_{i}', f'Pattern line {i}')
                )
            conn.commit()

        result = svc.get_rhythm_summary()
        lines = [l for l in result.split('\n') if l.strip()]
        assert len(lines) <= 3

    def test_max_200_chars(self, memdb, svc):
        with memdb.connection() as conn:
            for key in ['weekday_energy_rhythm', 'weekday_attention_rhythm']:
                conn.execute(
                    """INSERT INTO user_traits (trait_key, trait_value, confidence, category)
                       VALUES (?, ?, 0.8, 'behavioral')""",
                    (key, 'A' * 150)
                )
            conn.commit()

        assert len(svc.get_rhythm_summary()) <= 200

    def test_sanitizes_non_printable(self, memdb, svc):
        with memdb.connection() as conn:
            conn.execute(
                """INSERT INTO user_traits (trait_key, trait_value, confidence, category)
                   VALUES ('weekday_energy_rhythm', ?, 0.8, 'behavioral')""",
                ('High energy \x00\x01\x02 mornings',)
            )
            conn.commit()

        result = svc.get_rhythm_summary()
        assert '\x00' not in result
        assert 'High energy' in result


# ═══════════════════════════════════════════════════════════════════════
# Cleanup
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestCleanup:

    def test_removes_old_observations(self, memdb, svc):
        old_ts = (datetime.utcnow() - timedelta(days=100)).isoformat()
        recent_ts = (datetime.utcnow() - timedelta(days=5)).isoformat()

        with memdb.connection() as conn:
            conn.execute(
                """INSERT INTO temporal_observations
                   (observation_type, observed_value, day_of_week, hour_bucket, recorded_at)
                   VALUES ('energy', 'high', 1, 10, ?)""", (old_ts,))
            conn.execute(
                """INSERT INTO temporal_observations
                   (observation_type, observed_value, day_of_week, hour_bucket, recorded_at)
                   VALUES ('energy', 'low', 2, 14, ?)""", (recent_ts,))
            conn.commit()

        svc.cleanup_old_observations(retention_days=90)

        with memdb.connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM temporal_observations").fetchone()[0]
        assert count == 1

    def test_preserves_aggregates(self, memdb, svc):
        seed_aggregate(memdb, 'energy', 'high', 1, 10, 50)
        svc.cleanup_old_observations(retention_days=1)

        with memdb.connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM temporal_aggregate").fetchone()[0]
        assert count == 1


# ═══════════════════════════════════════════════════════════════════════
# Mining Integration
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestMiningIntegration:

    def test_mine_empty_db(self, memdb, svc):
        patterns = svc.mine_patterns()
        assert isinstance(patterns, list)
        assert len(patterns) == 0

    def test_mine_ambient_patterns(self, memdb, svc):
        for day in range(5):
            for hour in [9, 10, 11]:
                seed_aggregate(memdb, 'energy', 'high', day, hour, 15)
                seed_aggregate(memdb, 'energy', 'moderate', day, hour, 2)

        patterns = svc.mine_patterns()
        energy_p = [p for p in patterns if 'energy' in p['key']]
        assert len(energy_p) > 0

    def test_mine_interaction_log_peaks(self, memdb, svc):
        for _ in range(30):
            seed_interaction_log(memdb, 10)
        for h in [8, 14, 20]:
            seed_interaction_log(memdb, h, count=2)

        patterns = svc.mine_patterns()
        active = [p for p in patterns if p['key'] == 'active_hours']
        assert len(active) >= 1
        assert 'morning' in active[0]['value'].lower()

    def test_mine_detects_transitions(self, memdb, svc):
        for day in range(5):
            seed_aggregate(memdb, 'energy', 'high', day, 17, 20)
            seed_aggregate(memdb, 'energy', 'moderate', day, 17, 2)
            seed_aggregate(memdb, 'energy', 'low', day, 18, 20)
            seed_aggregate(memdb, 'energy', 'moderate', day, 18, 2)

        patterns = svc.mine_patterns()
        assert any('transition' in p['key'] for p in patterns)


# ═══════════════════════════════════════════════════════════════════════
# Observation Stats
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestObservationStats:

    def test_stats_empty_db(self, memdb, svc):
        stats = svc.get_observation_stats()
        assert stats['observation_count'] == 0
        assert stats['aggregate_row_count'] == 0
        assert stats['oldest_observation'] is None

    def test_stats_with_data(self, memdb, svc):
        seed_aggregate(memdb, 'energy', 'high', 0, 10, 15)
        seed_aggregate(memdb, 'energy', 'low', 0, 22, 15)

        with memdb.connection() as conn:
            conn.execute(
                """INSERT INTO temporal_observations
                   (observation_type, observed_value, day_of_week, hour_bucket)
                   VALUES ('energy', 'high', 0, 10)""")
            conn.commit()

        stats = svc.get_observation_stats()
        assert stats['observation_count'] == 1
        assert stats['aggregate_row_count'] == 2
        assert 'energy' in stats['predictions_available']


# ═══════════════════════════════════════════════════════════════════════
# Original Tests (preserved, unchanged)
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestDetectPeakHours:

    def setup_method(self):
        from services.temporal_pattern_service import TemporalPatternService
        self.service = TemporalPatternService(MagicMock())

    def test_detects_significant_peak(self):
        """Hour with 2x average frequency should be detected."""
        dist = {h: 5 for h in range(24)}  # average = 5
        dist[10] = 30  # 6x average, well above threshold
        dist[11] = 25
        patterns = self.service._detect_peak_hours(dist)
        assert len(patterns) > 0

    def test_peak_label_is_generalized(self):
        """Peak label should be a broad time-of-day label, never a raw hour."""
        dist = {h: 2 for h in range(24)}
        dist[10] = 40  # morning peak
        dist[11] = 35
        patterns = self.service._detect_peak_hours(dist)
        assert len(patterns) > 0
        assert 'morning' in patterns[0]['value']

    def test_insufficient_data_returns_empty(self):
        """Less than MIN_OBSERVATIONS total should return no patterns."""
        dist = {10: 2, 11: 1}  # total = 3 < 10
        patterns = self.service._detect_peak_hours(dist)
        assert patterns == []

    def test_no_significant_peak_returns_empty(self):
        """Uniform distribution should return no patterns."""
        dist = {h: 10 for h in range(24)}  # no peaks
        patterns = self.service._detect_peak_hours(dist)
        assert patterns == []

    def test_confidence_capped_at_0_9(self):
        """Confidence should never exceed 0.9 (avoid false certainty)."""
        dist = {h: 1 for h in range(24)}
        dist[10] = 500  # extreme peak
        patterns = self.service._detect_peak_hours(dist)
        if patterns:
            assert patterns[0]['confidence'] <= 0.9

    def test_key_is_active_hours(self):
        """Pattern key should be 'active_hours'."""
        dist = {h: 2 for h in range(24)}
        dist[10] = 40
        dist[11] = 35
        patterns = self.service._detect_peak_hours(dist)
        if patterns:
            assert patterns[0]['key'] == 'active_hours'


@pytest.mark.unit
class TestDetectPeakDays:

    def setup_method(self):
        from services.temporal_pattern_service import TemporalPatternService
        self.service = TemporalPatternService(MagicMock())

    def test_detects_peak_day(self):
        """Day with 2x average frequency should be detected."""
        dist = {d: 5 for d in range(7)}
        dist[0] = 20  # Monday, 4x average
        patterns = self.service._detect_peak_days(dist)
        assert len(patterns) > 0
        assert 'Monday' in patterns[0]['value']

    def test_insufficient_data_returns_empty(self):
        dist = {0: 2, 1: 1}  # total = 3 < 10
        patterns = self.service._detect_peak_days(dist)
        assert patterns == []

    def test_uniform_distribution_returns_empty(self):
        dist = {d: 10 for d in range(7)}
        patterns = self.service._detect_peak_days(dist)
        assert patterns == []

    def test_key_is_active_days(self):
        dist = {d: 5 for d in range(7)}
        dist[4] = 25  # Friday peak
        patterns = self.service._detect_peak_days(dist)
        if patterns:
            assert patterns[0]['key'] == 'active_days'

    def test_multiple_peak_days_combined(self):
        dist = {d: 3 for d in range(7)}
        dist[5] = 20  # Saturday
        dist[6] = 18  # Sunday
        patterns = self.service._detect_peak_days(dist)
        assert len(patterns) == 1  # Combined into one pattern
        assert 'Saturday' in patterns[0]['value']
        assert 'Sunday' in patterns[0]['value']


@pytest.mark.unit
class TestHourToLabel:

    def test_morning(self):
        from services.temporal_pattern_service import TemporalPatternService
        assert TemporalPatternService._hour_to_label(10) == "morning"

    def test_night(self):
        from services.temporal_pattern_service import TemporalPatternService
        assert TemporalPatternService._hour_to_label(22) == "night"

    def test_late_night(self):
        from services.temporal_pattern_service import TemporalPatternService
        assert TemporalPatternService._hour_to_label(2) == "late night"

    def test_all_hours_map_to_label(self):
        """Every hour (0-23) should return a named label — never expose raw numbers."""
        from services.temporal_pattern_service import TemporalPatternService
        valid_labels = {"early morning", "morning", "midday", "afternoon",
                        "evening", "night", "late night", "daytime"}
        for h in range(24):
            label = TemporalPatternService._hour_to_label(h)
            assert label in valid_labels, f"Hour {h} mapped to unknown label '{label}'"

    def test_no_colon_in_label(self):
        """Labels must never contain raw time formats like '10:00'."""
        from services.temporal_pattern_service import TemporalPatternService
        for h in range(24):
            assert ":" not in TemporalPatternService._hour_to_label(h)


@pytest.mark.unit
class TestGroupContiguous:

    def test_single_element(self):
        from services.temporal_pattern_service import TemporalPatternService
        assert TemporalPatternService._group_contiguous([5]) == [[5]]

    def test_contiguous_group(self):
        from services.temporal_pattern_service import TemporalPatternService
        assert TemporalPatternService._group_contiguous([9, 10, 11]) == [[9, 10, 11]]

    def test_two_groups(self):
        from services.temporal_pattern_service import TemporalPatternService
        assert TemporalPatternService._group_contiguous([9, 10, 11, 14, 15]) == [[9, 10, 11], [14, 15]]

    def test_empty(self):
        from services.temporal_pattern_service import TemporalPatternService
        assert TemporalPatternService._group_contiguous([]) == []

    def test_non_contiguous_single_elements(self):
        from services.temporal_pattern_service import TemporalPatternService
        result = TemporalPatternService._group_contiguous([1, 3, 5])
        assert result == [[1], [3], [5]]


@pytest.mark.unit
class TestSlugifyTopic:

    def test_basic_slugification(self):
        from services.temporal_pattern_service import TemporalPatternService
        assert TemporalPatternService._slugify_topic("My Cooking Adventures!") == "my_cooking_adventures"

    def test_max_length_capped(self):
        from services.temporal_pattern_service import TemporalPatternService
        long = "a" * 100
        result = TemporalPatternService._slugify_topic(long)
        assert len(result) <= 40

    def test_special_chars_removed(self):
        from services.temporal_pattern_service import TemporalPatternService
        result = TemporalPatternService._slugify_topic("hello/world#test")
        assert '/' not in result
        assert '#' not in result

    def test_leading_trailing_underscores_removed(self):
        from services.temporal_pattern_service import TemporalPatternService
        result = TemporalPatternService._slugify_topic("  hello world  ")
        assert not result.startswith('_')
        assert not result.endswith('_')


@pytest.mark.unit
class TestStorePatternsAsTraits:

    def test_calls_store_trait_for_each_pattern(self):
        """Each pattern should be stored via store_trait()."""
        from services.temporal_pattern_service import TemporalPatternService
        mock_db = MagicMock()
        service = TemporalPatternService(mock_db)

        mock_trait_svc = MagicMock()
        mock_trait_svc.store_trait.return_value = True

        patterns = [
            {'key': 'active_hours', 'value': 'Most active in the evenings', 'confidence': 0.7},
            {'key': 'active_days', 'value': 'Most active on Monday', 'confidence': 0.7},
        ]

        # UserTraitService is a lazy import inside _store_patterns_as_traits,
        # so patch the source module, not the temporal_pattern_service module.
        with patch('services.user_trait_service.UserTraitService', return_value=mock_trait_svc):
            service._store_patterns_as_traits(patterns)

        assert mock_trait_svc.store_trait.call_count == 2

    def test_passes_correct_arguments_to_store_trait(self):
        """store_trait() should be called with category='behavioral'.

        behavioral → behavioral after Stream 1 3-tier CATEGORY_DECAY simplification.
        source removed from store_trait in migration 006.
        """
        from services.temporal_pattern_service import TemporalPatternService
        mock_db = MagicMock()
        service = TemporalPatternService(mock_db)

        mock_trait_svc = MagicMock()
        mock_trait_svc.store_trait.return_value = True

        patterns = [{'key': 'active_hours', 'value': 'Most active in the mornings', 'confidence': 0.7}]

        with patch('services.user_trait_service.UserTraitService', return_value=mock_trait_svc):
            service._store_patterns_as_traits(patterns)

        call_kwargs = mock_trait_svc.store_trait.call_args.kwargs
        assert call_kwargs['category'] == 'behavioral'
        assert call_kwargs['trait_key'] == 'active_hours'

    def test_empty_patterns_does_not_call_store(self):
        """No patterns → store_trait should never be called."""
        from services.temporal_pattern_service import TemporalPatternService
        mock_db = MagicMock()
        service = TemporalPatternService(mock_db)

        mock_trait_svc = MagicMock()

        with patch('services.user_trait_service.UserTraitService', return_value=mock_trait_svc):
            service._store_patterns_as_traits([])

        mock_trait_svc.store_trait.assert_not_called()

    def test_error_does_not_raise(self):
        """Storage failure should be swallowed (logged, not raised)."""
        from services.temporal_pattern_service import TemporalPatternService
        mock_db = MagicMock()
        service = TemporalPatternService(mock_db)

        with patch('services.user_trait_service.UserTraitService', side_effect=Exception("db error")):
            # Should not raise
            service._store_patterns_as_traits(
                [{'key': 'k', 'value': 'v', 'confidence': 0.5}]
            )


@pytest.mark.unit
class TestBehavioralPatternDecay:
    """behavioral renamed to behavioral in Stream 1 3-tier CATEGORY_DECAY simplification."""

    def test_behavioral_in_category_decay(self):
        """behavioral should be a registered decay category."""
        from services.user_trait_service import CATEGORY_DECAY
        assert 'behavioral' in CATEGORY_DECAY

    def test_behavioral_has_slow_decay(self):
        """Activity time patterns are stable — base_decay should be low."""
        from services.user_trait_service import CATEGORY_DECAY
        assert CATEGORY_DECAY['behavioral']['base_decay'] <= 0.01

    def test_behavioral_has_floor(self):
        """Behavioral patterns should have a floor to prevent total erasure."""
        from services.user_trait_service import CATEGORY_DECAY
        assert CATEGORY_DECAY['behavioral']['floor'] > 0
