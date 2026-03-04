"""
Temporal Pattern Service — mines behavioral patterns from interaction history
and ambient inference signals.

Two data sources:
1. interaction_log (chat timestamps) — existing: peak hours, peak days, topic-time associations
2. temporal_observations + temporal_aggregate (ambient signals) — NEW: attention, energy,
   place, tempo, mobility rhythms observed every heartbeat

Design principles:
- Generalization over specificity: all time references use broad labels ("evenings"),
  never specific hours ("10-11pm"). Even correct patterns can feel invasive if too precise.
- Deduplication via store_trait(): UserTraitService.store_trait() handles conflict
  resolution (reinforce same value, overwrite if confidence > 2x). Repeated mining
  cycles won't create duplicates.
- Slugified trait keys: topic names are slugified and capped at 40 chars.
- Write serialization: TemporalObservationBuffer accumulates observations in-memory,
  flushes to SQLite on a single writer thread (avoids busy locks in WAL mode).
- Aggregate-first reads: mining queries read temporal_aggregate (bounded ~17k rows),
  not raw temporal_observations.
- Laplace-smoothed predictions: avoids overconfident predictions on small samples.
- Probability-ratio anomaly detection: flags when predicted_prob > 0.75 AND actual_prob < 0.3.
"""

import logging
import re
import threading
import time as _time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

LOG_PREFIX = "[TEMPORAL]"

# ── Statistical thresholds ──────────────────────────────────────────
# Interaction-log mining (existing)
MIN_OBSERVATIONS = 10
PEAK_THRESHOLD = 0.15
SIGNIFICANCE_RATIO = 2.0

# Ambient prediction (new)
MIN_OBSERVATIONS_PER_BUCKET = 10
MIN_PREDICTION_CONFIDENCE = 0.6
LAPLACE_ALPHA = 1.0

# Anomaly detection
ANOMALY_PREDICTED_THRESHOLD = 0.75
ANOMALY_ACTUAL_THRESHOLD = 0.3
ANOMALY_MIN_OBSERVATIONS = 20

# Retention
DEFAULT_RETENTION_DAYS = 90

# Known observation types and their possible values (for Laplace smoothing)
OBSERVATION_TYPES = {
    'attention': ['deep_focus', 'casual', 'distracted', 'away'],
    'energy': ['high', 'moderate', 'low'],
    'place': ['home', 'work', 'transit', 'out'],
    'tempo': ['rushed', 'relaxed', 'reflective'],
    'mobility': ['stationary', 'commuting', 'traveling'],
}

HOUR_LABELS = {
    range(5, 9): "early morning",
    range(9, 12): "morning",
    range(12, 14): "midday",
    range(14, 17): "afternoon",
    range(17, 20): "evening",
    range(20, 24): "night",
    range(0, 5): "late night",
}

DOW_LABELS = {
    0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
    4: "Friday", 5: "Saturday", 6: "Sunday",
}

SEGMENT_WEEKDAY = 'weekday'
SEGMENT_WEEKEND = 'weekend'
SEGMENT_ALL = 'all'

WEEKDAY_DAYS = {0, 1, 2, 3, 4}
WEEKEND_DAYS = {5, 6}


# ── Observation Buffer (thread-safe write serialization) ────────────

class TemporalObservationBuffer:
    """Thread-safe write buffer for ambient observations.

    Accumulates observations in memory, flushes to SQLite in a single
    transaction (both raw rows and aggregate UPSERTs). Avoids concurrent
    write contention on SQLite.
    """

    def __init__(self, max_buffer: int = 50):
        self._buffer: List[Dict] = []
        self._lock = threading.Lock()
        self._max_buffer = max_buffer
        self._write_errors = 0
        self._last_flush_time = 0.0

    def append(self, observation: dict):
        """Thread-safe append. Auto-flushes if buffer is full."""
        with self._lock:
            self._buffer.append(observation)
            if len(self._buffer) >= self._max_buffer:
                self._flush_locked()

    def flush(self, db_service=None):
        """Flush buffer to SQLite. Called by worker thread."""
        with self._lock:
            self._flush_locked(db_service)

    @property
    def write_errors_count(self) -> int:
        return self._write_errors

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._buffer)

    def _flush_locked(self, db_service=None):
        """Write buffer to SQLite. Must be called under lock.

        Single transaction: INSERT raw rows + UPSERT aggregate counts.
        Uses parameterized prepared statements.
        """
        if not self._buffer:
            return

        batch = self._buffer[:]
        self._buffer.clear()

        if db_service is None:
            try:
                from services.database_service import get_shared_db_service
                db_service = get_shared_db_service()
            except Exception as e:
                logger.warning(f"{LOG_PREFIX} Buffer flush: no DB available: {e}")
                self._write_errors += 1
                return

        try:
            with db_service.connection() as conn:
                cursor = conn.cursor()

                # 1. INSERT raw observations
                cursor.executemany(
                    """
                    INSERT INTO temporal_observations
                        (observation_type, observed_value, day_of_week, hour_bucket,
                         device_class, location_hash)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (o['observation_type'], o['observed_value'], o['day_of_week'],
                         o['hour_bucket'], o.get('device_class', ''),
                         o.get('location_hash', ''))
                        for o in batch
                    ]
                )

                # 2. UPSERT aggregate counts
                now_iso = datetime.utcnow().isoformat()
                for o in batch:
                    cursor.execute(
                        """
                        INSERT INTO temporal_aggregate
                            (user_id, observation_type, observed_value, day_of_week,
                             hour_bucket, device_class, count, last_seen)
                        VALUES ('primary', ?, ?, ?, ?, ?, 1, ?)
                        ON CONFLICT(user_id, observation_type, observed_value,
                                    day_of_week, hour_bucket, device_class)
                        DO UPDATE SET count = count + 1, last_seen = ?
                        """,
                        (o['observation_type'], o['observed_value'], o['day_of_week'],
                         o['hour_bucket'], o.get('device_class', ''),
                         now_iso, now_iso)
                    )

                conn.commit()
                cursor.close()

            self._last_flush_time = _time.time()
            logger.debug(f"{LOG_PREFIX} Flushed {len(batch)} observations to DB")

        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Buffer flush failed: {e}")
            self._write_errors += 1


# Module-level singleton buffer (shared across threads)
observation_buffer = TemporalObservationBuffer()


# ── Main Service ────────────────────────────────────────────────────

class TemporalPatternService:
    """Mines and stores temporal behavioral patterns from interaction history
    and ambient inference signals."""

    def __init__(self, database_service):
        self.db = database_service
        self._last_mining_duration = 0.0

    # ── Public API ──────────────────────────────────────────────────

    def mine_patterns(self, user_id: str = 'primary', lookback_days: int = 30) -> List[Dict]:
        """
        Main entry point. Mines patterns from both interaction_log and
        ambient aggregates, stores as user traits.

        Returns list of discovered patterns for logging.
        """
        start = _time.monotonic()
        cutoff = datetime.utcnow() - timedelta(days=lookback_days)

        patterns = []

        # 1. Activity time patterns from interaction_log (existing)
        hour_dist = self._get_hour_distribution(cutoff)
        patterns.extend(self._detect_peak_hours(hour_dist))

        # 2. Day-of-week patterns from interaction_log (existing)
        dow_dist = self._get_dow_distribution(cutoff)
        patterns.extend(self._detect_peak_days(dow_dist))

        # 3. Topic-time associations from interaction_log (existing)
        patterns.extend(self._get_topic_time_associations(cutoff))

        # 4. Ambient rhythm patterns from temporal_aggregate (new)
        for obs_type in OBSERVATION_TYPES:
            for segment in (SEGMENT_WEEKDAY, SEGMENT_WEEKEND):
                dist = self._get_ambient_distribution(obs_type, segment)
                ambient_patterns = self._detect_ambient_peaks(dist, obs_type, segment)
                patterns.extend(ambient_patterns)

        # 5. Transition patterns (new)
        patterns.extend(self._detect_transitions())

        # Store all as user traits
        self._store_patterns_as_traits(patterns, user_id)

        self._last_mining_duration = _time.monotonic() - start
        return patterns

    def predict(self, obs_type: str, day: Optional[int] = None,
                hour: Optional[int] = None) -> Optional[Dict]:
        """Predict most likely ambient value for a given time slot.

        Uses Laplace-smoothed confidence to avoid overconfident predictions
        on small samples:
            confidence = (top_count + alpha) / (total_count + alpha * num_values)

        Args:
            obs_type: 'attention', 'energy', 'place', 'tempo', 'mobility'
            day: 0=Monday..6=Sunday (defaults to current day)
            hour: 0-23 (defaults to current hour)

        Returns:
            {value, confidence, observation_count, runner_up, runner_up_confidence}
            or None if insufficient data.
        """
        if obs_type not in OBSERVATION_TYPES:
            return None

        now = datetime.utcnow()
        if day is None:
            day = now.weekday()
        if hour is None:
            hour = now.hour

        # Select segment based on day
        segment = SEGMENT_WEEKDAY if day in WEEKDAY_DAYS else SEGMENT_WEEKEND

        try:
            counts = self._get_bucket_counts(obs_type, day, hour, segment)
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} predict() query failed: {e}")
            return None

        total = sum(counts.values())
        if total < MIN_OBSERVATIONS_PER_BUCKET:
            return None

        num_values = len(OBSERVATION_TYPES[obs_type])
        alpha = LAPLACE_ALPHA

        # Laplace-smoothed probabilities
        smoothed = {
            v: (counts.get(v, 0) + alpha) / (total + alpha * num_values)
            for v in OBSERVATION_TYPES[obs_type]
        }

        # Sort by probability descending
        ranked = sorted(smoothed.items(), key=lambda x: x[1], reverse=True)
        top_value, top_confidence = ranked[0]
        runner_up, runner_up_confidence = ranked[1] if len(ranked) > 1 else (None, 0.0)

        if top_confidence < MIN_PREDICTION_CONFIDENCE:
            return None

        return {
            'value': top_value,
            'confidence': round(top_confidence, 3),
            'observation_count': total,
            'runner_up': runner_up,
            'runner_up_confidence': round(runner_up_confidence, 3),
        }

    def get_upcoming_transitions(self, lookahead_minutes: int = 60) -> List[Dict]:
        """Predict state changes at hour boundaries within lookahead window.

        Operates at hour granularity for reliable statistics.

        Returns:
            [{obs_type, from_value, to_value, expected_hour, confidence}]
        """
        now = datetime.utcnow()
        current_day = now.weekday()
        current_hour = now.hour

        hours_ahead = max(1, lookahead_minutes // 60)
        transitions = []

        for obs_type in OBSERVATION_TYPES:
            current_pred = self.predict(obs_type, current_day, current_hour)
            if not current_pred:
                continue

            for offset in range(1, hours_ahead + 1):
                future_hour = (current_hour + offset) % 24
                # Handle day rollover
                future_day = current_day if future_hour > current_hour else (current_day + 1) % 7

                future_pred = self.predict(obs_type, future_day, future_hour)
                if not future_pred:
                    continue

                if future_pred['value'] != current_pred['value']:
                    transitions.append({
                        'obs_type': obs_type,
                        'from_value': current_pred['value'],
                        'to_value': future_pred['value'],
                        'expected_hour': future_hour,
                        'confidence': min(current_pred['confidence'],
                                          future_pred['confidence']),
                    })
                    break  # Only first transition per type

        return transitions

    def get_rhythm_summary(self, user_id: str = 'primary') -> str:
        """Human-readable rhythm summary for prompt injection.

        Returns max 3 most salient lines, sanitized, total < 200 chars.
        """
        try:
            from services.user_trait_service import UserTraitService
            trait_service = UserTraitService(self.db)
            all_traits = trait_service.get_all_traits(user_id)
            traits = [t for t in all_traits if t.get('category') == 'behavioral_pattern']

            if not traits:
                return ''

            # Prioritize ambient rhythm patterns, then interaction patterns
            rhythm_keys = ['weekday_energy_rhythm', 'weekday_attention_rhythm',
                           'weekday_place_rhythm', 'weekend_energy_rhythm',
                           'active_hours', 'active_days']

            lines = []
            for key in rhythm_keys:
                for t in traits:
                    if t.get('trait_key') == key and t.get('trait_value'):
                        lines.append(t['trait_value'])
                        break
                if len(lines) >= 3:
                    break

            # If no priority matches, take top-confidence traits
            if not lines:
                sorted_traits = sorted(traits, key=lambda t: t.get('confidence', 0),
                                       reverse=True)
                for t in sorted_traits[:3]:
                    if t.get('trait_value'):
                        lines.append(t['trait_value'])

            if not lines:
                return ''

            # Sanitize: strip non-printable chars
            sanitized = '\n'.join(
                re.sub(r'[^\x20-\x7E]', '', line) for line in lines
            )
            return sanitized[:200]

        except Exception as e:
            logger.debug(f"{LOG_PREFIX} get_rhythm_summary failed: {e}")
            return ''

    def detect_anomaly(self, obs_type: str, current_value: str,
                       day: Optional[int] = None,
                       hour: Optional[int] = None) -> Optional[Dict]:
        """Detect behavioral anomaly using probability ratio.

        Flags anomaly when:
        - predicted_probability > ANOMALY_PREDICTED_THRESHOLD (strong pattern)
        - current_value_probability < ANOMALY_ACTUAL_THRESHOLD (rare for this slot)
        - total observations >= ANOMALY_MIN_OBSERVATIONS (sufficient data)

        Returns:
            {expected, actual, predicted_prob, actual_prob, total_count} or None
        """
        prediction = self.predict(obs_type, day, hour)
        if not prediction:
            return None
        if prediction['observation_count'] < ANOMALY_MIN_OBSERVATIONS:
            return None
        if prediction['confidence'] < ANOMALY_PREDICTED_THRESHOLD:
            return None

        # If current value IS the predicted value, no anomaly
        if current_value == prediction['value']:
            return None

        # Get probability of the current value in this bucket
        actual_prob = self._get_value_probability(obs_type, current_value, day, hour)
        if actual_prob >= ANOMALY_ACTUAL_THRESHOLD:
            return None  # Not anomalous enough

        return {
            'expected': prediction['value'],
            'actual': current_value,
            'predicted_prob': prediction['confidence'],
            'actual_prob': round(actual_prob, 3),
            'total_count': prediction['observation_count'],
        }

    def cleanup_old_observations(self, retention_days: int = DEFAULT_RETENTION_DAYS):
        """Delete raw observations older than retention period.

        Aggregates are permanent (bounded by value space, not time).
        """
        cutoff = (datetime.utcnow() - timedelta(days=retention_days)).isoformat()
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM temporal_observations WHERE recorded_at < ?",
                    (cutoff,)
                )
                deleted = cursor.rowcount
                conn.commit()
                cursor.close()
                if deleted > 0:
                    logger.info(f"{LOG_PREFIX} Cleaned up {deleted} observations older "
                                f"than {retention_days} days")
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Cleanup failed: {e}")

    def get_observation_stats(self) -> Dict[str, Any]:
        """Get statistics for observability endpoint."""
        stats = {
            'observation_count': 0,
            'aggregate_row_count': 0,
            'oldest_observation': None,
            'patterns_discovered': 0,
            'predictions_available': [],
            'last_mining_duration_seconds': round(self._last_mining_duration, 3),
            'buffer_pending': observation_buffer.pending_count,
            'write_errors_total': observation_buffer.write_errors_count,
        }
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()

                # Raw observation count
                cursor.execute("SELECT COUNT(*) FROM temporal_observations")
                row = cursor.fetchone()
                stats['observation_count'] = row[0] if row else 0

                # Oldest observation
                cursor.execute(
                    "SELECT MIN(recorded_at) FROM temporal_observations"
                )
                row = cursor.fetchone()
                stats['oldest_observation'] = row[0] if row and row[0] else None

                # Aggregate row count
                cursor.execute("SELECT COUNT(*) FROM temporal_aggregate")
                row = cursor.fetchone()
                stats['aggregate_row_count'] = row[0] if row else 0

                # Predictions available (types with enough data)
                for obs_type in OBSERVATION_TYPES:
                    cursor.execute(
                        """
                        SELECT SUM(count) FROM temporal_aggregate
                        WHERE observation_type = ?
                        """,
                        (obs_type,)
                    )
                    row = cursor.fetchone()
                    if row and row[0] and row[0] >= MIN_OBSERVATIONS_PER_BUCKET:
                        stats['predictions_available'].append(obs_type)

                # Behavioral pattern traits count
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM user_traits
                    WHERE category = 'behavioral_pattern'
                    """
                )
                row = cursor.fetchone()
                stats['patterns_discovered'] = row[0] if row else 0

                cursor.close()
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} get_observation_stats failed: {e}")

        return stats

    # ── Ambient distribution queries (read from aggregates) ─────────

    def _get_ambient_distribution(self, obs_type: str,
                                  segment: str = SEGMENT_ALL) -> Dict[Tuple[int, int], Dict[str, int]]:
        """Query temporal_aggregate for distribution of an observation type.

        Args:
            obs_type: 'attention', 'energy', 'place', 'tempo', 'mobility'
            segment: 'weekday' (0-4), 'weekend' (5-6), or 'all'

        Returns:
            {(day_of_week, hour_bucket): {value: count, ...}}
        """
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()

                if segment == SEGMENT_WEEKDAY:
                    day_filter = "AND day_of_week BETWEEN 0 AND 4"
                elif segment == SEGMENT_WEEKEND:
                    day_filter = "AND day_of_week BETWEEN 5 AND 6"
                else:
                    day_filter = ""

                cursor.execute(
                    f"""
                    SELECT observed_value, day_of_week, hour_bucket, SUM(count) AS total
                    FROM temporal_aggregate
                    WHERE observation_type = ? {day_filter}
                    GROUP BY observed_value, day_of_week, hour_bucket
                    """,
                    (obs_type,)
                )
                rows = cursor.fetchall()
                cursor.close()

                dist: Dict[Tuple[int, int], Dict[str, int]] = {}
                for value, dow, hour, total in rows:
                    key = (int(dow), int(hour))
                    dist.setdefault(key, {})[value] = int(total)

                return dist

        except Exception as e:
            logger.warning(f"{LOG_PREFIX} ambient distribution query failed: {e}")
            return {}

    def _detect_ambient_peaks(self, distribution: Dict[Tuple[int, int], Dict[str, int]],
                              obs_type: str, segment: str) -> List[Dict]:
        """Detect significant ambient patterns from aggregate distribution.

        For each observation type, finds the dominant value per hour across the segment,
        then groups into rhythm descriptions.
        """
        if not distribution:
            return []

        # Aggregate across days within segment to get hourly profile
        hourly_profile: Dict[int, Dict[str, int]] = {}
        for (dow, hour), value_counts in distribution.items():
            hourly_profile.setdefault(hour, {})
            for value, count in value_counts.items():
                hourly_profile[hour][value] = hourly_profile[hour].get(value, 0) + count

        # Find dominant value per hour
        hourly_dominant: Dict[int, Tuple[str, float]] = {}
        for hour, value_counts in hourly_profile.items():
            total = sum(value_counts.values())
            if total < MIN_OBSERVATIONS_PER_BUCKET:
                continue
            top_value = max(value_counts, key=value_counts.get)
            confidence = value_counts[top_value] / total
            if confidence >= MIN_PREDICTION_CONFIDENCE:
                hourly_dominant[hour] = (top_value, confidence)

        if not hourly_dominant:
            return []

        # Group contiguous hours with same dominant value into windows
        windows = self._group_dominant_windows(hourly_dominant)

        # Build rhythm description
        segment_prefix = 'weekday' if segment == SEGMENT_WEEKDAY else 'weekend'
        patterns = []

        # Pick the most significant windows (top 2 by span)
        windows.sort(key=lambda w: len(w['hours']), reverse=True)
        parts = []
        for w in windows[:3]:
            label = self._hour_to_label(w['hours'][0])
            parts.append(f"{w['value']} in the {label}")

        if parts:
            rhythm = f"Typically {', '.join(parts)}"
            avg_confidence = sum(
                hourly_dominant[h][1] for w in windows[:3] for h in w['hours']
                if h in hourly_dominant
            ) / max(1, sum(len(w['hours']) for w in windows[:3]))

            patterns.append({
                'key': f'{segment_prefix}_{obs_type}_rhythm',
                'value': rhythm,
                'confidence': min(0.9, round(avg_confidence, 2)),
            })

        return patterns

    def _detect_transitions(self) -> List[Dict]:
        """Detect recurring hour-to-hour transitions from aggregates.

        Finds cases where the dominant value changes between adjacent hours
        with high confidence in both slots.
        """
        patterns = []

        for obs_type in OBSERVATION_TYPES:
            for segment in (SEGMENT_WEEKDAY, SEGMENT_WEEKEND):
                dist = self._get_ambient_distribution(obs_type, segment)

                # Build hourly dominant profile
                hourly_profile: Dict[int, Dict[str, int]] = {}
                for (dow, hour), value_counts in dist.items():
                    hourly_profile.setdefault(hour, {})
                    for value, count in value_counts.items():
                        hourly_profile[hour][value] = hourly_profile[hour].get(value, 0) + count

                for hour in range(23):
                    curr_counts = hourly_profile.get(hour, {})
                    next_counts = hourly_profile.get(hour + 1, {})

                    curr_total = sum(curr_counts.values())
                    next_total = sum(next_counts.values())

                    if curr_total < MIN_OBSERVATIONS_PER_BUCKET or next_total < MIN_OBSERVATIONS_PER_BUCKET:
                        continue

                    curr_top = max(curr_counts, key=curr_counts.get) if curr_counts else None
                    next_top = max(next_counts, key=next_counts.get) if next_counts else None

                    if curr_top and next_top and curr_top != next_top:
                        curr_conf = curr_counts[curr_top] / curr_total
                        next_conf = next_counts[next_top] / next_total

                        if curr_conf >= 0.6 and next_conf >= 0.6:
                            segment_prefix = 'weekday' if segment == SEGMENT_WEEKDAY else 'weekend'
                            hour_label = self._hour_to_label(hour + 1)
                            patterns.append({
                                'key': f'{segment_prefix}_transition_{obs_type}',
                                'value': (f'{obs_type.capitalize()} typically shifts from '
                                          f'{curr_top} to {next_top} in the {hour_label}'),
                                'confidence': min(0.85, round(min(curr_conf, next_conf), 2)),
                            })
                            break  # One transition per obs_type per segment

        return patterns

    # ── Prediction helpers ──────────────────────────────────────────

    def _get_bucket_counts(self, obs_type: str, day: int, hour: int,
                           segment: str = SEGMENT_ALL) -> Dict[str, int]:
        """Get value counts for a specific bucket from temporal_aggregate.

        When segment is weekday/weekend, aggregates across all days in the segment
        for the given hour (not just the specific day).
        """
        with self.db.connection() as conn:
            cursor = conn.cursor()

            if segment == SEGMENT_WEEKDAY:
                cursor.execute(
                    """
                    SELECT observed_value, SUM(count) AS total
                    FROM temporal_aggregate
                    WHERE observation_type = ? AND hour_bucket = ?
                      AND day_of_week BETWEEN 0 AND 4
                    GROUP BY observed_value
                    """,
                    (obs_type, hour)
                )
            elif segment == SEGMENT_WEEKEND:
                cursor.execute(
                    """
                    SELECT observed_value, SUM(count) AS total
                    FROM temporal_aggregate
                    WHERE observation_type = ? AND hour_bucket = ?
                      AND day_of_week BETWEEN 5 AND 6
                    GROUP BY observed_value
                    """,
                    (obs_type, hour)
                )
            else:
                cursor.execute(
                    """
                    SELECT observed_value, SUM(count) AS total
                    FROM temporal_aggregate
                    WHERE observation_type = ? AND day_of_week = ? AND hour_bucket = ?
                    GROUP BY observed_value
                    """,
                    (obs_type, day, hour)
                )

            rows = cursor.fetchall()
            cursor.close()
            return {str(r[0]): int(r[1]) for r in rows}

    def _get_value_probability(self, obs_type: str, value: str,
                               day: Optional[int] = None,
                               hour: Optional[int] = None) -> float:
        """Get Laplace-smoothed probability of a specific value in a bucket."""
        now = datetime.utcnow()
        if day is None:
            day = now.weekday()
        if hour is None:
            hour = now.hour

        segment = SEGMENT_WEEKDAY if day in WEEKDAY_DAYS else SEGMENT_WEEKEND

        try:
            counts = self._get_bucket_counts(obs_type, day, hour, segment)
        except Exception:
            return 0.0

        total = sum(counts.values())
        if total == 0:
            return 0.0

        num_values = len(OBSERVATION_TYPES.get(obs_type, [value]))
        alpha = LAPLACE_ALPHA
        return (counts.get(value, 0) + alpha) / (total + alpha * num_values)

    # ── Interaction-log queries (existing, unchanged) ───────────────

    def _get_hour_distribution(self, cutoff: datetime) -> Dict[int, int]:
        """Query interaction_log for hourly distribution of user_input events."""
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT CAST(strftime('%H', created_at) AS INTEGER) AS hour, COUNT(*) AS cnt
                    FROM interaction_log
                    WHERE event_type = 'user_input'
                      AND created_at >= ?
                    GROUP BY hour
                    ORDER BY hour
                    """,
                    (cutoff.isoformat(),)
                )
                rows = cursor.fetchall()
                cursor.close()
                return {int(r[0]): int(r[1]) for r in rows}
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} hour distribution query failed: {e}")
            return {}

    def _get_dow_distribution(self, cutoff: datetime) -> Dict[int, int]:
        """Query interaction_log for day-of-week distribution."""
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT CAST(strftime('%w', created_at) AS INTEGER) AS dow, COUNT(*) AS cnt
                    FROM interaction_log
                    WHERE event_type = 'user_input'
                      AND created_at >= ?
                    GROUP BY dow
                    ORDER BY dow
                    """,
                    (cutoff.isoformat(),)
                )
                rows = cursor.fetchall()
                cursor.close()
                # SQLite DOW: 0=Sunday, adjust to 0=Monday
                return {(int(r[0]) - 1) % 7: int(r[1]) for r in rows}
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} dow distribution query failed: {e}")
            return {}

    def _get_topic_time_associations(self, cutoff: datetime) -> List[Dict]:
        """Find topics that cluster at specific times of day."""
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT topic,
                           CAST(strftime('%H', created_at) AS INTEGER) AS hour,
                           COUNT(*) AS cnt
                    FROM interaction_log
                    WHERE event_type = 'user_input'
                      AND created_at >= ?
                      AND topic IS NOT NULL
                    GROUP BY topic, hour
                    HAVING COUNT(*) >= 3
                    ORDER BY cnt DESC
                    LIMIT 50
                    """,
                    (cutoff.isoformat(),)
                )
                rows = cursor.fetchall()
                cursor.close()

            # Group by topic, find peak hours per topic
            topic_hours: Dict[str, Dict[int, int]] = {}
            for topic, hour, cnt in rows:
                topic_hours.setdefault(topic, {})[int(hour)] = int(cnt)

            patterns = []
            for topic, hours in topic_hours.items():
                total = sum(hours.values())
                if total < MIN_OBSERVATIONS:
                    continue
                avg = total / 24.0
                for hour, cnt in hours.items():
                    ratio = cnt / max(avg, 1)
                    if ratio >= SIGNIFICANCE_RATIO and cnt / total >= PEAK_THRESHOLD:
                        hour_label = self._hour_to_label(hour)
                        safe_topic = self._slugify_topic(topic)
                        patterns.append({
                            'key': f'topic_time_{safe_topic}',
                            'value': f'Often discusses {topic} in the {hour_label}',
                            'confidence': min(0.9, 0.5 + (ratio - SIGNIFICANCE_RATIO) * 0.1),
                        })

            return patterns

        except Exception as e:
            logger.warning(f"{LOG_PREFIX} topic-time query failed: {e}")
            return []

    # ── Interaction-log detection (existing, unchanged) ─────────────

    def _detect_peak_hours(self, hour_dist: Dict[int, int]) -> List[Dict]:
        """Detect significant hour-of-day peaks."""
        total = sum(hour_dist.values())
        if total < MIN_OBSERVATIONS:
            return []

        avg = total / 24.0
        peak_hours = []
        for hour in range(24):
            cnt = hour_dist.get(hour, 0)
            ratio = cnt / max(avg, 1)
            if ratio >= SIGNIFICANCE_RATIO and cnt / total >= PEAK_THRESHOLD:
                peak_hours.append(hour)

        if not peak_hours:
            return []

        patterns = []
        windows = self._group_contiguous(peak_hours)
        for window in windows:
            start_label = self._hour_to_label(window[0])
            peak_pct = sum(hour_dist.get(h, 0) for h in window) / total
            patterns.append({
                'key': 'active_hours',
                'value': f'Most active in the {start_label} ({int(peak_pct * 100)}% of interactions)',
                'confidence': min(0.9, 0.5 + peak_pct),
            })

        return patterns

    def _detect_peak_days(self, dow_dist: Dict[int, int]) -> List[Dict]:
        """Detect significant day-of-week peaks."""
        total = sum(dow_dist.values())
        if total < MIN_OBSERVATIONS:
            return []

        avg = total / 7.0
        peak_days = []
        for dow in range(7):
            cnt = dow_dist.get(dow, 0)
            ratio = cnt / max(avg, 1)
            if ratio >= SIGNIFICANCE_RATIO:
                peak_days.append(DOW_LABELS[dow])

        if not peak_days:
            return []

        return [{
            'key': 'active_days',
            'value': f'Most active on {", ".join(peak_days)}',
            'confidence': 0.7,
        }]

    # ── Trait storage (existing, unchanged) ─────────────────────────

    def _store_patterns_as_traits(self, patterns: List[Dict], user_id: str):
        """Store discovered patterns as user traits.

        Uses store_trait() which handles deduplication internally:
        - Same value observed again → reinforces confidence
        - Different value → overwrites if new confidence > 2x old
        """
        if not patterns:
            return

        try:
            from services.user_trait_service import UserTraitService
            trait_service = UserTraitService(self.db)

            stored = 0
            for p in patterns:
                result = trait_service.store_trait(
                    trait_key=p['key'],
                    trait_value=p['value'],
                    confidence=p['confidence'],
                    category='behavioral_pattern',
                    source='inferred',
                    is_literal=True,
                    user_id=user_id,
                )
                if result:
                    stored += 1

            logger.info(f"{LOG_PREFIX} Stored/reinforced {stored}/{len(patterns)} behavioral patterns")

        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Failed to store patterns: {e}")

    # ── Static helpers ──────────────────────────────────────────────

    @staticmethod
    def _hour_to_label(hour: int) -> str:
        """Convert hour (0-23) to generalized human-readable label.

        CRITICAL: Always use broad time-of-day labels, never specific hours.
        "Most active in the evenings" is fine.
        "Active between 10-11pm" feels invasive.
        """
        for hour_range, label in HOUR_LABELS.items():
            if hour in hour_range:
                return label
        return "daytime"

    @staticmethod
    def _slugify_topic(topic: str, max_len: int = 40) -> str:
        """Slugify and cap topic length for clean trait keys."""
        slug = re.sub(r'[^a-z0-9]+', '_', topic.lower().strip())
        slug = slug.strip('_')
        return slug[:max_len]

    @staticmethod
    def _group_contiguous(hours: List[int]) -> List[List[int]]:
        """Group contiguous hours into windows."""
        if not hours:
            return []
        groups = [[hours[0]]]
        for h in hours[1:]:
            if h == groups[-1][-1] + 1:
                groups[-1].append(h)
            else:
                groups.append([h])
        return groups

    @staticmethod
    def _group_dominant_windows(hourly_dominant: Dict[int, Tuple[str, float]]) -> List[Dict]:
        """Group contiguous hours with the same dominant value into windows."""
        if not hourly_dominant:
            return []

        sorted_hours = sorted(hourly_dominant.keys())
        windows = []
        current_window = {
            'value': hourly_dominant[sorted_hours[0]][0],
            'hours': [sorted_hours[0]],
        }

        for h in sorted_hours[1:]:
            if (h == current_window['hours'][-1] + 1 and
                    hourly_dominant[h][0] == current_window['value']):
                current_window['hours'].append(h)
            else:
                windows.append(current_window)
                current_window = {
                    'value': hourly_dominant[h][0],
                    'hours': [h],
                }

        windows.append(current_window)
        return windows


# ── Worker ──────────────────────────────────────────────────────────

def temporal_pattern_worker(shared_state):
    """6h cycle: mine temporal patterns + flush observation buffer + cleanup."""
    import random
    from services.database_service import get_shared_db_service
    from services.memory_client import MemoryClientService

    _time.sleep(300)  # 5min warmup — avoid startup load

    store = MemoryClientService.create_connection()
    throttle_key = "temporal:mining_throttle"

    while True:
        try:
            # Check if mining is throttled (admin can set this during heavy load)
            if store.get(throttle_key):
                logger.info(f"{LOG_PREFIX} Mining throttled, skipping cycle")
            else:
                db = get_shared_db_service()

                # 1. Flush any pending observations
                observation_buffer.flush(db)

                # 2. Mine patterns (reads aggregates — fast)
                service = TemporalPatternService(db)
                patterns = service.mine_patterns()
                logger.info(
                    f"{LOG_PREFIX} Mined {len(patterns)} patterns "
                    f"in {service._last_mining_duration:.2f}s"
                )

                # 3. Cleanup raw observations older than retention period
                service.cleanup_old_observations()

                # 4. Store last run time for observability
                store.setex("temporal:last_mining_run",
                            86400, datetime.utcnow().isoformat())

        except Exception as e:
            logger.error(f"{LOG_PREFIX} Mining cycle failed: {e}")

        # 6h base cycle ±30% jitter
        cycle = 6 * 3600
        _time.sleep(cycle * random.uniform(0.7, 1.3))
