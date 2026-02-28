"""
Temporal Pattern Service — mines behavioral patterns from interaction history.

Analyzes hour-of-day and day-of-week distributions in interaction_log to detect
statistically significant patterns. Stores discovered patterns as user traits
with category 'behavioral_pattern', leveraging the existing trait injection pipeline.

Design principles:
- Generalization over specificity: all time references use broad labels ("evenings"),
  never specific hours ("10-11pm"). Even correct patterns can feel invasive if too precise.
- Deduplication via store_trait(): UserTraitService.store_trait() handles conflict
  resolution (reinforce same value, overwrite if confidence > 2x). Repeated mining
  cycles won't create duplicates.
- Slugified trait keys: topic names are slugified and capped at 40 chars.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List

logger = logging.getLogger(__name__)

# Minimum observations needed to consider a pattern significant
MIN_OBSERVATIONS = 10
# A time bucket must have >X% of total to count as a peak
PEAK_THRESHOLD = 0.15
# Minimum concentration ratio (peak bucket / average) to be "significant"
SIGNIFICANCE_RATIO = 2.0

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


class TemporalPatternService:
    """Mines and stores temporal behavioral patterns from interaction history."""

    def __init__(self, database_service):
        self.db = database_service

    def mine_patterns(self, user_id: str = 'primary', lookback_days: int = 30) -> List[Dict]:
        """
        Main entry point. Mines patterns and stores as user traits.

        Returns list of discovered patterns for logging.
        """
        cutoff = datetime.utcnow() - timedelta(days=lookback_days)

        patterns = []

        # 1. Activity time patterns (when does the user chat?)
        hour_dist = self._get_hour_distribution(cutoff)
        hour_patterns = self._detect_peak_hours(hour_dist)
        patterns.extend(hour_patterns)

        # 2. Day-of-week patterns
        dow_dist = self._get_dow_distribution(cutoff)
        dow_patterns = self._detect_peak_days(dow_dist)
        patterns.extend(dow_patterns)

        # 3. Topic-time associations (what topics at what times?)
        topic_time = self._get_topic_time_associations(cutoff)
        patterns.extend(topic_time)

        # Store as user traits
        self._store_patterns_as_traits(patterns, user_id)

        return patterns

    def _get_hour_distribution(self, cutoff: datetime) -> Dict[int, int]:
        """Query interaction_log for hourly distribution of user_input events."""
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT EXTRACT(HOUR FROM created_at)::int AS hour, COUNT(*) AS cnt
                    FROM interaction_log
                    WHERE event_type = 'user_input'
                      AND created_at >= %s
                    GROUP BY hour
                    ORDER BY hour
                    """,
                    (cutoff,)
                )
                rows = cursor.fetchall()
                cursor.close()
                return {int(r[0]): int(r[1]) for r in rows}
        except Exception as e:
            logger.warning(f"[TEMPORAL] hour distribution query failed: {e}")
            return {}

    def _get_dow_distribution(self, cutoff: datetime) -> Dict[int, int]:
        """Query interaction_log for day-of-week distribution."""
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT EXTRACT(DOW FROM created_at)::int AS dow, COUNT(*) AS cnt
                    FROM interaction_log
                    WHERE event_type = 'user_input'
                      AND created_at >= %s
                    GROUP BY dow
                    ORDER BY dow
                    """,
                    (cutoff,)
                )
                rows = cursor.fetchall()
                cursor.close()
                # PostgreSQL DOW: 0=Sunday, adjust to 0=Monday
                return {(int(r[0]) - 1) % 7: int(r[1]) for r in rows}
        except Exception as e:
            logger.warning(f"[TEMPORAL] dow distribution query failed: {e}")
            return {}

    def _get_topic_time_associations(self, cutoff: datetime) -> List[Dict]:
        """Find topics that cluster at specific times of day."""
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT topic,
                           EXTRACT(HOUR FROM created_at)::int AS hour,
                           COUNT(*) AS cnt
                    FROM interaction_log
                    WHERE event_type = 'user_input'
                      AND created_at >= %s
                      AND topic IS NOT NULL
                    GROUP BY topic, hour
                    HAVING COUNT(*) >= 3
                    ORDER BY cnt DESC
                    LIMIT 50
                    """,
                    (cutoff,)
                )
                rows = cursor.fetchall()
                cursor.close()

            # Group by topic, find peak hours per topic
            topic_hours = {}
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
            logger.warning(f"[TEMPORAL] topic-time query failed: {e}")
            return []

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

    def _store_patterns_as_traits(self, patterns: List[Dict], user_id: str):
        """
        Store discovered patterns as user traits.

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

            logger.info(f"[TEMPORAL] Stored/reinforced {stored}/{len(patterns)} behavioral patterns")

        except Exception as e:
            logger.warning(f"[TEMPORAL] Failed to store patterns: {e}")

    @staticmethod
    def _hour_to_label(hour: int) -> str:
        """
        Convert hour (0-23) to generalized human-readable label.

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
        import re
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


def temporal_pattern_worker(shared_state):
    """24h cycle: mine temporal behavioral patterns from interaction history."""
    import time
    from services.database_service import get_shared_db_service

    cycle_seconds = 24 * 3600
    time.sleep(300)  # 5min warmup — avoid startup load

    while True:
        try:
            db = get_shared_db_service()
            service = TemporalPatternService(db)
            patterns = service.mine_patterns()
            logging.info(f"[TEMPORAL-PATTERN] Mined {len(patterns)} patterns")
        except Exception as e:
            logging.error(f"[TEMPORAL-PATTERN] Mining failed: {e}")

        time.sleep(cycle_seconds)
