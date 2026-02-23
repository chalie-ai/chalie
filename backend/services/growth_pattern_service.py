# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Growth Pattern Service - Periodic service that tracks longitudinal shifts in the user's
communication style and detects meaningful growth signals.

Runs every 30 minutes (same cycle as decay engine). Compares current communication style
against a slowly-updated baseline to detect persistent shifts in how the user thinks,
communicates, and engages.

Detected signals are stored as user traits (category='core') and surfaced sparingly
by AdaptiveLayerService as growth reflections in the response prompt.
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Dimensions tracked for growth detection
TRACKED_DIMENSIONS = [
    'certainty_level',
    'depth_preference',
    'challenge_appetite',
    'verbosity',
    'formality',
]

# A delta must persist for at least this many consecutive cycles to qualify as a signal
SIGNIFICANCE_CONSECUTIVE_CYCLES = 3

# Minimum magnitude shift to consider a delta significant (on 1-10 scale)
SIGNIFICANCE_MAGNITUDE = 1.0

# Baseline update rate (very slow EMA — 0.1 new + 0.9 old)
BASELINE_EMA_WEIGHT = 0.1

# Service cycle interval in seconds (30 minutes)
DEFAULT_INTERVAL = 1800


class GrowthPatternService:
    """
    Periodic background service that detects longitudinal communication style shifts.

    Cycle:
    1. Read current communication style from user_traits
    2. Compare against stored style baseline
    3. Detect dimensions with persistent significant shifts
    4. Store/update growth signals (category='core', key='growth_signal:{dim}')
    5. Slowly update baseline (EMA)
    6. Log detections to interaction_log
    """

    def __init__(self, interval: int = DEFAULT_INTERVAL):
        self.interval = interval
        logger.info(f"[GROWTH PATTERN] Initialized (interval={interval}s)")

    def run(self, shared_state: Optional[dict] = None) -> None:
        """Main service loop."""
        logger.info("[GROWTH PATTERN] Service started")

        while True:
            try:
                time.sleep(self.interval)
                logger.info("[GROWTH PATTERN] Running growth cycle...")
                self.run_growth_cycle()
            except KeyboardInterrupt:
                logger.info("[GROWTH PATTERN] Service shutting down...")
                break
            except Exception as e:
                logger.error(f"[GROWTH PATTERN] Error: {e}", exc_info=True)
                time.sleep(60)

    def run_growth_cycle(self, user_id: str = 'primary') -> dict:
        """
        Run one full growth detection cycle.

        Returns:
            dict with 'signals_detected', 'baseline_updated', 'errors'
        """
        result = {'signals_detected': 0, 'baseline_updated': False, 'errors': []}

        try:
            from services.user_trait_service import UserTraitService
            from services.database_service import get_shared_db_service

            db_service = get_shared_db_service()
            trait_service = UserTraitService(db_service)

            current_style = trait_service.get_communication_style(user_id=user_id)
            if not current_style:
                logger.debug("[GROWTH PATTERN] No communication style found, skipping cycle")
                return result

            # Require enough observations before comparing against baseline
            obs_count = current_style.get('_observation_count', 0)
            if obs_count < 5:
                logger.debug(f"[GROWTH PATTERN] Only {obs_count} observations, waiting for more data")
                return result

            baseline = self._get_baseline(user_id, db_service)

            if not baseline:
                # First cycle — store current style as baseline and exit
                self._store_baseline(user_id, current_style, db_service)
                logger.info("[GROWTH PATTERN] Baseline initialized")
                return result

            # Compute deltas and detect significant shifts
            deltas = self._compute_deltas(current_style, baseline)
            significant = [d for d in deltas if d['magnitude'] >= SIGNIFICANCE_MAGNITUDE]

            # Update or create growth signals
            for delta in significant:
                stored = self._update_growth_signal(delta, user_id, db_service)
                if stored:
                    result['signals_detected'] += 1
                    self._log_growth_signal(delta, user_id)

            # Prune weak growth signals (those that reversed direction)
            self._prune_reversed_signals(deltas, user_id, db_service)

            # Slowly update baseline
            updated = self._update_baseline_slowly(current_style, baseline, user_id, db_service)
            result['baseline_updated'] = updated

            logger.info(
                f"[GROWTH PATTERN] Cycle complete: "
                f"{result['signals_detected']} signals detected, "
                f"baseline_updated={updated}"
            )

        except Exception as e:
            logger.error(f"[GROWTH PATTERN] Cycle error: {e}", exc_info=True)
            result['errors'].append(str(e))

        return result

    def _get_baseline(self, user_id: str, db_service) -> Optional[dict]:
        """Read stored style baseline from user_traits."""
        try:
            with db_service.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT trait_value FROM user_traits "
                    "WHERE user_id = %s AND trait_key = 'style_baseline' LIMIT 1",
                    (user_id,)
                )
                row = cursor.fetchone()
                cursor.close()
                if row and row[0]:
                    return json.loads(row[0])
                return None
        except Exception as e:
            logger.warning(f"[GROWTH PATTERN] Failed to read baseline: {e}")
            return None

    def _store_baseline(self, user_id: str, style: dict, db_service) -> None:
        """Store communication style as the initial baseline."""
        try:
            from services.user_trait_service import UserTraitService
            trait_service = UserTraitService(db_service)
            # Only store the numeric dimensions, not metadata keys
            baseline = {
                k: v for k, v in style.items()
                if k in TRACKED_DIMENSIONS and isinstance(v, (int, float))
            }
            trait_service.store_trait(
                trait_key='style_baseline',
                trait_value=json.dumps(baseline),
                confidence=0.8,
                category='core',
                source='inferred',
                is_literal=True,
                user_id=user_id,
            )
        except Exception as e:
            logger.warning(f"[GROWTH PATTERN] Failed to store baseline: {e}")

    def _compute_deltas(self, current: dict, baseline: dict) -> list:
        """Compute per-dimension deltas between current style and baseline."""
        deltas = []
        for dim in TRACKED_DIMENSIONS:
            current_val = current.get(dim)
            baseline_val = baseline.get(dim)
            if current_val is None or baseline_val is None:
                continue
            if not isinstance(current_val, (int, float)) or not isinstance(baseline_val, (int, float)):
                continue
            magnitude = abs(current_val - baseline_val)
            direction = 'increasing' if current_val > baseline_val else 'decreasing'
            deltas.append({
                'dimension': dim,
                'current': round(float(current_val), 2),
                'baseline': round(float(baseline_val), 2),
                'magnitude': round(magnitude, 2),
                'direction': direction,
            })
        return deltas

    def _update_growth_signal(self, delta: dict, user_id: str, db_service) -> bool:
        """
        Create or update a growth signal trait for a significant dimension shift.

        Returns True if a signal was stored/updated.
        """
        try:
            dim = delta['dimension']
            trait_key = f"growth_signal:{dim}"

            # Read existing signal
            existing_signal = None
            try:
                with db_service.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT trait_value FROM user_traits "
                        "WHERE user_id = %s AND trait_key = %s LIMIT 1",
                        (user_id, trait_key)
                    )
                    row = cursor.fetchone()
                    cursor.close()
                    if row and row[0]:
                        existing_signal = json.loads(row[0])
            except Exception:
                pass

            now_iso = datetime.now(timezone.utc).isoformat()

            if existing_signal:
                # If direction matches, increment consecutive cycles
                if existing_signal.get('direction') == delta['direction']:
                    consecutive = existing_signal.get('consecutive_cycles', 1) + 1
                    signal_data = {
                        **existing_signal,
                        'magnitude': delta['magnitude'],
                        'consecutive_cycles': consecutive,
                        'last_seen': now_iso,
                    }
                else:
                    # Direction reversed — reset
                    signal_data = {
                        'dimension': dim,
                        'direction': delta['direction'],
                        'magnitude': delta['magnitude'],
                        'first_detected': now_iso,
                        'last_seen': now_iso,
                        'consecutive_cycles': 1,
                    }
            else:
                signal_data = {
                    'dimension': dim,
                    'direction': delta['direction'],
                    'magnitude': delta['magnitude'],
                    'first_detected': now_iso,
                    'last_seen': now_iso,
                    'consecutive_cycles': 1,
                }

            from services.user_trait_service import UserTraitService
            trait_service = UserTraitService(db_service)
            trait_service.store_trait(
                trait_key=trait_key,
                trait_value=json.dumps(signal_data),
                confidence=0.7,
                category='core',
                source='inferred',
                is_literal=True,
                user_id=user_id,
            )
            return True

        except Exception as e:
            logger.warning(f"[GROWTH PATTERN] Failed to update growth signal: {e}")
            return False

    def _prune_reversed_signals(self, current_deltas: list, user_id: str, db_service) -> None:
        """
        Remove growth signals for dimensions that are no longer showing a significant shift
        (magnitude has dropped below threshold, indicating the user reverted).
        """
        try:
            significant_dims = {d['dimension'] for d in current_deltas if d['magnitude'] >= SIGNIFICANCE_MAGNITUDE}

            with db_service.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT trait_key FROM user_traits "
                    "WHERE user_id = %s AND trait_key LIKE 'growth_signal:%%' AND category = 'core'",
                    (user_id,)
                )
                rows = cursor.fetchall()
                cursor.close()

            for (trait_key,) in rows:
                dim = trait_key.replace('growth_signal:', '')
                if dim not in significant_dims:
                    # Dimension no longer significant — let decay handle it naturally
                    # (category='core' has floor=0.3, so it won't be deleted immediately)
                    pass  # Decay engine will handle confidence reduction over time

        except Exception as e:
            logger.debug(f"[GROWTH PATTERN] Prune check failed: {e}")

    def _update_baseline_slowly(
        self, current: dict, baseline: dict, user_id: str, db_service
    ) -> bool:
        """Slowly move baseline toward current style using a very slow EMA."""
        try:
            updated_baseline = {}
            for dim in TRACKED_DIMENSIONS:
                current_val = current.get(dim)
                baseline_val = baseline.get(dim)
                if current_val is None or baseline_val is None:
                    if baseline_val is not None:
                        updated_baseline[dim] = baseline_val
                    continue
                updated_baseline[dim] = round(
                    BASELINE_EMA_WEIGHT * float(current_val) + (1 - BASELINE_EMA_WEIGHT) * float(baseline_val),
                    2
                )

            from services.user_trait_service import UserTraitService
            trait_service = UserTraitService(db_service)
            trait_service.store_trait(
                trait_key='style_baseline',
                trait_value=json.dumps(updated_baseline),
                confidence=0.8,
                category='core',
                source='inferred',
                is_literal=True,
                user_id=user_id,
            )
            return True
        except Exception as e:
            logger.warning(f"[GROWTH PATTERN] Failed to update baseline: {e}")
            return False

    def _log_growth_signal(self, delta: dict, user_id: str) -> None:
        """Log growth signal detection to interaction_log for observability."""
        try:
            from services.interaction_log_service import InteractionLogService
            log_service = InteractionLogService()
            log_service.log_event(
                event_type='growth_signal_detected',
                payload={
                    'dimension': delta['dimension'],
                    'direction': delta['direction'],
                    'magnitude': delta['magnitude'],
                    'user_id': user_id,
                },
                topic='general',
                source='growth_pattern_service',
            )
        except Exception:
            pass  # Non-critical — logging failure should not interrupt the cycle


def growth_pattern_worker(shared_state: Optional[dict] = None) -> None:
    """
    Module-level entry point for multiprocessing (consumer.py pattern).
    Instantiates the service inside the child process.
    """
    service = GrowthPatternService(interval=DEFAULT_INTERVAL)
    service.run(shared_state)
