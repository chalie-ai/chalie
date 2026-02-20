"""
Identity Service - Dynamic identity control vectors.

Manages 6 personality dimensions with baseline + activation levels.
Activation is modulated by dual-channel reinforcement (emotion + reward)
and pulled toward baseline by inertia. Baseline drifts slowly under
sustained, consistent reinforcement.
"""

import json
import logging
import statistics
from datetime import datetime, timedelta

from .config_service import ConfigService

logger = logging.getLogger(__name__)


class IdentityService:
    """Manages identity control vectors: read, update, inertia, drift, coherence."""

    def __init__(self, database_service):
        self.db = database_service

        # Load config
        try:
            config = ConfigService.get_agent_config("identity")
        except Exception:
            config = {}

        reinforcement_cfg = config.get('reinforcement', {})
        self.signal_history_size = reinforcement_cfg.get('signal_history_size', 20)
        self.emotion_weight = reinforcement_cfg.get('emotion_weight', 0.6)
        self.reward_weight = reinforcement_cfg.get('reward_weight', 0.4)

        drift_cfg = config.get('baseline_drift', {})
        self.drift_rate = drift_cfg.get('rate', 0.005)
        self.reinforcement_threshold = drift_cfg.get('reinforcement_threshold', 10)
        self.max_drift_per_day = drift_cfg.get('max_drift_per_day', 0.02)
        self.direction_consistency_min = drift_cfg.get('direction_consistency_min', 0.7)
        self.variance_max = drift_cfg.get('variance_max', 0.15)

        coherence_cfg = config.get('coherence', {})
        self.relational_constraints = coherence_cfg.get('relational_constraints', [
            {"a": "assertiveness", "b": "warmth", "type": "floor_ratio", "a_threshold": 0.75, "b_floor": 0.35, "nudge": 0.05},
            {"a": "skepticism", "b": "warmth", "type": "floor_ratio", "a_threshold": 0.75, "b_floor": 0.35, "nudge": 0.05},
            {"a": "assertiveness", "b": "skepticism", "type": "ceiling_pair", "threshold": 0.75, "target": 0.7},
        ])

    def get_vectors(self) -> dict:
        """Load all identity vectors from DB.

        Returns:
            dict: {name: {baseline_weight, current_activation, plasticity_rate,
                          inertia_rate, min_cap, max_cap, reinforcement_count, ...}}
        """
        vectors = {}
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT vector_name, baseline_weight, current_activation,
                           plasticity_rate, inertia_rate, min_cap, max_cap,
                           reinforcement_count, signal_history,
                           baseline_drift_today, drift_window_start
                    FROM identity_vectors
                """)
                for row in cursor.fetchall():
                    vectors[row[0]] = {
                        'baseline_weight': row[1],
                        'current_activation': row[2],
                        'plasticity_rate': row[3],
                        'inertia_rate': row[4],
                        'min_cap': row[5],
                        'max_cap': row[6],
                        'reinforcement_count': row[7],
                        'signal_history': row[8] if isinstance(row[8], list) else json.loads(row[8] or '[]'),
                        'baseline_drift_today': row[9] or 0.0,
                        'drift_window_start': row[10],
                    }
                cursor.close()
        except Exception as e:
            logger.error(f"[IDENTITY] Failed to load vectors: {e}")
        return vectors

    def update_activation(self, vector_name: str, emotion_signal: float, reward_signal: float, topic: str = None):
        """
        Dual-channel reinforcement update.

        total_signal = (emotion_signal * 0.6) + (reward_signal * 0.4)
        delta = total_signal * plasticity_rate
        current_activation = clamp(current_activation + delta, min_cap, max_cap)
        """
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()

                # Read current state
                cursor.execute("""
                    SELECT current_activation, plasticity_rate, min_cap, max_cap,
                           reinforcement_count, signal_history
                    FROM identity_vectors WHERE vector_name = %s
                """, (vector_name,))
                row = cursor.fetchone()
                if not row:
                    logger.warning(f"[IDENTITY] Unknown vector: {vector_name}")
                    cursor.close()
                    return

                old_activation = row[0]
                plasticity_rate = row[1]
                min_cap = row[2]
                max_cap = row[3]
                reinforcement_count = row[4] or 0
                signal_history = row[5] if isinstance(row[5], list) else json.loads(row[5] or '[]')

                # Compute total signal
                total_signal = (emotion_signal * self.emotion_weight) + (reward_signal * self.reward_weight)
                delta = total_signal * plasticity_rate
                new_activation = max(min_cap, min(max_cap, old_activation + delta))

                # Update signal history (bounded ring buffer)
                signal_history.append(total_signal)
                if len(signal_history) > self.signal_history_size:
                    signal_history = signal_history[-self.signal_history_size:]

                # Write back
                cursor.execute("""
                    UPDATE identity_vectors
                    SET current_activation = %s,
                        reinforcement_count = %s,
                        signal_history = %s,
                        last_updated_at = NOW()
                    WHERE vector_name = %s
                """, (new_activation, reinforcement_count + 1, json.dumps(signal_history), vector_name))

                # Log event
                if abs(new_activation - old_activation) > 0.001:
                    self._log_event(cursor, vector_name, old_activation, new_activation,
                                    'reinforcement', total_signal, topic)

                cursor.close()

                # Evaluate baseline drift after sufficient reinforcements
                if (reinforcement_count + 1) >= self.reinforcement_threshold:
                    self.evaluate_baseline_drift(vector_name)

        except Exception as e:
            logger.error(f"[IDENTITY] Failed to update activation for {vector_name}: {e}")

    def apply_inertia(self) -> int:
        """
        Pull all activations toward their baselines. Called by decay engine.

        Formula: current_activation += (baseline_weight - current_activation) * inertia_rate

        Returns:
            Number of vectors adjusted
        """
        count = 0
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT vector_name, current_activation, baseline_weight,
                           inertia_rate, min_cap, max_cap
                    FROM identity_vectors
                """)
                rows = cursor.fetchall()

                for row in rows:
                    name, activation, baseline, inertia_rate, min_cap, max_cap = row
                    diff = baseline - activation
                    if abs(diff) < 0.005:
                        continue

                    new_activation = activation + diff * inertia_rate
                    new_activation = max(min_cap, min(max_cap, new_activation))

                    if abs(new_activation - activation) > 0.001:
                        cursor.execute("""
                            UPDATE identity_vectors
                            SET current_activation = %s, last_updated_at = NOW()
                            WHERE vector_name = %s
                        """, (new_activation, name))
                        self._log_event(cursor, name, activation, new_activation, 'inertia', diff * inertia_rate)
                        count += 1

                cursor.close()

            if count > 0:
                logger.info(f"[IDENTITY] Inertia applied to {count} vectors")
        except Exception as e:
            logger.error(f"[IDENTITY] Inertia failed: {e}")
        return count

    def evaluate_baseline_drift(self, vector_name: str):
        """
        Stability-gated baseline drift. Only shifts when ALL conditions met:
        1. reinforcement_count >= threshold
        2. Last N signals have consistent direction (>70% same sign)
        3. Signal variance < threshold
        4. Cumulative drift today < max_baseline_drift_per_day
        """
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT baseline_weight, signal_history, reinforcement_count,
                           baseline_drift_today, drift_window_start, min_cap, max_cap
                    FROM identity_vectors WHERE vector_name = %s
                """, (vector_name,))
                row = cursor.fetchone()
                if not row:
                    cursor.close()
                    return

                baseline = row[0]
                signal_history = row[1] if isinstance(row[1], list) else json.loads(row[1] or '[]')
                reinforcement_count = row[2] or 0
                baseline_drift_today = row[3] or 0.0
                drift_window_start = row[4]
                min_cap = row[5]
                max_cap = row[6]

                # Gate 1: enough reinforcements
                if reinforcement_count < self.reinforcement_threshold:
                    cursor.close()
                    return

                # Use last N signals
                recent = signal_history[-self.signal_history_size:]
                if len(recent) < self.reinforcement_threshold:
                    cursor.close()
                    return

                # Gate 2: direction consistency
                positive_count = sum(1 for s in recent if s > 0)
                negative_count = sum(1 for s in recent if s < 0)
                dominant_count = max(positive_count, negative_count)
                consistency = dominant_count / len(recent) if recent else 0
                if consistency < self.direction_consistency_min:
                    cursor.close()
                    return

                # Gate 3: low variance
                try:
                    variance = statistics.variance(recent) if len(recent) > 1 else 0
                except Exception:
                    variance = 0
                if variance > self.variance_max:
                    cursor.close()
                    return

                # Gate 4: daily drift budget
                now = datetime.now()
                if drift_window_start and (now - drift_window_start) > timedelta(hours=24):
                    baseline_drift_today = 0.0
                    drift_window_start = now

                if baseline_drift_today >= self.max_drift_per_day:
                    cursor.close()
                    return

                # All gates passed — drift baseline
                avg_direction = statistics.mean(recent)
                drift_sign = 1.0 if avg_direction > 0 else -1.0
                old_baseline = baseline
                new_baseline = baseline + drift_sign * self.drift_rate
                new_baseline = max(min_cap, min(max_cap, new_baseline))
                new_drift_today = baseline_drift_today + abs(new_baseline - old_baseline)

                cursor.execute("""
                    UPDATE identity_vectors
                    SET baseline_weight = %s,
                        baseline_drift_today = %s,
                        drift_window_start = %s,
                        reinforcement_count = 0,
                        signal_history = '[]',
                        last_updated_at = NOW()
                    WHERE vector_name = %s
                """, (new_baseline, new_drift_today, drift_window_start or now, vector_name))

                self._log_event(cursor, vector_name, old_baseline, new_baseline,
                                'drift', drift_sign * self.drift_rate)

                logger.info(f"[IDENTITY] Baseline drift: {vector_name} {old_baseline:.3f} → {new_baseline:.3f}")
                cursor.close()

        except Exception as e:
            logger.error(f"[IDENTITY] Baseline drift evaluation failed for {vector_name}: {e}")

    def check_coherence(self) -> bool:
        """
        Two-level coherence check:
        Level 1 — Cap enforcement (clamp all activations)
        Level 2 — Relational constraints (prevent socially unstable combos)

        CRITICAL: Coherence adjustments do NOT append to signal_history,
        increment reinforcement_count, or trigger baseline drift.
        """
        coherent = True
        try:
            with self.db.connection() as conn:
                cursor = conn.cursor()

                # Load all vectors
                cursor.execute("""
                    SELECT vector_name, current_activation, min_cap, max_cap
                    FROM identity_vectors
                """)
                vectors = {}
                for row in cursor.fetchall():
                    vectors[row[0]] = {
                        'current_activation': row[1],
                        'min_cap': row[2],
                        'max_cap': row[3],
                    }

                # Level 1: Cap enforcement
                for name, v in vectors.items():
                    clamped = max(v['min_cap'], min(v['max_cap'], v['current_activation']))
                    if abs(clamped - v['current_activation']) > 0.001:
                        old = v['current_activation']
                        cursor.execute("""
                            UPDATE identity_vectors
                            SET current_activation = %s, last_updated_at = NOW()
                            WHERE vector_name = %s
                        """, (clamped, name))
                        self._log_event(cursor, name, old, clamped, 'coherence', None)
                        v['current_activation'] = clamped
                        coherent = False
                        logger.warning(f"[IDENTITY] Cap enforcement: {name} {old:.3f} → {clamped:.3f}")

                # Level 2: Relational constraints
                for constraint in self.relational_constraints:
                    a_name = constraint['a']
                    b_name = constraint['b']
                    if a_name not in vectors or b_name not in vectors:
                        continue

                    a_val = vectors[a_name]['current_activation']
                    b_val = vectors[b_name]['current_activation']

                    if constraint['type'] == 'floor_ratio':
                        a_threshold = constraint.get('a_threshold', 0.75)
                        b_floor = constraint.get('b_floor', 0.35)
                        nudge = constraint.get('nudge', 0.05)

                        if a_val > a_threshold and b_val < b_floor:
                            old_b = b_val
                            new_b = min(b_val + nudge, vectors[b_name]['max_cap'])
                            cursor.execute("""
                                UPDATE identity_vectors
                                SET current_activation = %s, last_updated_at = NOW()
                                WHERE vector_name = %s
                            """, (new_b, b_name))
                            self._log_event(cursor, b_name, old_b, new_b, 'coherence', None)
                            vectors[b_name]['current_activation'] = new_b
                            coherent = False
                            logger.warning(
                                f"[IDENTITY] Floor ratio: {a_name}={a_val:.2f} > {a_threshold}, "
                                f"nudging {b_name} {old_b:.3f} → {new_b:.3f}"
                            )

                    elif constraint['type'] == 'ceiling_pair':
                        threshold = constraint.get('threshold', 0.75)
                        target = constraint.get('target', 0.7)

                        if a_val > threshold and b_val > threshold:
                            for vec_name in [a_name, b_name]:
                                old_val = vectors[vec_name]['current_activation']
                                new_val = old_val + (target - old_val) * 0.3
                                cursor.execute("""
                                    UPDATE identity_vectors
                                    SET current_activation = %s, last_updated_at = NOW()
                                    WHERE vector_name = %s
                                """, (new_val, vec_name))
                                self._log_event(cursor, vec_name, old_val, new_val, 'coherence', None)
                                vectors[vec_name]['current_activation'] = new_val
                            coherent = False
                            logger.warning(
                                f"[IDENTITY] Ceiling pair: {a_name}={a_val:.2f}, {b_name}={b_val:.2f} "
                                f"both > {threshold}, reducing toward {target}"
                            )

                cursor.close()

        except Exception as e:
            logger.error(f"[IDENTITY] Coherence check failed: {e}")
        return coherent

    def _log_event(self, cursor, vector_name: str, old_activation: float,
                   new_activation: float, signal_source: str,
                   signal_value: float = None, topic: str = None):
        """Append to identity_events for observability."""
        try:
            cursor.execute("""
                INSERT INTO identity_events
                    (vector_name, old_activation, new_activation, signal_source, signal_value, topic)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (vector_name, old_activation, new_activation, signal_source, signal_value, topic))
        except Exception as e:
            logger.debug(f"[IDENTITY] Event logging failed: {e}")
