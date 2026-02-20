"""
Homeostatic Stability Regulator

Control-theoretic approach to topic classification parameter tuning.

Principles:
- One metric → one parameter (no coupling)
- Primary + fallback control for boundary conditions
- Stability bands with persistence (3-day deviation required)
- Single adjustment per day (prevent compounding)
- 48h cooldown per parameter (delayed feedback compensation)
- Minimal corrections (0.01 max)
- Disk persistence (configs/generated/) for critical state

This is a closed-loop dynamical system, not ML weight tuning.
"""

import logging
import time
import json
import os
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np

from services.database_service import DatabaseService, get_merged_db_config


logger = logging.getLogger(__name__)


def _json_default(obj):
    """JSON serializer for types not supported by default (Decimal from PostgreSQL)."""
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# Disk storage paths (persistent, version-controllable)
GENERATED_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "configs",
    "generated"
)
CONFIG_FILE = os.path.join(GENERATED_CONFIG_DIR, "topic_stability_config.json")
METRICS_HISTORY_FILE = os.path.join(GENERATED_CONFIG_DIR, "topic_metrics_history.json")


class TopicStabilityRegulator:
    """
    Homeostatic regulator for topic classification parameters.

    Control Architecture:
    - Monitors behavioral metrics
    - Detects sustained deviations from stability bands
    - Makes minimal single-parameter adjustments
    - Enforces cooldowns to prevent oscillation
    - Persists to disk (survives Redis flush)
    """

    # Default parameters
    DEFAULT_CONFIG = {
        "switch_threshold": 0.65,
        "decay_constant": 300,
        "w_semantic": 0.6,
        "w_freshness": 0.3,
        "w_salience": 0.1,

        # Cooldown tracking (ISO timestamps)
        "last_adjusted": {
            "switch_threshold": None,
            "decay_constant": None,
            "w_semantic": None,
            "w_freshness": None
        }
    }

    # Control constraints
    MAX_DAILY_ADJUSTMENT = 0.01      # Micro-adjustments only
    COOLDOWN_HOURS = 48              # Delayed feedback compensation
    PERSISTENCE_DAYS = 3             # Sustained deviation required
    MIN_DAILY_MESSAGES = 10          # Minimum activity threshold

    # Stability bands (healthy operating ranges)
    STABILITY_BANDS = {
        "switch_frequency": (0.05, 0.25),    # 1 switch per 4-20 messages
        "avg_topic_lifespan": (180, 2400),   # 3-40 minutes
        "fragmentation_rate": (0.0, 0.15)    # Max 15% micro-topics
    }

    # Hard parameter bounds
    PARAMETER_BOUNDS = {
        "switch_threshold": (0.3, 0.7),
        "decay_constant": (60, 1800),
        "w_semantic": (0.3, 0.8),
        "w_freshness": (0.1, 0.6),
        "w_salience": (0.05, 0.2)
    }

    # Metric → Parameter mapping (primary + fallback for boundary conditions)
    # Format: (primary_parameter, fallback_parameter)
    CONTROL_MAP = {
        "switch_frequency": ("switch_threshold", None),           # Primary only
        "avg_topic_lifespan": ("decay_constant", None),          # Primary only
        "fragmentation_rate": ("switch_threshold", "w_semantic") # Primary + fallback
    }

    # Rationale for fragmentation control:
    # - Primary: switch_threshold (fragmentation mainly caused by threshold too high)
    # - Fallback: w_semantic (if threshold at bound, adjust semantic weight instead)

    def __init__(self, db=None):
        self.db = db or DatabaseService(get_merged_db_config())

        # Ensure generated config directory exists
        os.makedirs(GENERATED_CONFIG_DIR, exist_ok=True)

    def run_regulation_cycle(self):
        """
        Run one regulation cycle (called every 24 hours).

        Control loop:
        1. Measure current behavioral metrics
        2. Compare to stability bands
        3. Check for sustained (3-day) deviations
        4. Select single worst deviation
        5. Adjust corresponding parameter (if not on cooldown)
        6. Log and save
        """
        logger.info("[STABILITY REGULATOR] Starting 24h regulation cycle...")
        start_time = time.time()

        # Load current config
        config = self._load_config()

        # Measure current metrics
        current_metrics = self._measure_current_metrics()

        if current_metrics['total_messages'] < self.MIN_DAILY_MESSAGES:
            logger.info(
                f"[STABILITY REGULATOR] Insufficient activity "
                f"({current_metrics['total_messages']} messages). Skipping."
            )
            self._record_metrics(current_metrics)
            return

        # Record metrics for persistence tracking
        self._record_metrics(current_metrics)

        # Check stability (requires 3-day persistence)
        deviations = self._detect_sustained_deviations()

        if not deviations:
            logger.info("[STABILITY REGULATOR] System stable. No adjustments needed.")
            return

        # Select worst deviation
        worst_metric, deviation_severity = max(deviations.items(), key=lambda x: x[1])

        # Map to parameter (primary + optional fallback)
        primary_param, fallback_param = self.CONTROL_MAP[worst_metric]

        # Select target parameter (primary or fallback)
        target_parameter = self._select_parameter(
            config,
            worst_metric,
            primary_param,
            fallback_param
        )

        if not target_parameter:
            logger.info(
                f"[STABILITY REGULATOR] No available parameter for {worst_metric}. "
                f"Primary and fallback both unavailable."
            )
            return

        # Check cooldown
        if self._is_on_cooldown(config, target_parameter):
            logger.info(
                f"[STABILITY REGULATOR] {target_parameter} on cooldown. "
                f"Deferring adjustment."
            )
            return

        # Calculate correction
        adjustment = self._calculate_correction(
            worst_metric,
            current_metrics[worst_metric],
            target_parameter
        )

        # Apply adjustment
        new_config = self._apply_adjustment(config, target_parameter, adjustment)

        # Save
        self._save_config(new_config, current_metrics, target_parameter, adjustment)

        elapsed = time.time() - start_time
        logger.info(
            f"[STABILITY REGULATOR] Cycle complete in {elapsed:.2f}s. "
            f"Adjusted {target_parameter} by {adjustment:+.4f}"
        )

    def _load_config(self) -> Dict:
        """Load current configuration from disk."""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                logger.info(f"[STABILITY REGULATOR] Loaded config from {CONFIG_FILE}")
                return config
            except Exception as e:
                logger.error(f"[STABILITY REGULATOR] Failed to load config: {e}")

        logger.info("[STABILITY REGULATOR] Initializing default config")
        return self.DEFAULT_CONFIG.copy()

    def _save_config(self, config: Dict, metrics: Dict, parameter: str, adjustment: float):
        """
        Save updated config to disk with audit trail.

        Args:
            config: New configuration
            metrics: Current metrics
            parameter: Adjusted parameter name
            adjustment: Applied adjustment value
        """
        # Update last_adjusted timestamp
        config['last_adjusted'][parameter] = datetime.now().isoformat()

        # Save to disk (atomic write via temp file)
        temp_file = CONFIG_FILE + ".tmp"
        try:
            with open(temp_file, 'w') as f:
                json.dump(config, f, indent=2, default=_json_default)
            os.replace(temp_file, CONFIG_FILE)

            logger.info(
                f"[STABILITY REGULATOR] Saved config to {CONFIG_FILE}. "
                f"Updated {parameter}: {config[parameter] - adjustment:.4f} → {config[parameter]:.4f}"
            )
        except Exception as e:
            logger.error(f"[STABILITY REGULATOR] Failed to save config: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def _record_metrics(self, metrics: Dict):
        """
        Record metrics to history on disk (for persistence tracking).

        Keeps last 7 days of daily metrics.
        """
        # Load existing history
        history = []
        if os.path.exists(METRICS_HISTORY_FILE):
            try:
                with open(METRICS_HISTORY_FILE, 'r') as f:
                    history = json.load(f)
            except Exception as e:
                logger.error(f"[STABILITY REGULATOR] Failed to load metrics history: {e}")

        # Add today's metrics
        entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'metrics': metrics
        }
        history.append(entry)

        # Keep last 7 days
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        history = [
            h for h in history
            if datetime.fromisoformat(h['timestamp']) > cutoff
        ]

        # Save to disk (atomic write)
        temp_file = METRICS_HISTORY_FILE + ".tmp"
        try:
            with open(temp_file, 'w') as f:
                json.dump(history, f, indent=2, default=_json_default)
            os.replace(temp_file, METRICS_HISTORY_FILE)
        except Exception as e:
            logger.error(f"[STABILITY REGULATOR] Failed to save metrics history: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def _measure_current_metrics(self) -> Dict:
        """
        Measure behavioral metrics from last 24 hours.

        Returns:
            {
                'total_messages': int,
                'switch_frequency': float,
                'avg_topic_lifespan': float (seconds),
                'fragmentation_rate': float
            }
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)

        query = """
            SELECT
                name,
                created_at,
                last_updated,
                message_count,
                EXTRACT(EPOCH FROM (last_updated - created_at)) as lifespan_seconds
            FROM topics
            WHERE last_updated >= %s
        """

        topics = []
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (cutoff_time,))
            rows = cursor.fetchall()

            for row in rows:
                topics.append({
                    'name': row[0],
                    'created_at': row[1],
                    'last_updated': row[2],
                    'message_count': row[3],
                    'lifespan_seconds': row[4] or 0
                })

        if not topics:
            return {
                'total_messages': 0,
                'switch_frequency': 0,
                'avg_topic_lifespan': 0,
                'fragmentation_rate': 0
            }

        total_messages = sum(t['message_count'] for t in topics)
        new_topics = [t for t in topics if t['created_at'] >= cutoff_time]
        switch_frequency = len(new_topics) / total_messages if total_messages > 0 else 0

        active_topics = [t for t in topics if t['message_count'] > 1]
        avg_lifespan = (
            np.mean([t['lifespan_seconds'] for t in active_topics])
            if active_topics else 0
        )

        fragmented = [t for t in topics if t['message_count'] < 3]
        fragmentation_rate = len(fragmented) / len(topics) if topics else 0

        return {
            'total_messages': total_messages,
            'switch_frequency': switch_frequency,
            'avg_topic_lifespan': avg_lifespan,
            'fragmentation_rate': fragmentation_rate
        }

    def _detect_sustained_deviations(self) -> Dict[str, float]:
        """
        Detect metrics that have been outside stability bands
        for PERSISTENCE_DAYS consecutive days.

        Returns:
            {metric_name: deviation_severity}
        """
        if not os.path.exists(METRICS_HISTORY_FILE):
            return {}

        try:
            with open(METRICS_HISTORY_FILE, 'r') as f:
                history = json.load(f)
        except Exception as e:
            logger.error(f"[STABILITY REGULATOR] Failed to load metrics history: {e}")
            return {}

        # Get last N days
        recent = history[-self.PERSISTENCE_DAYS:]

        if len(recent) < self.PERSISTENCE_DAYS:
            logger.info(
                f"[STABILITY REGULATOR] Only {len(recent)} days of history. "
                f"Need {self.PERSISTENCE_DAYS} for persistence check."
            )
            return {}

        deviations = {}

        for metric_name, (lower, upper) in self.STABILITY_BANDS.items():
            # Check if ALL recent days are outside band
            all_outside = all(
                not (lower <= entry['metrics'].get(metric_name, 0) <= upper)
                for entry in recent
            )

            if all_outside:
                # Calculate average deviation severity
                current_value = recent[-1]['metrics'].get(metric_name, 0)

                if current_value < lower:
                    severity = (lower - current_value) / lower
                else:
                    severity = (current_value - upper) / upper

                deviations[metric_name] = severity

                logger.info(
                    f"[STABILITY REGULATOR] Sustained deviation detected: "
                    f"{metric_name}={current_value:.3f} "
                    f"(band: [{lower:.3f}, {upper:.3f}], severity: {severity:.3f})"
                )

        return deviations

    def _select_parameter(
        self,
        config: Dict,
        metric_name: str,
        primary_param: str,
        fallback_param: Optional[str]
    ) -> Optional[str]:
        """
        Select target parameter (primary or fallback).

        Primary + fallback logic:
        1. Try primary parameter
        2. If primary at bound → use fallback (if available)
        3. If both at bound → return None

        Args:
            config: Current configuration
            metric_name: Metric being corrected
            primary_param: Primary parameter
            fallback_param: Fallback parameter (or None)

        Returns:
            Selected parameter name or None
        """
        # Check if primary is at bound
        primary_value = config[primary_param]
        lower, upper = self.PARAMETER_BOUNDS[primary_param]

        primary_at_bound = (primary_value <= lower or primary_value >= upper)

        if not primary_at_bound:
            # Primary parameter available
            logger.info(
                f"[STABILITY REGULATOR] Using primary parameter {primary_param} "
                f"for {metric_name}"
            )
            return primary_param

        if fallback_param:
            # Primary at bound, use fallback
            logger.info(
                f"[STABILITY REGULATOR] {primary_param} at bound ({primary_value:.3f}). "
                f"Using fallback {fallback_param} for {metric_name}"
            )
            return fallback_param

        # Both unavailable
        logger.warning(
            f"[STABILITY REGULATOR] {primary_param} at bound ({primary_value:.3f}) "
            f"and no fallback available for {metric_name}"
        )
        return None

    def _is_on_cooldown(self, config: Dict, parameter: str) -> bool:
        """
        Check if parameter is on cooldown (48h since last adjustment).

        Args:
            config: Current configuration
            parameter: Parameter name

        Returns:
            True if on cooldown, False otherwise
        """
        last_adjusted_str = config['last_adjusted'].get(parameter)

        if not last_adjusted_str:
            return False  # Never adjusted, not on cooldown

        last_adjusted = datetime.fromisoformat(last_adjusted_str)
        elapsed = datetime.now(timezone.utc) - last_adjusted
        on_cooldown = elapsed.total_seconds() < (self.COOLDOWN_HOURS * 3600)

        if on_cooldown:
            remaining_hours = (self.COOLDOWN_HOURS * 3600 - elapsed.total_seconds()) / 3600
            logger.info(
                f"[STABILITY REGULATOR] {parameter} on cooldown "
                f"({remaining_hours:.1f}h remaining)"
            )

        return on_cooldown

    def _calculate_correction(
        self,
        metric_name: str,
        current_value: float,
        parameter: str
    ) -> float:
        """
        Calculate minimal correction for parameter.

        Direction logic:
        - switch_frequency too high → INCREASE threshold (harder to create topic)
        - switch_frequency too low → DECREASE threshold (easier to create topic)
        - lifespan too short → INCREASE decay_constant (longer memory)
        - lifespan too long → DECREASE decay_constant (shorter memory)
        - fragmentation too high → INCREASE w_semantic (prioritize similarity)

        Returns:
            Adjustment value (±0.01 max)
        """
        lower, upper = self.STABILITY_BANDS[metric_name]

        if current_value > upper:
            # Above band
            if metric_name == "switch_frequency":
                # Too many switches → INCREASE threshold
                adjustment = +self.MAX_DAILY_ADJUSTMENT
            elif metric_name == "avg_topic_lifespan":
                # Topics too long → DECREASE decay
                adjustment = -10  # Decay constant in seconds
            elif metric_name == "fragmentation_rate":
                # Too fragmented → INCREASE semantic weight
                adjustment = +self.MAX_DAILY_ADJUSTMENT
            else:
                adjustment = 0

        elif current_value < lower:
            # Below band
            if metric_name == "switch_frequency":
                # Too few switches → DECREASE threshold
                adjustment = -self.MAX_DAILY_ADJUSTMENT
            elif metric_name == "avg_topic_lifespan":
                # Topics too short → INCREASE decay
                adjustment = +10
            elif metric_name == "fragmentation_rate":
                # Not a problem (low fragmentation is good)
                adjustment = 0
            else:
                adjustment = 0

        else:
            adjustment = 0

        return adjustment

    def _apply_adjustment(
        self,
        config: Dict,
        parameter: str,
        adjustment: float
    ) -> Dict:
        """
        Apply adjustment with hard bounds.

        Args:
            config: Current configuration
            parameter: Parameter to adjust
            adjustment: Adjustment value

        Returns:
            Updated configuration
        """
        new_config = config.copy()

        old_value = new_config[parameter]
        new_value = old_value + adjustment

        # Apply hard bounds
        lower, upper = self.PARAMETER_BOUNDS[parameter]
        new_value = np.clip(new_value, lower, upper)

        new_config[parameter] = float(new_value)

        logger.info(
            f"[STABILITY REGULATOR] {parameter}: "
            f"{old_value:.4f} → {new_value:.4f} (Δ={adjustment:+.4f})"
        )

        return new_config

    def get_current_parameters(self) -> Dict:
        """
        Get current regulated parameters.

        Returns:
            {switch_threshold, decay_constant, w_semantic, w_freshness, w_salience}
        """
        config = self._load_config()
        return {
            'switch_threshold': config['switch_threshold'],
            'decay_constant': config['decay_constant'],
            'w_semantic': config['w_semantic'],
            'w_freshness': config['w_freshness'],
            'w_salience': config['w_salience']
        }


def topic_stability_regulator_worker(shared_state=None):
    """
    Worker entry point for scheduled regulation.

    Runs every 24 hours.
    """
    logger.info("[STABILITY REGULATOR WORKER] Starting regulation cycle...")

    try:
        regulator = TopicStabilityRegulator()
        regulator.run_regulation_cycle()

    except Exception as e:
        logger.error(
            f"[STABILITY REGULATOR WORKER] Regulation failed: {e}",
            exc_info=True
        )

    # Sleep 24 hours
    time.sleep(86400)
