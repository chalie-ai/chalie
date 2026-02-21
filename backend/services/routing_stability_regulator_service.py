"""
Routing Stability Regulator — Single authority for mode router weight mutation.

Follows TopicStabilityRegulatorService pattern:
- Runs on 24h cycle (registered as service in consumer.py)
- Reads pressure signals from routing_decisions table
- Computes: tie-breaker rate, mode entropy, misroute rate, reflection disagreement
- Selects worst pressure, maps to single parameter adjustment
- Max ±0.02 per day, 48h cooldown per parameter, hard bounds on all weights
- Persists to configs/generated/mode_router_config.json
- Closed-loop: tracks adjustment effects and reverts if no improvement
"""

import os
import json
import time
import math
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal

import numpy as np

from services.database_service import DatabaseService, get_merged_db_config
from services.routing_decision_service import RoutingDecisionService

logger = logging.getLogger(__name__)

LOG_PREFIX = "[ROUTING REGULATOR]"


def _json_default(obj):
    """JSON serializer for non-standard types."""
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# Disk storage paths
GENERATED_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "configs", "generated"
)
CONFIG_FILE = os.path.join(GENERATED_CONFIG_DIR, "mode_router_config.json")
METRICS_HISTORY_FILE = os.path.join(GENERATED_CONFIG_DIR, "routing_metrics_history.json")
ADJUSTMENT_LOG_FILE = os.path.join(GENERATED_CONFIG_DIR, "routing_adjustment_log.json")


class RoutingStabilityRegulator:
    """
    Single authority for mode router weight mutation.

    Only this service can modify router weights. Monitors log pressure signals,
    and this service consumes them to make bounded adjustments.
    """

    # Control constraints
    MAX_DAILY_ADJUSTMENT = 0.02
    COOLDOWN_HOURS = 48
    MIN_DAILY_DECISIONS = 10

    # Pressure thresholds (when to act)
    PRESSURE_THRESHOLDS = {
        'tiebreaker_high': 0.20,      # >20% tie-breaker rate after day 7
        'act_underused': 0.10,         # ACT <2% of decisions (kept for legacy data compatibility)
        'respond_overused': 0.10,      # RESPOND >85%
        'misroute_rate': 0.10,         # >10% misroute feedback
        'reflection_disagreement': 0.20,  # >20% reflection disagreement
        'triage_miscalibration': 0.25,  # >25% incorrect triage decisions
    }

    # Hard bounds on weight parameters
    PARAMETER_BOUNDS = {
        'RESPOND.base': (0.30, 0.70),
        'CLARIFY.base': (0.10, 0.50),
        'ACT.base': (0.05, 0.40),
        'ACKNOWLEDGE.base': (0.00, 0.30),
        'respond.warmth_boost': (0.05, 0.40),
        'respond.cold_penalty': (0.05, 0.30),
        'clarify.cold_boost': (0.10, 0.40),
        'clarify.warm_penalty': (0.05, 0.35),
        'act.question_moderate_context': (0.05, 0.35),
        'act.implicit_reference': (0.05, 0.30),
        'acknowledge.greeting': (0.30, 0.80),
        'acknowledge.question_penalty': (0.10, 0.50),
        'tiebreaker_base_margin': (0.05, 0.30),
    }

    # Pressure → Parameter control map
    CONTROL_MAP = {
        'act_underused': ('ACT.base', +1),
        'respond_overused': ('RESPOND.base', -1),
        'tiebreaker_high': ('tiebreaker_base_margin', -1),
        'misroute_rate': None,  # depends on dominant misroute type
        'reflection_disagreement': None,  # depends on dimensional analysis
    }

    def __init__(self, db=None):
        self.db = db or DatabaseService(get_merged_db_config())
        self.decision_service = RoutingDecisionService(self.db)
        os.makedirs(GENERATED_CONFIG_DIR, exist_ok=True)

    def run_regulation_cycle(self):
        """
        Run one regulation cycle (called every 24 hours).

        1. Measure pressure signals from routing_decisions
        2. Check pending adjustment effects (closed-loop)
        3. Select worst pressure
        4. Apply bounded correction
        """
        logger.info(f"{LOG_PREFIX} Starting 24h regulation cycle...")
        start_time = time.time()

        # Load current config
        config = self._load_config()

        # Measure current pressures
        pressures = self._measure_pressures()
        total_decisions = pressures.pop('_total_decisions', 0)

        if total_decisions < self.MIN_DAILY_DECISIONS:
            logger.info(
                f"{LOG_PREFIX} Insufficient activity ({total_decisions} decisions). Skipping."
            )
            self._record_metrics(pressures, total_decisions)
            return

        # Record metrics
        self._record_metrics(pressures, total_decisions)

        # Check pending adjustment effects (closed-loop)
        self._evaluate_pending_adjustments(config)

        # Filter to actionable pressures (above threshold)
        actionable = {
            k: v for k, v in pressures.items()
            if v > self.PRESSURE_THRESHOLDS.get(k, 0.10)
        }

        if not actionable:
            logger.info(f"{LOG_PREFIX} All pressures within bounds. No adjustment needed.")
            return

        # Select worst pressure
        worst_pressure, severity = max(actionable.items(), key=lambda x: x[1])
        logger.info(f"{LOG_PREFIX} Worst pressure: {worst_pressure} = {severity:.3f}")

        # Map to parameter
        target_param, direction = self._map_pressure_to_param(worst_pressure, pressures)
        if not target_param:
            logger.info(f"{LOG_PREFIX} No parameter mapping for {worst_pressure}. Skipping.")
            return

        # Check cooldown
        if self._is_on_cooldown(config, target_param):
            logger.info(f"{LOG_PREFIX} {target_param} on cooldown. Deferring.")
            return

        # Calculate and apply adjustment
        adjustment = direction * min(severity * 0.05, self.MAX_DAILY_ADJUSTMENT)
        new_config = self._apply_adjustment(config, target_param, adjustment)

        # Log adjustment for closed-loop tracking
        self._log_adjustment(target_param, adjustment, pressures)

        # Save
        self._save_config(new_config, target_param, adjustment)

        elapsed = time.time() - start_time
        logger.info(
            f"{LOG_PREFIX} Cycle complete in {elapsed:.2f}s. "
            f"Adjusted {target_param} by {adjustment:+.4f}"
        )

    def _measure_pressures(self) -> Dict[str, float]:
        """Measure all pressure signals from routing_decisions."""
        decisions = self.decision_service.get_recent_decisions(hours=24)
        total = len(decisions)

        if total == 0:
            return {'_total_decisions': 0}

        # Tie-breaker rate
        tb_count = sum(1 for d in decisions if d.get('tiebreaker_used'))
        tb_rate = tb_count / total

        # Mode distribution
        mode_counts = {}
        for d in decisions:
            mode = d['selected_mode']
            mode_counts[mode] = mode_counts.get(mode, 0) + 1

        respond_pct = mode_counts.get('RESPOND', 0) / total
        act_pct = mode_counts.get('ACT', 0) / total

        # Misroute rate (from feedback)
        decisions_with_feedback = [d for d in decisions if d.get('feedback')]
        misroute_count = sum(
            1 for d in decisions_with_feedback
            if d['feedback'].get('misroute', False)
        )
        misroute_rate = (
            misroute_count / len(decisions_with_feedback)
            if decisions_with_feedback else 0.0
        )

        # Reflection disagreement rate
        decisions_with_reflection = [d for d in decisions if d.get('reflection')]
        disagree_count = sum(
            1 for d in decisions_with_reflection
            if d['reflection'].get('counted', False)
            and not d['reflection'].get('agree_with_decision', True)
        )
        disagree_rate = (
            disagree_count / len(decisions_with_reflection)
            if decisions_with_reflection else 0.0
        )

        # Mode entropy
        if mode_counts:
            probs = [c / total for c in mode_counts.values()]
            entropy = -sum(p * math.log2(p) for p in probs if p > 0)
            max_entropy = math.log2(5)  # 5 modes
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        else:
            normalized_entropy = 0

        # Measure triage miscalibration pressure
        triage_miscalibration = 0.0
        try:
            from services.triage_calibration_service import TriageCalibrationService
            cal_stats = TriageCalibrationService().get_calibration_stats()
            false_rate = cal_stats.get('false_positive_rate', 0) + cal_stats.get('false_negative_rate', 0)
            if false_rate > self.PRESSURE_THRESHOLDS.get('triage_miscalibration', 0.25):
                triage_miscalibration = false_rate
        except Exception:
            pass

        return {
            '_total_decisions': total,
            'tiebreaker_high': tb_rate,
            'respond_overused': max(0, respond_pct - 0.75),  # pressure only above 75%
            'act_underused': 0,  # ACT routing now controlled by triage; kept for data compatibility
            'misroute_rate': misroute_rate,
            'reflection_disagreement': disagree_rate,
            'triage_miscalibration': triage_miscalibration,
        }

    def _map_pressure_to_param(
        self,
        pressure_name: str,
        all_pressures: Dict[str, float],
    ) -> Tuple[Optional[str], int]:
        """
        Map a pressure signal to a parameter and direction.

        Returns (parameter_name, direction) or (None, 0).
        """
        mapping = self.CONTROL_MAP.get(pressure_name)

        if mapping is None:
            # Dynamic mapping for misroute/reflection
            if pressure_name == 'misroute_rate':
                return self._map_misroute_pressure()
            elif pressure_name == 'reflection_disagreement':
                return self._map_reflection_pressure()
            return None, 0

        return mapping

    def _map_misroute_pressure(self) -> Tuple[Optional[str], int]:
        """Map misroute feedback to specific parameter adjustment."""
        # Check dominant misroute type from recent feedback
        decisions = self.decision_service.get_recent_decisions(hours=24)
        misroute_types = {}
        for d in decisions:
            fb = d.get('feedback', {})
            if fb and fb.get('misroute'):
                mtype = fb.get('type', 'unknown')
                misroute_types[mtype] = misroute_types.get(mtype, 0) + 1

        if not misroute_types:
            return None, 0

        dominant = max(misroute_types, key=misroute_types.get)

        TYPE_TO_PARAM = {
            'missed_clarify': ('clarify.cold_boost', +1),
            'missed_act': ('act.question_moderate_context', +1),
            'under_engagement': ('acknowledge.question_penalty', +1),
        }

        return TYPE_TO_PARAM.get(dominant, (None, 0))

    def _map_reflection_pressure(self) -> Tuple[Optional[str], int]:
        """Map reflection disagreement to parameter based on dimensional analysis."""
        # Get recent reflections with dimension data
        decisions = self.decision_service.get_recent_decisions(hours=168)  # 7 days
        dimension_counts = {}

        for d in decisions:
            refl = d.get('reflection', {})
            if not refl or refl.get('agree_with_decision', True):
                continue
            for dim in refl.get('uncertainty_dimensions', []):
                dim_name = dim.get('dimension', '')
                dimension_counts[dim_name] = dimension_counts.get(dim_name, 0) + 1

        if not dimension_counts:
            return None, 0

        # Check statistical consistency (>30% of disagreements)
        total_disagreements = sum(dimension_counts.values())
        dominant_dim = max(dimension_counts, key=dimension_counts.get)
        if dimension_counts[dominant_dim] / total_disagreements < 0.30:
            logger.info(
                f"{LOG_PREFIX} No dominant dimension (highest: {dominant_dim} "
                f"at {dimension_counts[dominant_dim]}/{total_disagreements})"
            )
            return None, 0

        DIMENSION_TO_PARAM = {
            'memory_availability': ('act.question_moderate_context', +1),
            'intent_clarity': ('clarify.cold_boost', +1),
            'tone_ambiguity': ('tiebreaker_base_margin', +1),
            'context_sufficiency': ('respond.cold_penalty', +1),
            'social_vs_substantive': ('acknowledge.question_penalty', +1),
        }

        return DIMENSION_TO_PARAM.get(dominant_dim, (None, 0))

    def _is_on_cooldown(self, config: Dict, parameter: str) -> bool:
        """Check if parameter has been adjusted within cooldown period."""
        last_adjusted = config.get('last_adjusted', {}).get(parameter)
        if not last_adjusted:
            return False

        last_time = datetime.fromisoformat(last_adjusted)
        elapsed = datetime.now(timezone.utc) - last_time
        on_cooldown = elapsed.total_seconds() < (self.COOLDOWN_HOURS * 3600)

        if on_cooldown:
            remaining = (self.COOLDOWN_HOURS * 3600 - elapsed.total_seconds()) / 3600
            logger.info(f"{LOG_PREFIX} {parameter} on cooldown ({remaining:.1f}h remaining)")

        return on_cooldown

    def _apply_adjustment(
        self,
        config: Dict,
        parameter: str,
        adjustment: float,
    ) -> Dict:
        """Apply bounded adjustment to a parameter."""
        new_config = json.loads(json.dumps(config, default=_json_default))

        # Determine where the parameter lives
        if parameter.endswith('.base'):
            # Base score: e.g. "ACT.base" → config["base_scores"]["ACT"]
            mode = parameter.split('.')[0]
            old_value = new_config.get('base_scores', {}).get(mode, 0.0)
            new_value = old_value + adjustment
        elif parameter == 'tiebreaker_base_margin':
            old_value = new_config.get('tiebreaker_base_margin', 0.20)
            new_value = old_value + adjustment
        else:
            # Weight parameter: e.g. "act.question_moderate_context"
            old_value = new_config.get('weights', {}).get(parameter, 0.0)
            new_value = old_value + adjustment

        # Apply hard bounds
        bounds = self.PARAMETER_BOUNDS.get(parameter)
        if bounds:
            new_value = max(bounds[0], min(bounds[1], new_value))

        # Store back
        if parameter.endswith('.base'):
            mode = parameter.split('.')[0]
            if 'base_scores' not in new_config:
                new_config['base_scores'] = {}
            new_config['base_scores'][mode] = new_value
        elif parameter == 'tiebreaker_base_margin':
            new_config['tiebreaker_base_margin'] = new_value
        else:
            if 'weights' not in new_config:
                new_config['weights'] = {}
            new_config['weights'][parameter] = new_value

        # Record cooldown
        if 'last_adjusted' not in new_config:
            new_config['last_adjusted'] = {}
        new_config['last_adjusted'][parameter] = datetime.now(timezone.utc).isoformat()

        logger.info(
            f"{LOG_PREFIX} {parameter}: {old_value:.4f} → {new_value:.4f} "
            f"(Δ={adjustment:+.4f})"
        )

        return new_config

    def _evaluate_pending_adjustments(self, config: Dict):
        """
        Closed-loop control: check if previous adjustments had the desired effect.

        If no improvement after 48h, revert. If degradation, revert and halve future delta.
        """
        if not os.path.exists(ADJUSTMENT_LOG_FILE):
            return

        try:
            with open(ADJUSTMENT_LOG_FILE, 'r') as f:
                adjustments = json.load(f)
        except Exception:
            return

        # Check adjustments older than 48h
        cutoff = datetime.now(timezone.utc) - timedelta(hours=48)
        remaining = []
        current_pressures = self._measure_pressures()

        for adj in adjustments:
            adj_time = datetime.fromisoformat(adj['timestamp'])

            if adj_time > cutoff:
                remaining.append(adj)
                continue

            if adj.get('confirmed'):
                remaining.append(adj)
                continue

            # Evaluate effect
            pre = adj.get('pre_metrics', {})
            param = adj['param']

            # Compare the specific pressure that triggered this adjustment
            trigger = adj.get('trigger_pressure', '')
            pre_value = pre.get(trigger, 0)
            post_value = current_pressures.get(trigger, 0)

            if post_value < pre_value * 0.9:
                # Improvement — confirm
                adj['confirmed'] = True
                adj['effect'] = 'improved'
                logger.info(
                    f"{LOG_PREFIX} Adjustment confirmed: {param} "
                    f"({trigger}: {pre_value:.3f} → {post_value:.3f})"
                )
            elif post_value > pre_value * 1.1:
                # Degradation — revert
                logger.warning(
                    f"{LOG_PREFIX} Adjustment degraded: {param} "
                    f"({trigger}: {pre_value:.3f} → {post_value:.3f}). Reverting."
                )
                config = self._apply_adjustment(config, param, -adj['delta'])
                self._save_config(config, param, -adj['delta'])
                adj['confirmed'] = True
                adj['effect'] = 'reverted'
            else:
                # No measurable effect — revert conservatively
                logger.info(
                    f"{LOG_PREFIX} Adjustment neutral: {param}. Reverting."
                )
                config = self._apply_adjustment(config, param, -adj['delta'])
                self._save_config(config, param, -adj['delta'])
                adj['confirmed'] = True
                adj['effect'] = 'reverted_neutral'

            remaining.append(adj)

        # Save updated log
        try:
            temp = ADJUSTMENT_LOG_FILE + ".tmp"
            with open(temp, 'w') as f:
                json.dump(remaining[-50:], f, indent=2, default=_json_default)
            os.replace(temp, ADJUSTMENT_LOG_FILE)
        except Exception as e:
            logger.error(f"{LOG_PREFIX} Failed to save adjustment log: {e}")

    def _log_adjustment(
        self,
        param: str,
        delta: float,
        pressures: Dict[str, float],
    ):
        """Log an adjustment for closed-loop effect tracking."""
        adjustments = []
        if os.path.exists(ADJUSTMENT_LOG_FILE):
            try:
                with open(ADJUSTMENT_LOG_FILE, 'r') as f:
                    adjustments = json.load(f)
            except Exception:
                pass

        # Find which pressure triggered this
        trigger = max(
            (k for k in pressures if k != '_total_decisions'),
            key=lambda k: pressures.get(k, 0),
            default=''
        )

        adjustments.append({
            'param': param,
            'delta': delta,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'trigger_pressure': trigger,
            'pre_metrics': {k: v for k, v in pressures.items() if k != '_total_decisions'},
            'confirmed': False,
        })

        try:
            temp = ADJUSTMENT_LOG_FILE + ".tmp"
            with open(temp, 'w') as f:
                json.dump(adjustments[-50:], f, indent=2, default=_json_default)
            os.replace(temp, ADJUSTMENT_LOG_FILE)
        except Exception as e:
            logger.error(f"{LOG_PREFIX} Failed to log adjustment: {e}")

    def _load_config(self) -> Dict:
        """Load router config from disk (generated overrides)."""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                logger.info(f"{LOG_PREFIX} Loaded config from {CONFIG_FILE}")
                return config
            except Exception as e:
                logger.error(f"{LOG_PREFIX} Failed to load config: {e}")

        # Fall back to base config
        from services.config_service import ConfigService
        base = ConfigService.get_agent_config("mode-router")
        base['last_adjusted'] = {}
        return base

    def _save_config(self, config: Dict, parameter: str, adjustment: float):
        """Atomic write of config to disk."""
        temp = CONFIG_FILE + ".tmp"
        try:
            with open(temp, 'w') as f:
                json.dump(config, f, indent=2, default=_json_default)
            os.replace(temp, CONFIG_FILE)
            logger.info(f"{LOG_PREFIX} Saved config to {CONFIG_FILE}")
        except Exception as e:
            logger.error(f"{LOG_PREFIX} Failed to save config: {e}")
            if os.path.exists(temp):
                os.remove(temp)

    def _record_metrics(self, pressures: Dict[str, float], total_decisions: int):
        """Record metrics to history for trend analysis."""
        history = []
        if os.path.exists(METRICS_HISTORY_FILE):
            try:
                with open(METRICS_HISTORY_FILE, 'r') as f:
                    history = json.load(f)
            except Exception:
                pass

        history.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'total_decisions': total_decisions,
            'pressures': pressures,
        })

        # Keep last 30 days
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        history = [
            h for h in history
            if datetime.fromisoformat(h['timestamp']) > cutoff
        ]

        try:
            temp = METRICS_HISTORY_FILE + ".tmp"
            with open(temp, 'w') as f:
                json.dump(history, f, indent=2, default=_json_default)
            os.replace(temp, METRICS_HISTORY_FILE)
        except Exception as e:
            logger.error(f"{LOG_PREFIX} Failed to save metrics history: {e}")

    def get_current_config(self) -> Dict:
        """Get current regulated config (for mode router to consume)."""
        return self._load_config()


def routing_stability_regulator_worker(shared_state=None):
    """Worker entry point. Runs every 24 hours."""
    logger.info(f"{LOG_PREFIX} Starting regulation service...")

    while True:
        try:
            regulator = RoutingStabilityRegulator()
            regulator.run_regulation_cycle()
        except Exception as e:
            logger.error(f"{LOG_PREFIX} Regulation failed: {e}", exc_info=True)

        # Sleep 24 hours
        time.sleep(86400)
