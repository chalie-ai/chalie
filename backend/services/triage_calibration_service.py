"""
Triage Calibration Service — Self-improving feedback loop for cognitive triage.

Logs every triage decision, detects outcome signals from subsequent user messages,
and runs a nightly calibration cycle to compute correctness scores.

The calibration data feeds into the routing stability regulator as a pressure signal.

Key design: >=2 negative user signals required to label a decision as incorrect.
A single confusing follow-up is noise, not evidence of routing failure.
"""

import json
import logging
import time
import re
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

LOG_PREFIX = "[TRIAGE CALIBRATION]"

MIN_DAILY_DECISIONS = 10  # Minimum events before calibration adjusts anything
RETENTION_DAYS = 90

# Output file for calibration stats
import os
GENERATED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'generated')
CALIBRATION_STATS_FILE = os.path.join(GENERATED_DIR, 'triage_calibration.json')

# Patterns for user signal detection
_REPHRASE_PATTERNS = [
    re.compile(r'\b(same question|asked (you |this )?before|what I (just |already )?said|I already told you)\b', re.IGNORECASE),
    re.compile(r'\b(what I meant (was|is)|let me rephrase|what I\'m saying is|to clarify)\b', re.IGNORECASE),
]
_CORRECTION_PATTERNS = [
    re.compile(r'\b(no,?\s+(that\'?s|I meant|what I want)|not what I|wrong|incorrect|try again)\b', re.IGNORECASE),
    re.compile(r'\b(I said|that\'?s not|you misunderstood|not quite|you missed)\b', re.IGNORECASE),
]
_EXPLICIT_LOOKUP_PATTERNS = [
    re.compile(r'\b(can you (check|look up|search|find)|can you look|search (for|online)|check online)\b', re.IGNORECASE),
    re.compile(r'\b(look it up|search the web|find online|check (the )?internet)\b', re.IGNORECASE),
]


class TriageCalibrationService:
    """Logs triage decisions, detects outcomes, runs nightly calibration."""

    def __init__(self, db_service=None):
        self._db = db_service
        os.makedirs(GENERATED_DIR, exist_ok=True)

    def _get_db(self):
        if self._db:
            return self._db
        from services.database_service import DatabaseService, get_merged_db_config
        return DatabaseService(get_merged_db_config())

    def log_triage_decision(self, exchange_id: str, topic: str, result) -> None:
        """
        Called immediately after triage. Stores partial event (outcome fields NULL initially).

        Args:
            result: TriageResult dataclass from CognitiveTriageService
        """
        db = self._get_db()
        try:
            db.execute(
                """
                INSERT INTO triage_calibration_events
                    (exchange_id, topic, triage_branch, triage_mode, tool_selected,
                     confidence_internal, confidence_tool_need, reasoning,
                     freshness_risk, decision_entropy, self_eval_override, self_eval_reason)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                """,
                (
                    exchange_id or '',
                    topic or '',
                    result.branch,
                    result.mode,
                    result.tools or [],
                    result.confidence_internal,
                    result.confidence_tool_need,
                    result.reasoning or '',
                    result.freshness_risk,
                    result.decision_entropy,
                    result.self_eval_override,
                    result.self_eval_reason or '',
                )
            )
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} log_triage_decision failed: {e}")
        finally:
            if not self._db:
                db.close_pool()

    def log_outcome(self, exchange_id: str, outcome: dict) -> None:
        """
        Called after response generation / ACT loop completion.
        Updates the calibration event with actual outcome.
        """
        if not exchange_id:
            return
        db = self._get_db()
        try:
            db.execute(
                """
                UPDATE triage_calibration_events
                SET outcome_mode = %s,
                    outcome_tools_used = %s,
                    outcome_tool_success = %s,
                    outcome_latency_ms = %s,
                    tool_abstention = %s
                WHERE exchange_id = %s
                """,
                (
                    outcome.get('mode'),
                    outcome.get('tools_used', []),
                    outcome.get('tool_success'),
                    outcome.get('latency_ms'),
                    outcome.get('tool_abstention', False),
                    exchange_id,
                )
            )
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} log_outcome failed: {e}")
        finally:
            if not self._db:
                db.close_pool()

    def detect_user_signals(self, previous_exchange_id: str, next_text: str, next_intent: dict) -> None:
        """
        Called on the NEXT user message to detect correction/confusion signals.
        Lightweight SQL UPDATE + regex check (~1ms, no LLM).

        IMPORTANT: Only labels as incorrect if >=2 signals present.
        """
        if not previous_exchange_id or not next_text:
            return

        text = next_text.strip()
        signal_rephrase = any(p.search(text) for p in _REPHRASE_PATTERNS)
        signal_correction = any(p.search(text) for p in _CORRECTION_PATTERNS)
        signal_explicit_lookup = any(p.search(text) for p in _EXPLICIT_LOOKUP_PATTERNS)
        signal_abandonment = False  # Not detectable from text alone

        if not any([signal_rephrase, signal_correction, signal_explicit_lookup]):
            return  # No signals, skip DB update

        db = self._get_db()
        try:
            db.execute(
                """
                UPDATE triage_calibration_events
                SET signal_rephrase = signal_rephrase OR %s,
                    signal_correction = signal_correction OR %s,
                    signal_explicit_lookup = signal_explicit_lookup OR %s,
                    signal_abandonment = signal_abandonment OR %s
                WHERE exchange_id = %s
                """,
                (
                    signal_rephrase,
                    signal_correction,
                    signal_explicit_lookup,
                    signal_abandonment,
                    previous_exchange_id,
                )
            )
            logger.debug(
                f"{LOG_PREFIX} Signals detected for {previous_exchange_id[:8]}: "
                f"rephrase={signal_rephrase}, correction={signal_correction}, "
                f"lookup={signal_explicit_lookup}"
            )
        except Exception as e:
            logger.debug(f"{LOG_PREFIX} detect_user_signals failed: {e}")
        finally:
            if not self._db:
                db.close_pool()

    def run_nightly_calibration(self) -> dict:
        """
        24h cycle. Computes correctness scores for uncalibrated events.
        Stores aggregate stats to configs/generated/triage_calibration.json.
        """
        logger.info(f"{LOG_PREFIX} Starting nightly calibration cycle...")
        db = self._get_db()
        stats = {}

        try:
            # 1. Query uncalibrated events from last 24h
            yesterday = datetime.now(timezone.utc) - timedelta(hours=24)
            rows = db.fetch_all(
                """
                SELECT id, exchange_id, triage_branch, triage_mode, tool_selected,
                       outcome_mode, outcome_tools_used, outcome_tool_success,
                       tool_abstention, self_eval_override,
                       signal_rephrase, signal_correction, signal_explicit_lookup, signal_abandonment
                FROM triage_calibration_events
                WHERE correctness_label IS NULL
                AND created_at > %s
                """,
                (yesterday,)
            )

            if not rows or len(rows) < MIN_DAILY_DECISIONS:
                logger.info(
                    f"{LOG_PREFIX} Insufficient data ({len(rows) if rows else 0} events < {MIN_DAILY_DECISIONS}), "
                    f"logging only"
                )
            else:
                # 2. Compute correctness for each event
                results = []
                for row in rows:
                    label, score = self._compute_correctness(row)
                    results.append((label, score, str(row['id'])))

                # 3. Batch update
                for label, score, event_id in results:
                    try:
                        db.execute(
                            """
                            UPDATE triage_calibration_events
                            SET correctness_label = %s, correctness_score = %s
                            WHERE id = %s::uuid
                            """,
                            (label, score, event_id)
                        )
                    except Exception as e:
                        logger.debug(f"{LOG_PREFIX} Update correctness failed for {event_id}: {e}")

                # 4. Aggregate stats
                labels = [r[0] for r in results]
                total = len(labels)
                stats = {
                    'computed_at': datetime.now(timezone.utc).isoformat(),
                    'total_events': total,
                    'correct_rate': labels.count('correct') / total if total else 0,
                    'false_positive_rate': labels.count('false_positive') / total if total else 0,
                    'false_negative_rate': labels.count('false_negative') / total if total else 0,
                    'tool_mismatch_rate': labels.count('tool_mismatch') / total if total else 0,
                    'act_success_rate': self._get_act_success_rate(db),
                }

                logger.info(
                    f"{LOG_PREFIX} Calibration: {total} events, "
                    f"correct={stats['correct_rate']:.1%}, "
                    f"false_pos={stats['false_positive_rate']:.1%}, "
                    f"false_neg={stats['false_negative_rate']:.1%}"
                )

            # 5. Always store stats (even if minimal)
            if not stats:
                stats = {
                    'computed_at': datetime.now(timezone.utc).isoformat(),
                    'total_events': len(rows) if rows else 0,
                    'note': 'insufficient_data',
                }
            self._save_stats(stats)

            # 6. Cleanup events older than retention period
            cutoff = datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)
            try:
                db.execute(
                    "DELETE FROM triage_calibration_events WHERE created_at < %s",
                    (cutoff,)
                )
            except Exception as e:
                logger.debug(f"{LOG_PREFIX} Cleanup failed: {e}")

        except Exception as e:
            logger.error(f"{LOG_PREFIX} Nightly calibration failed: {e}", exc_info=True)
            stats = {'error': str(e), 'computed_at': datetime.now(timezone.utc).isoformat()}
        finally:
            if not self._db:
                db.close_pool()

        return stats

    def _compute_correctness(self, row: dict) -> tuple:
        """Compute (label, score) for a single event."""
        branch = row.get('triage_branch', '')
        outcome_mode = row.get('outcome_mode', '')
        tool_abstention = row.get('tool_abstention', False)
        tool_selected = row.get('tool_selected', []) or []
        outcome_tools = row.get('outcome_tools_used', []) or []

        # Count negative signals (>=2 required for incorrect label)
        neg_signals = sum([
            bool(row.get('signal_rephrase')),
            bool(row.get('signal_correction')),
            bool(row.get('signal_explicit_lookup')),
            bool(row.get('signal_abandonment')),
        ])

        # False positive: selected ACT but tools abstained or >=2 negative signals
        if branch == 'act' and (tool_abstention or neg_signals >= 2):
            return ('false_positive', 0.2)

        # False negative: selected respond/clarify, >=2 of rephrase/correction/explicit_lookup
        if branch in ('respond', 'clarify') and neg_signals >= 2:
            return ('false_negative', 0.2)

        # Tool mismatch: branch=act but different tools actually used
        if branch == 'act' and outcome_tools and tool_selected:
            if set(tool_selected) != set(str(t) for t in outcome_tools):
                return ('tool_mismatch', 0.5)

        # Default: correct
        return ('correct', 1.0)

    def _get_act_success_rate(self, db) -> float:
        """Get ACT success rate from last 7 days."""
        try:
            rows = db.fetch_all(
                """
                SELECT
                    COUNT(*) FILTER (WHERE triage_branch = 'act') AS act_count,
                    COUNT(*) FILTER (WHERE triage_branch = 'act' AND correctness_label = 'correct') AS act_correct
                FROM triage_calibration_events
                WHERE created_at > NOW() - INTERVAL '7 days'
                """
            )
            if rows and rows[0]['act_count']:
                return rows[0]['act_correct'] / rows[0]['act_count']
            return 1.0
        except Exception:
            return 1.0

    def get_calibration_stats(self, days: int = 7) -> dict:
        """Returns aggregate stats for monitoring."""
        try:
            with open(CALIBRATION_STATS_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_stats(self, stats: dict) -> None:
        try:
            with open(CALIBRATION_STATS_FILE, 'w') as f:
                json.dump(stats, f, indent=2)
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} Failed to save stats: {e}")


def triage_calibration_worker(shared_state=None):
    """Entry point for consumer.py service registration. Runs 24h cycle."""
    logger.info("[TRIAGE CALIBRATION WORKER] Starting 24h calibration cycle service...")
    while True:
        try:
            service = TriageCalibrationService()
            service.run_nightly_calibration()
        except Exception as e:
            logger.error(f"[TRIAGE CALIBRATION WORKER] Cycle failed: {e}", exc_info=True)
        time.sleep(24 * 3600)
