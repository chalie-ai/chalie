"""
Constraint Memory Service — Query layer for gate rejection signals.

Reads rejection events from interaction_log, computes rolling summaries,
caches them in MemoryStore, and formats constraint context for LLM prompts.

This service makes deterministic gate decisions visible to the memory pipeline
and LLM prompts, so the cognitive architecture can learn from what it
considered but couldn't do.
"""

import json
import logging
from collections import Counter
from typing import Dict, Any, List, Optional

from services.memory_client import MemoryClientService

logger = logging.getLogger(__name__)

LOG_PREFIX = "[CONSTRAINT MEMORY]"

# All rejection event types written by gate systems
ALL_REJECTION_TYPES = (
    'action_gate_rejected',
    'plan_rejected',
    'assimilation_rejected',
    'triage_override',
    'routing_anti_oscillation',
    'reflex_rejected',
    'reliability_warning',
    'uncertainty_downgraded',
)

# MemoryStore cache keys
_SUMMARY_KEY = "constraint_memory:summary"
_SUMMARY_TTL = 60  # seconds


class ConstraintMemoryService:
    """Query layer for constraint/gate rejection signals."""

    def __init__(self, db_service=None):
        if db_service is None:
            from services.database_service import get_shared_db_service
            db_service = get_shared_db_service()
        self.db_service = db_service
        self.store = MemoryClientService.create_connection()

    def get_recent_rejections(self, hours: int = 24, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch raw rejection events from interaction_log, newest first."""
        from services.interaction_log_service import InteractionLogService
        log_service = InteractionLogService(self.db_service)
        return log_service.get_events_by_types(
            event_types=list(ALL_REJECTION_TYPES),
            since_hours=hours,
            limit=limit,
        )

    def get_constraint_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Grouped summary of recent constraint activity.

        MemoryStore-cached (60s TTL) to avoid repeated DB queries.

        Returns:
            {
                'rejection_counts': {event_type: count},
                'top_reasons': [str],
                'blocked_actions': [str],
                'total_rejections': int,
            }
        """
        # Check cache
        cached = self.store.get(_SUMMARY_KEY)
        if cached:
            try:
                return json.loads(cached)
            except (json.JSONDecodeError, TypeError):
                pass

        # Build summary from raw events
        events = self.get_recent_rejections(hours=hours, limit=200)
        summary = self._build_summary(events)

        # Cache
        try:
            self.store.setex(_SUMMARY_KEY, _SUMMARY_TTL, json.dumps(summary))
        except Exception:
            pass

        return summary

    def format_for_prompt(self, mode: str = 'act', max_tokens: int = 200) -> str:
        """
        Compact LLM-readable constraint context string.

        Returns empty string when no noteworthy constraints exist.
        Mode-specific formatting:
          - act: blocked actions, reliability warnings, failure patterns
          - plan: plan rejection patterns, tool combination failures
          - respond: only capability gaps (very light)
          - drift: blocked paths for creative routing
        """
        summary = self.get_constraint_summary()

        if summary['total_rejections'] == 0:
            return ''

        lines = []

        if mode == 'act':
            lines = self._format_act(summary)
        elif mode == 'plan':
            lines = self._format_plan(summary)
        elif mode == 'respond':
            lines = self._format_respond(summary)
        elif mode == 'drift':
            lines = self._format_drift(summary)
        else:
            lines = self._format_act(summary)

        if not lines:
            return ''

        result = '\n'.join(lines)
        # Budget guard: rough truncation at estimated token limit
        if len(result) > max_tokens * 4:
            result = result[:max_tokens * 4] + '\n...(truncated)'
        return result

    def get_blocked_action_patterns(self, hours: int = 48) -> List[Dict[str, Any]]:
        """
        Actions with 3+ gate rejections in the window, sorted by frequency.

        Used by procedural memory and idle consolidation.
        """
        events = self.get_recent_rejections(hours=hours, limit=500)

        action_reasons: Dict[str, Counter] = {}

        for event in events:
            payload = event.get('payload', {})
            event_type = event.get('event_type', '')

            if event_type == 'action_gate_rejected':
                rejections = payload.get('rejections', [])
                for r in rejections:
                    action = r.get('action', 'unknown')
                    reason = r.get('reason', 'unknown')
                    if action not in action_reasons:
                        action_reasons[action] = Counter()
                    action_reasons[action][reason] += 1
            elif event_type == 'triage_override':
                rule = payload.get('rule', 'unknown')
                key = f"triage:{payload.get('original_mode', '?')}→{payload.get('final_mode', '?')}"
                if key not in action_reasons:
                    action_reasons[key] = Counter()
                action_reasons[key][rule] += 1
            elif event_type == 'plan_rejected':
                key = 'plan_decomposition'
                reason = payload.get('rejection_type', 'unknown')
                if key not in action_reasons:
                    action_reasons[key] = Counter()
                action_reasons[key][reason] += 1

        # Filter to 3+ rejections and sort by total
        patterns = []
        for action, reasons in action_reasons.items():
            total = sum(reasons.values())
            if total >= 3:
                top_reason = reasons.most_common(1)[0][0] if reasons else 'unknown'
                patterns.append({
                    'action': action,
                    'total_rejections': total,
                    'top_reason': top_reason,
                    'reason_breakdown': dict(reasons.most_common(5)),
                })

        patterns.sort(key=lambda p: p['total_rejections'], reverse=True)
        return patterns

    # ── Internal helpers ─────────────────────────────────────────────────

    @staticmethod
    def _build_summary(events: List[Dict]) -> Dict[str, Any]:
        """Build grouped summary from raw rejection events."""
        type_counts: Counter = Counter()
        reason_counts: Counter = Counter()
        blocked_actions: Counter = Counter()

        for event in events:
            event_type = event.get('event_type', '')
            payload = event.get('payload', {})
            type_counts[event_type] += 1

            # Extract reasons based on event type
            if event_type == 'action_gate_rejected':
                for r in payload.get('rejections', []):
                    reason_counts[r.get('reason', 'unknown')] += 1
                    blocked_actions[r.get('action', 'unknown')] += 1
            elif event_type == 'plan_rejected':
                reason_counts[payload.get('rejection_type', 'unknown')] += 1
            elif event_type == 'triage_override':
                reason_counts[payload.get('rule', 'unknown')] += 1
            elif event_type == 'reflex_rejected':
                reason_counts[payload.get('reasoning', 'unknown')[:60]] += 1
            elif event_type == 'routing_anti_oscillation':
                reason_counts[f"anti_oscillation_{payload.get('suppressed_mode', '?')}"] += 1
            elif event_type == 'reliability_warning':
                reason_counts[f"unreliable_{payload.get('memory_type', '?')}"] += 1
            elif event_type == 'uncertainty_downgraded':
                reason_counts['uncertainty_anti_nag'] += 1
            elif event_type == 'assimilation_rejected':
                reason_counts[payload.get('rejection_type', 'unknown')] += 1

        return {
            'rejection_counts': dict(type_counts),
            'top_reasons': [r for r, _ in reason_counts.most_common(5)],
            'blocked_actions': [a for a, _ in blocked_actions.most_common(5)],
            'total_rejections': sum(type_counts.values()),
        }

    @staticmethod
    def _format_act(summary: Dict) -> List[str]:
        """Format constraint context for ACT mode."""
        lines = []
        blocked = summary.get('blocked_actions', [])
        if blocked:
            lines.append(f"Recently blocked actions: {', '.join(blocked[:3])}")

        counts = summary.get('rejection_counts', {})
        if counts.get('reliability_warning', 0) > 0:
            lines.append(f"Reliability warnings active ({counts['reliability_warning']} recent)")

        if counts.get('plan_rejected', 0) >= 2:
            lines.append(f"Recent plan decomposition failures ({counts['plan_rejected']}x)")

        reasons = summary.get('top_reasons', [])
        if reasons:
            lines.append(f"Top constraint reasons: {', '.join(reasons[:3])}")

        return lines

    @staticmethod
    def _format_plan(summary: Dict) -> List[str]:
        """Format constraint context for plan decomposition."""
        lines = []
        counts = summary.get('rejection_counts', {})

        if counts.get('plan_rejected', 0) > 0:
            lines.append(f"Previous plan rejections: {counts['plan_rejected']}x in last 24h")

        reasons = summary.get('top_reasons', [])
        plan_reasons = [r for r in reasons if r in (
            'dag_invalid', 'step_quality', 'step_count_bounds',
            'low_confidence', 'llm_call_failed', 'parse_failed',
        )]
        if plan_reasons:
            lines.append(f"Common failure modes: {', '.join(plan_reasons)}")

        return lines

    @staticmethod
    def _format_respond(summary: Dict) -> List[str]:
        """Format constraint context for RESPOND mode (very light)."""
        counts = summary.get('rejection_counts', {})
        # Only surface if there are capability gaps
        if counts.get('triage_override', 0) > 0:
            return [f"Note: {counts['triage_override']} recent triage overrides active"]
        return []

    @staticmethod
    def _format_drift(summary: Dict) -> List[str]:
        """Format constraint context for cognitive drift."""
        lines = []
        blocked = summary.get('blocked_actions', [])
        if blocked:
            lines.append(f"Currently constrained actions: {', '.join(blocked[:3])}")

        reasons = summary.get('top_reasons', [])
        if reasons:
            lines.append(f"Active constraints: {', '.join(reasons[:3])}")

        return lines
