"""
Tool Health Service — Ephemeral tool effectiveness tracking via MemoryStore.

Tracks per-tool "potential" (0.0–1.0) that decays on failures and recovers
on successes. Signals persist across ACT loops via MemoryStore with TTL,
so transient issues (rate limits, API outages) are visible to subsequent
loops without permanent penalty.

Design:
  - Potential starts at 1.0 (default when no key exists)
  - Decays multiplicatively on failure (fast descent)
  - Recovers additively on success (slow climb back)
  - TTL = 15 min → self-healing if failures stop
  - Stored in MemoryStore (thread-safe, no schema changes)
"""

import json
import logging
from services.time_utils import utc_now

logger = logging.getLogger(__name__)
LOG_PREFIX = "[TOOL HEALTH]"

# ── Decay / recovery constants ─────────────────────────────────────────
DECAY_EMPTY = 0.5       # Empty/no-results: halve potential
DECAY_CRITIC = 0.4      # Critic correction: harsh penalty
DECAY_ERROR = 0.3       # Error/timeout: severe penalty
RECOVERY_BOOST = 0.15   # Successful result: additive recovery
FLOOR_POTENTIAL = 0.05  # Never reach exactly 0 (always recoverable)

# ── MemoryStore config ─────────────────────────────────────────────────
KEY_PREFIX = "tool_health:"
TTL_SECONDS = 900       # 15 minutes — self-healing window


def _get_store():
    from services.memory_client import MemoryClientService
    return MemoryClientService.create_connection()


def get_potential(tool_name: str) -> float:
    """Read current tool potential (0.0–1.0). Returns 1.0 if no data."""
    try:
        store = _get_store()
        raw = store.get(f"{KEY_PREFIX}{tool_name}")
        if raw is None:
            return 1.0
        data = json.loads(raw if isinstance(raw, str) else raw.decode())
        return data.get('potential', 1.0)
    except Exception:
        return 1.0


def get_all_health() -> dict:
    """
    Return {tool_name: {potential, failures, successes, last_event}} for all
    tools with active health data. Used for world state / observability.
    """
    try:
        store = _get_store()
        keys = store.keys(f"{KEY_PREFIX}*")
        result = {}
        for key in keys:
            key_str = key if isinstance(key, str) else key.decode()
            tool_name = key_str[len(KEY_PREFIX):]
            raw = store.get(key)
            if raw:
                data = json.loads(raw if isinstance(raw, str) else raw.decode())
                result[tool_name] = data
        return result
    except Exception:
        return {}


def record_outcome(tool_name: str, outcome: str) -> float:
    """
    Update tool potential based on action outcome.

    Args:
        tool_name: Tool identifier
        outcome: One of 'success', 'empty', 'critic_correction', 'error', 'timeout'

    Returns:
        New potential value
    """
    try:
        store = _get_store()
        key = f"{KEY_PREFIX}{tool_name}"

        # Read current state
        raw = store.get(key)
        if raw:
            data = json.loads(raw if isinstance(raw, str) else raw.decode())
        else:
            data = {
                'potential': 1.0,
                'failures': 0,
                'successes': 0,
                'last_event': None,
            }

        old_potential = data['potential']

        if outcome == 'success':
            data['potential'] = min(1.0, old_potential + RECOVERY_BOOST)
            data['successes'] = data.get('successes', 0) + 1
        elif outcome == 'empty':
            data['potential'] = max(FLOOR_POTENTIAL, old_potential * DECAY_EMPTY)
            data['failures'] = data.get('failures', 0) + 1
        elif outcome == 'critic_correction':
            data['potential'] = max(FLOOR_POTENTIAL, old_potential * DECAY_CRITIC)
            data['failures'] = data.get('failures', 0) + 1
        elif outcome in ('error', 'timeout'):
            data['potential'] = max(FLOOR_POTENTIAL, old_potential * DECAY_ERROR)
            data['failures'] = data.get('failures', 0) + 1

        data['last_event'] = outcome
        data['updated_at'] = utc_now().isoformat()

        store.setex(key, TTL_SECONDS, json.dumps(data))

        if data['potential'] != old_potential:
            logger.info(
                f"{LOG_PREFIX} {tool_name}: {old_potential:.2f} → {data['potential']:.2f} "
                f"({outcome})"
            )

        return data['potential']

    except Exception as e:
        logger.debug(f"{LOG_PREFIX} Failed to record outcome for {tool_name}: {e}")
        return 1.0


def format_health_hint(tool_potentials: dict) -> str:
    """
    Format tool health signals for injection into ACT loop context.

    Only surfaces tools with degraded potential (< 0.8). Returns empty
    string when all tools are healthy (no noise).
    """
    degraded = {
        name: p for name, p in tool_potentials.items()
        if p < 0.8
    }
    if not degraded:
        return ''

    lines = []
    for name, potential in sorted(degraded.items(), key=lambda x: x[1]):
        if potential < 0.2:
            lines.append(
                f"⚠ {name}: very low effectiveness ({potential:.0%}) — "
                f"consider answering from your own knowledge or trying a different approach"
            )
        elif potential < 0.5:
            lines.append(
                f"⚠ {name}: reduced effectiveness ({potential:.0%}) — "
                f"results have been unreliable, try different queries or tools"
            )
        else:
            lines.append(f"ℹ {name}: slightly degraded ({potential:.0%})")

    return "\n".join(lines)


def classify_result(action_result: dict) -> str:
    """
    Classify an action execution result into an outcome category.

    Examines status, result content, and count fields to determine
    whether the tool returned useful results.
    """
    status = action_result.get('status', '')

    if status == 'timeout':
        return 'timeout'
    if status == 'error':
        return 'error'
    if status == 'critic_correction':
        return 'critic_correction'

    # Success — but was the result actually useful?
    if status == 'success':
        result = action_result.get('result', '')

        # Check for structured results with count/results fields
        if isinstance(result, dict):
            count = result.get('count', -1)
            results_list = result.get('results', None)
            if count == 0 or results_list == []:
                return 'empty'

        # Check for string results that indicate emptiness
        if isinstance(result, str):
            lower = result.lower()
            if any(phrase in lower for phrase in [
                'no results', 'not found', 'no matches',
                'nothing found', 'no items',
            ]):
                return 'empty'

        return 'success'

    return 'success'  # Unknown status → assume ok
