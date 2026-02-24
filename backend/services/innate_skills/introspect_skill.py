"""
Introspect Skill — Self-examination of internal state and metacognitive signals.

Perception directed inward. Returns context warmth, memory density, skill stats,
FOK signals, and world state — all computed, no LLM.
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


def handle_introspect(topic: str, params: dict) -> str:
    """
    Gather and report internal state + metacognitive signals.

    Args:
        topic: Current conversation topic
        params: {} (no parameters required)

    Returns:
        Structured state report
    """
    state = {}

    # Gather signals from Redis
    state["gist_count"] = _get_gist_count(topic)
    state["fact_count"] = _get_fact_count(topic)
    state["working_memory_depth"] = _get_working_memory_depth(topic)
    state["context_warmth"] = _compute_context_warmth(state)
    state["partial_match_signal"] = _get_fok_signal(topic)
    state["world_state"] = _get_world_state(topic)
    state["topic_age"] = _get_topic_age(topic)

    # Gather signals from PostgreSQL
    state["recent_modes"] = _get_recent_modes(topic)
    state["skill_stats"] = _get_skill_stats(topic)
    state["recall_failure_rate"] = _get_recall_failure_rate(topic)
    state["tool_details"] = _get_tool_details()

    # Cross-feature signals
    state["focus_active"] = _get_focus_active(params.get('thread_id', topic))
    state["communication_style"] = _get_communication_style()

    return _format_state(state, topic)


def _get_gist_count(topic: str) -> int:
    """Count active gists for topic."""
    try:
        from services.gist_storage_service import GistStorageService

        service = GistStorageService()
        gists = service.get_latest_gists(topic)
        return len(gists)
    except Exception as e:
        logger.warning(f"[INTROSPECT] gist count failed: {e}")
        return 0


def _get_fact_count(topic: str) -> int:
    """Count active facts for topic."""
    try:
        from services.fact_store_service import FactStoreService

        service = FactStoreService()
        facts = service.get_all_facts(topic)
        return len(facts)
    except Exception as e:
        logger.warning(f"[INTROSPECT] fact count failed: {e}")
        return 0


def _get_working_memory_depth(topic: str) -> int:
    """Count turns in working memory buffer."""
    try:
        from services.redis_client import RedisClientService

        redis = RedisClientService.create_connection()
        key = f"working_memory:{topic}"
        return redis.llen(key)
    except Exception as e:
        logger.warning(f"[INTROSPECT] working memory depth failed: {e}")
        return 0


def _compute_context_warmth(state: Dict) -> float:
    """
    Compute context warmth (0.0-1.0) from memory density signals.

    context_warmth is a heuristic for "how much do I know about this topic?"
    Higher = more memory resources available.
    """
    gist_score = min(1.0, state.get("gist_count", 0) / 5.0)
    fact_score = min(1.0, state.get("fact_count", 0) / 10.0)
    wm_score = min(1.0, state.get("working_memory_depth", 0) / 4.0)

    # Weighted combination: gists most important for warmth
    warmth = (gist_score * 0.4) + (fact_score * 0.3) + (wm_score * 0.3)
    return round(warmth, 3)


def _get_fok_signal(topic: str) -> int:
    """
    Read Feeling-of-Knowing signal from last recall operation.
    Stored by recall skill as partial match count.
    """
    try:
        from services.redis_client import RedisClientService

        redis = RedisClientService.create_connection()
        value = redis.get(f"fok:{topic}")
        return int(value) if value else 0
    except Exception:
        return 0


def _get_world_state(topic: str) -> str:
    """Get current world state for topic."""
    try:
        from services.world_state_service import WorldStateService

        service = WorldStateService()
        return service.get_world_state(topic)
    except Exception as e:
        logger.warning(f"[INTROSPECT] world state failed: {e}")
        return "(unavailable)"


def _get_topic_age(topic: str) -> str:
    """Get how long the current topic has been active."""
    try:
        from services.redis_client import RedisClientService

        redis = RedisClientService.create_connection()
        ttl = redis.ttl(f"recent_topic")
        if ttl and ttl > 0:
            # recent_topic has 30min TTL, so age = 1800 - remaining TTL
            age_seconds = 1800 - ttl
            if age_seconds < 60:
                return f"{age_seconds}s"
            elif age_seconds < 3600:
                return f"{age_seconds // 60}min"
            else:
                return f"{age_seconds // 3600}h {(age_seconds % 3600) // 60}min"
        return "unknown"
    except Exception:
        return "unknown"


def _get_recent_modes(topic: str) -> list:
    """Get last N mode selections from routing decisions."""
    try:
        from services.routing_decision_service import RoutingDecisionService
        from services.database_service import get_shared_db_service

        db_service = get_shared_db_service()
        service = RoutingDecisionService(db_service)

        decisions = service.get_recent_decisions(hours=1, limit=5)
        return [d["selected_mode"] for d in decisions]
    except Exception as e:
        logger.warning(f"[INTROSPECT] recent modes failed: {e}")
        return []


def _get_skill_stats(topic: str) -> dict:
    """
    Get stats for all actions from procedural memory.

    Dynamically queries all actions (innate skills + tools) instead of
    hardcoding skill names. Includes trust metrics: weight, successes,
    failures, avg_reward, and Bayesian-smoothed reliability.
    """
    try:
        from services.procedural_memory_service import ProceduralMemoryService
        from services.database_service import get_shared_db_service

        db_service = get_shared_db_service()
        service = ProceduralMemoryService(db_service)

        stats = {}
        # Query all actions dynamically
        all_weights = service.get_all_policy_weights()
        for action_name in all_weights:
            action_stats = service.get_action_stats(action_name)
            if action_stats:
                attempts = action_stats.get('total_attempts', 0)
                successes = action_stats.get('total_successes', 0)
                failures = attempts - successes
                avg_reward = action_stats.get('avg_reward', 0) or 0

                # Bayesian smoothed reliability (Laplace smoothing)
                smoothed_reliability = (successes + 1) / (attempts + 2) if attempts > 0 else 0.5

                stats[action_name] = {
                    "weight": round(action_stats.get('weight', 1.0), 3),
                    "successes": successes,
                    "failures": failures,
                    "avg_reward": round(avg_reward, 2),
                    "reliability": round(smoothed_reliability, 3),
                }

        return stats

    except Exception as e:
        logger.warning(f"[INTROSPECT] skill stats failed: {e}")
        return {}


def _get_tool_details() -> dict:
    """
    Get full manifest details for all loaded tools.

    Returns tips, examples, constraints from each tool's manifest.
    Available via introspect for the ACT prompt to access on demand.
    """
    try:
        from services.tool_registry_service import ToolRegistryService
        registry = ToolRegistryService()

        details = {}
        for tool_name in registry.get_tool_names():
            manifest = registry.get_tool_full_description(tool_name)
            if manifest:
                details[tool_name] = {
                    "description": manifest.get("description", ""),
                    "tips": manifest.get("tips", []),
                    "examples": manifest.get("examples", []),
                    "constraints": manifest.get("constraints", {}),
                }
        return details
    except Exception as e:
        logger.debug(f"[INTROSPECT] tool details unavailable: {e}")
        return {}


def _get_focus_active(thread_id: str) -> bool:
    """Check if a focus session is active for this thread."""
    try:
        from services.focus_session_service import FocusSessionService
        focus = FocusSessionService().get_focus(thread_id)
        return focus is not None
    except Exception as e:
        logger.debug(f"[INTROSPECT] focus active check failed: {e}")
        return False


def _get_communication_style() -> dict:
    """Get detected communication style dimensions."""
    try:
        from services.user_trait_service import UserTraitService
        from services.database_service import get_shared_db_service

        db_service = get_shared_db_service()
        service = UserTraitService(db_service)
        return service.get_communication_style()
    except Exception as e:
        logger.debug(f"[INTROSPECT] communication style failed: {e}")
        return {}


def _get_recall_failure_rate(topic: str) -> float:
    """
    Get per-topic recall failure rate from procedural memory.

    When high, signals ACT that internal retrieval is unreliable
    for this topic and delegation may be appropriate.
    """
    try:
        from services.procedural_memory_service import ProceduralMemoryService
        from services.database_service import get_shared_db_service

        db_service = get_shared_db_service()
        service = ProceduralMemoryService(db_service)

        # Get per-topic context stats from procedural memory's context_stats column
        action_stats = service.get_action_stats("recall")
        context_stats = (action_stats or {}).get("context_stats") or {}
        if context_stats and topic in context_stats:
            topic_stats = context_stats[topic]
            total = topic_stats.get("total", 0)
            failures = topic_stats.get("failures", 0)
            if total > 0:
                return round(failures / total, 3)

        return 0.0

    except Exception as e:
        logger.warning(f"[INTROSPECT] recall failure rate failed: {e}")
        return 0.0


def _format_state(state: Dict, topic: str) -> str:
    """Format internal state as structured report."""
    lines = [f"[INTROSPECT] Internal state for topic '{topic}':"]
    lines.append(f"  context_warmth: {state['context_warmth']}")
    lines.append(f"  gist_count: {state['gist_count']}")
    lines.append(f"  fact_count: {state['fact_count']}")
    lines.append(f"  working_memory_depth: {state['working_memory_depth']}")
    lines.append(f"  topic_age: {state['topic_age']}")
    lines.append(f"  partial_match_signal: {state['partial_match_signal']}")
    lines.append(f"  recall_failure_rate: {state['recall_failure_rate']}")

    # Cross-feature signals
    lines.append(f"  focus_active: {state.get('focus_active', False)}")
    comm_style = state.get('communication_style', {})
    if comm_style:
        style_str = ", ".join(f"{k}={v}" for k, v in comm_style.items())
        lines.append(f"  communication_style: {{{style_str}}}")
    else:
        lines.append("  communication_style: (not detected yet)")

    if state["recent_modes"]:
        lines.append(f"  recent_modes: {state['recent_modes']}")
    else:
        lines.append("  recent_modes: (none)")

    if state["skill_stats"]:
        lines.append("  skill_stats:")
        for skill, stats in state["skill_stats"].items():
            lines.append(f"    {skill}: {stats}")
    else:
        lines.append("  skill_stats: (no data yet)")

    # World state (may be multiline)
    ws = state.get("world_state", "")
    if ws and ws.strip():
        lines.append(f"  world_state: {ws.strip()}")
    else:
        lines.append("  world_state: (empty)")

    # Tool details (full manifest info for on-demand reference)
    tool_details = state.get("tool_details", {})
    if tool_details:
        lines.append("  tool_details:")
        for tool_name, details in tool_details.items():
            lines.append(f"    {tool_name}: {details.get('description', '')}")
            constraints = details.get("constraints", {})
            if constraints:
                lines.append(f"      constraints: {constraints}")
            tips = details.get("tips", [])
            for tip in tips:
                lines.append(f"      tip: {tip}")
            examples = details.get("examples", [])
            for ex in examples:
                lines.append(f"      example: {ex.get('description', '')} — {ex.get('params', {})}")

    return "\n".join(lines)
