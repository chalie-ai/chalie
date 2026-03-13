"""
Introspect Skill — Thin view on the Self-Model Service.

Delegates to SelfModelService for the always-fresh snapshot, then overlays
topic-specific signals (FOK, recall failure rate) that require the current
conversation topic. Falls back to legacy standalone gathering if the
self-model is unavailable.

Perception directed inward. All computed, no LLM.
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
    try:
        from services.self_model_service import SelfModelService
        service = SelfModelService()
        snapshot = service.get_snapshot()

        # Overlay topic-specific signals not in the base snapshot
        # (base snapshot uses whatever topic was active at refresh time;
        #  the introspect skill is called with the specific current topic)
        ep = snapshot.get("epistemic", {})
        ep["partial_match_signal"] = _get_fok_signal(topic)
        ep["recall_failure_rate"] = _get_recall_failure_rate(topic)
        ep["current_topic"] = topic

        # Gather signals not in self-model (topic-bound, user-facing only)
        extra = {}
        extra["world_state"] = _get_world_state(topic)
        extra["communication_style"] = _get_communication_style()
        extra["decision_explanations"] = _get_recent_decision_explanations()
        extra["recent_autonomous_actions"] = _get_recent_autonomous_actions()
        extra["skill_stats"] = _get_skill_stats(topic)

        # Triage-filtered tool details (context-dependent)
        triage_tools = params.get('triage_tools', [])
        if triage_tools:
            extra["tool_details"] = _get_filtered_tool_details(triage_tools)

        return _format_snapshot(snapshot, extra, topic)

    except Exception as e:
        logger.warning(f"[INTROSPECT] Self-model unavailable, falling back: {e}")
        return _legacy_handle_introspect(topic, params)


# ── Topic-specific signal helpers (kept here, not in self-model) ────


def _get_fok_signal(topic: str) -> int:
    """Read Feeling-of-Knowing signal from last recall operation."""
    try:
        from services.memory_client import MemoryClientService
        store = MemoryClientService.create_connection()
        value = store.get(f"fok:{topic}")
        return int(value) if value else 0
    except Exception:
        return 0


def _get_recall_failure_rate(topic: str) -> float:
    """Per-topic recall failure rate from procedural memory."""
    try:
        from services.procedural_memory_service import ProceduralMemoryService
        from services.database_service import get_shared_db_service
        db_service = get_shared_db_service()
        service = ProceduralMemoryService(db_service)
        action_stats = service.get_action_stats("recall")
        context_stats = (action_stats or {}).get("context_stats") or {}
        if context_stats and topic in context_stats:
            topic_stats = context_stats[topic]
            total = topic_stats.get("total", 0)
            failures = topic_stats.get("failures", 0)
            if total > 0:
                return round(failures / total, 3)
        return 0.0
    except Exception:
        return 0.0


def _get_world_state(topic: str) -> str:
    try:
        from services.world_state_service import WorldStateService
        return WorldStateService().get_world_state(topic)
    except Exception:
        return "(unavailable)"


def _get_communication_style() -> dict:
    try:
        from services.user_trait_service import UserTraitService
        from services.database_service import get_shared_db_service
        return UserTraitService(get_shared_db_service()).get_communication_style()
    except Exception:
        return {}


def _get_recent_decision_explanations(limit: int = 3) -> list:
    """Full decision context for recent routing decisions."""
    try:
        from services.routing_decision_service import RoutingDecisionService
        from services.database_service import get_shared_db_service
        service = RoutingDecisionService(get_shared_db_service())
        decisions = service.get_recent_decisions(hours=1, limit=limit)

        explanations = []
        for d in decisions:
            scores = d.get('scores') or {}
            signals = d.get('signal_snapshot') or {}
            key_signals = {}
            if signals.get('context_warmth') is not None:
                key_signals['context_warmth'] = round(signals['context_warmth'], 2)
            if signals.get('has_question_mark'):
                key_signals['question_detected'] = True
            if signals.get('greeting_pattern'):
                key_signals['greeting_detected'] = True
            if signals.get('explicit_feedback'):
                key_signals['user_feedback'] = signals['explicit_feedback']
            if signals.get('memory_confidence') is not None:
                key_signals['memory_confidence'] = round(signals['memory_confidence'], 2)
            if signals.get('is_new_topic'):
                key_signals['new_topic'] = True

            explanations.append({
                'mode': d.get('selected_mode'),
                'confidence': round(d.get('router_confidence', 0), 3),
                'scores': {k: round(v, 3) for k, v in scores.items()} if scores else {},
                'tiebreaker_used': d.get('tiebreaker_used', False),
                'tiebreaker_candidates': d.get('tiebreaker_candidates'),
                'key_signals': key_signals,
                'margin': round(d.get('margin', 0), 3),
            })
        return explanations
    except Exception:
        return []


def _get_recent_autonomous_actions(limit: int = 5) -> list:
    """Recent autonomous actions from interaction_log."""
    RELEVANT_TYPES = ('proactive_sent', 'cron_tool_executed', 'plan_proposed')
    try:
        from services.database_service import get_shared_db_service
        db_service = get_shared_db_service()
        placeholders = ','.join(['?'] * len(RELEVANT_TYPES))
        with db_service.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT event_type, payload, created_at "
                f"FROM interaction_log "
                f"WHERE event_type IN ({placeholders}) "
                f"ORDER BY created_at DESC LIMIT ?",
                (*RELEVANT_TYPES, limit)
            )
            rows = cursor.fetchall()
            cursor.close()
        return [
            {'event_type': r[0], 'payload_summary': str(r[1] if isinstance(r[1], dict) else {})[:200], 'created_at': str(r[2])}
            for r in rows
        ]
    except Exception:
        return []


def _get_skill_stats(topic: str) -> dict:
    """Full skill stats (weight, successes, failures, reliability) for all actions."""
    try:
        from services.procedural_memory_service import ProceduralMemoryService
        from services.database_service import get_shared_db_service
        service = ProceduralMemoryService(get_shared_db_service())
        stats = {}
        for action_name in service.get_all_policy_weights():
            action_stats = service.get_action_stats(action_name)
            if action_stats:
                attempts = action_stats.get('total_attempts', 0)
                successes = action_stats.get('total_successes', 0)
                avg_reward = action_stats.get('avg_reward', 0) or 0
                smoothed = (successes + 1) / (attempts + 2) if attempts > 0 else 0.5
                stats[action_name] = {
                    "weight": round(action_stats.get('weight', 1.0), 3),
                    "successes": successes,
                    "failures": attempts - successes,
                    "avg_reward": round(avg_reward, 2),
                    "reliability": round(smoothed, 3),
                }
        return stats
    except Exception:
        return {}


def _get_tool_details() -> dict:
    """Full manifest details for all loaded tools."""
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
    except Exception:
        return {}


def _get_filtered_tool_details(tool_names: list) -> dict:
    """Get manifest details for specific tools only (triage-selected)."""
    all_details = _get_tool_details()
    return {k: v for k, v in all_details.items() if k in tool_names}


# ── Formatting ──────────────────────────────────────────────────


def _format_snapshot(snapshot: dict, extra: dict, topic: str) -> str:
    """Format self-model snapshot + topic extras as structured report."""
    ep = snapshot.get("epistemic", {})
    op = snapshot.get("operational", {})
    cap = snapshot.get("capability", {})
    noteworthy = snapshot.get("noteworthy", [])

    lines = [f"[INTROSPECT] Internal state for topic '{topic}':"]

    # Epistemic
    lines.append(f"  context_warmth: {ep.get('context_warmth', 0)}")
    lines.append(f"  working_memory_depth: {ep.get('working_memory_depth', 0)}")
    lines.append(f"  topic_age: {ep.get('topic_age', 'unknown')}")
    lines.append(f"  partial_match_signal: {ep.get('partial_match_signal', 0)}")
    lines.append(f"  recall_failure_rate: {ep.get('recall_failure_rate', 0)}")
    lines.append(f"  focus_active: {ep.get('focus_active', False)}")

    # Communication style
    comm_style = extra.get('communication_style', {})
    if comm_style:
        style_str = ", ".join(f"{k}={v}" for k, v in comm_style.items())
        lines.append(f"  communication_style: {{{style_str}}}")
    else:
        lines.append("  communication_style: (not detected yet)")

    # Recent modes
    modes = ep.get('recent_modes', [])
    lines.append(f"  recent_modes: {modes}" if modes else "  recent_modes: (none)")

    # Skill stats (full detail from extra, not condensed)
    skill_stats = extra.get('skill_stats', {})
    if skill_stats:
        lines.append("  skill_stats:")
        for skill, stats in skill_stats.items():
            lines.append(f"    {skill}: {stats}")
    else:
        lines.append("  skill_stats: (no data yet)")

    # World state
    ws = extra.get('world_state', '')
    lines.append(f"  world_state: {ws.strip()}" if ws and ws.strip() else "  world_state: (empty)")

    # Operational awareness (NEW — from self-model)
    if noteworthy:
        lines.append("  noteworthy_state:")
        for item in noteworthy:
            lines.append(f"    - [{item['severity']:.1f}] {item['signal']}")
    else:
        lines.append("  noteworthy_state: (all systems nominal)")

    # Capability summary
    lines.append(f"  tool_count: {cap.get('tool_count', 0)}")
    cats = cap.get('capability_categories', {})
    if cats:
        lines.append("  capabilities: " + "; ".join(f"{c}: {', '.join(t)}" for c, t in cats.items()))

    # Decision explanations
    decision_exps = extra.get('decision_explanations', [])
    if decision_exps:
        lines.append("  recent_decision_explanations:")
        for i, exp in enumerate(decision_exps):
            lines.append(f"    [{i+1}] mode={exp['mode']}, confidence={exp['confidence']}, margin={exp['margin']}")
            if exp.get('scores'):
                lines.append(f"        scores: {exp['scores']}")
            if exp.get('key_signals'):
                lines.append(f"        key_signals: {exp['key_signals']}")
            if exp.get('tiebreaker_used'):
                lines.append(f"        tiebreaker between: {exp.get('tiebreaker_candidates')}")
    else:
        lines.append("  recent_decision_explanations: (none in last hour)")

    # Autonomous actions
    auto_actions = extra.get('recent_autonomous_actions', [])
    if auto_actions:
        lines.append("  recent_autonomous_actions:")
        for a in auto_actions:
            lines.append(f"    - {a['event_type']} at {a['created_at']}: {a['payload_summary'][:100]}")
    else:
        lines.append("  recent_autonomous_actions: (none recently)")

    # Tool details (triage-selected)
    tool_details = extra.get('tool_details')
    if tool_details:
        lines.append("  tool_details (triage-selected):")
        for tname, tinfo in tool_details.items():
            lines.append(f"    {tname}:")
            if tinfo.get('tips'):
                lines.append(f"      tips: {tinfo['tips']}")
            if tinfo.get('constraints'):
                lines.append(f"      constraints: {tinfo['constraints']}")
            if tinfo.get('examples'):
                examples_short = [str(e)[:120] for e in tinfo['examples'][:3]]
                lines.append(f"      examples: {examples_short}")

    return "\n".join(lines)


# ── Legacy fallback ─────────────────────────────────────────────


def _legacy_handle_introspect(topic: str, params: dict) -> str:
    """Standalone introspect without self-model (fallback)."""
    from services.memory_client import MemoryClientService
    store = MemoryClientService.create_connection()

    state = {}
    state["working_memory_depth"] = store.llen(f"working_memory:{topic}")

    wm_score = min(1.0, state["working_memory_depth"] / 4.0)
    state["context_warmth"] = round(wm_score, 3)
    state["partial_match_signal"] = _get_fok_signal(topic)
    state["recall_failure_rate"] = _get_recall_failure_rate(topic)
    state["world_state"] = _get_world_state(topic)
    state["topic_age"] = "unknown"
    state["recent_modes"] = []
    state["skill_stats"] = _get_skill_stats(topic)
    state["focus_active"] = False
    state["communication_style"] = _get_communication_style()
    state["decision_explanations"] = _get_recent_decision_explanations()
    state["recent_autonomous_actions"] = _get_recent_autonomous_actions()

    triage_tools = params.get('triage_tools', [])
    if triage_tools:
        state["tool_details"] = _get_filtered_tool_details(triage_tools)

    return _legacy_format(state, topic)


def _legacy_format(state: Dict, topic: str) -> str:
    """Format state dict in legacy format."""
    lines = [f"[INTROSPECT] Internal state for topic '{topic}':"]
    for key in ['context_warmth', 'working_memory_depth', 'topic_age',
                'partial_match_signal', 'recall_failure_rate']:
        lines.append(f"  {key}: {state.get(key, 'N/A')}")
    lines.append(f"  focus_active: {state.get('focus_active', False)}")
    return "\n".join(lines)
