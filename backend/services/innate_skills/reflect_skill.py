"""
Reflect Skill — On-demand experiential reflection.

Synthesizes recent experience into insight: what worked, what didn't,
patterns noticed, connections formed. Unlike introspect (raw state snapshot)
or recall (memory fetch without synthesis), this skill uses a lightweight
LLM call to produce genuine synthesis from ACT loop outcomes, episodes,
concepts, and strategy patterns.

This is an LLM-assisted skill (not sub-cortical) due to the synthesis step.
"""

import json
import logging
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

LOG_PREFIX = "[REFLECT SKILL]"


def handle_reflect(topic: str, params: dict) -> str:
    """
    Synthesize recent experience into reflective insight.

    Args:
        topic: Current conversation topic
        params: {
            query (str, optional): What to reflect on. Defaults to recent experience.
            scope (str, optional): "recent" (last few interactions, default),
                                   "session" (current thread), "broad" (wider search)
        }

    Returns:
        Synthesis text (LLM-generated) or structured fallback summary.
    """
    query = params.get('query', '').strip() or topic
    scope = params.get('scope', 'recent')

    # Determine retrieval limits based on scope
    iteration_limit = {'recent': 10, 'session': 20, 'broad': 30}.get(scope, 10)
    episode_limit = {'recent': 3, 'session': 5, 'broad': 5}.get(scope, 3)
    concept_limit = {'recent': 3, 'session': 5, 'broad': 5}.get(scope, 3)

    # ── 1. Retrieve ACT loop outcomes from cortex_iterations ──────────
    iterations = _get_recent_iterations(topic, iteration_limit)

    # ── 2. Retrieve relevant episodes ─────────────────────────────────
    episodes = _get_relevant_episodes(query, topic, episode_limit)

    # ── 3. Retrieve relevant concepts ─────────────────────────────────
    concepts = _get_relevant_concepts(query, concept_limit)

    # ── 4. Analyze ACT strategies ─────────────────────────────────────
    strategy_insight = _analyze_act_strategies(topic)

    # ── 5. Build context and synthesize via LLM ───────────────────────
    context_block = _build_context_block(
        topic, query, iterations, episodes, concepts, strategy_insight
    )

    synthesis = _synthesize(context_block)

    return synthesis


# ── Data retrieval helpers ────────────────────────────────────────────────


def _get_recent_iterations(topic: str, limit: int) -> List[Dict]:
    """Fetch recent ACT loop iterations from cortex_iterations."""
    try:
        from services.database_service import get_shared_db_service
        from services.time_utils import parse_utc
        db = get_shared_db_service()
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT loop_id, iteration_number, chosen_mode, chosen_confidence,
                       actions_executed, net_value, iteration_cost, termination_reason,
                       created_at
                FROM cortex_iterations
                WHERE topic = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (topic, limit))
            rows = cursor.fetchall()
            cursor.close()

        results = []
        for row in rows:
            actions = []
            if row[4]:
                try:
                    actions = json.loads(row[4]) if isinstance(row[4], str) else row[4]
                except (json.JSONDecodeError, TypeError):
                    pass
            results.append({
                'loop_id': row[0],
                'iteration_number': row[1],
                'chosen_mode': row[2],
                'chosen_confidence': row[3],
                'actions': actions,
                'net_value': row[5],
                'iteration_cost': row[6],
                'termination_reason': row[7],
                'created_at': str(row[8]) if row[8] else None,
            })
        return results
    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Failed to retrieve iterations: {e}")
        return []


def _get_relevant_episodes(query: str, topic: str, limit: int) -> List[Dict]:
    """Retrieve relevant episodes via EpisodicRetrievalService."""
    try:
        from services.database_service import get_shared_db_service
        from services.episodic_retrieval_service import EpisodicRetrievalService
        from services.config_service import ConfigService

        db = get_shared_db_service()
        episodic_config = ConfigService.resolve_agent_config("episodic-memory")
        retrieval = EpisodicRetrievalService(db, episodic_config)
        episodes = retrieval.retrieve_episodes(query_text=query, topic=topic, limit=limit)
        # Return simplified dicts
        return [
            {
                'narrative': ep.get('narrative', '')[:300],
                'topic': ep.get('topic', ''),
                'composite_score': ep.get('composite_score', 0),
            }
            for ep in episodes
        ]
    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Failed to retrieve episodes: {e}")
        return []


def _get_relevant_concepts(query: str, limit: int) -> List[Dict]:
    """Retrieve relevant concepts via SemanticRetrievalService."""
    try:
        from services.database_service import get_shared_db_service
        from services.semantic_retrieval_service import SemanticRetrievalService

        db = get_shared_db_service()
        retrieval = SemanticRetrievalService(db)
        concepts = retrieval.retrieve_concepts(query=query, limit=limit)
        return [
            {
                'concept_name': c.get('concept_name', ''),
                'description': (c.get('description', '') or '')[:200],
                'confidence': c.get('confidence', 0),
            }
            for c in concepts
        ]
    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Failed to retrieve concepts: {e}")
        return []


def _analyze_act_strategies(topic: str) -> Optional[Dict]:
    """
    Compare recent ACT loop tool combinations for strategy insights.

    Standalone implementation derived from ReflectAction._analyze_act_strategies.
    Queries cortex_iterations to find which action-type combos performed best.
    """
    try:
        from services.database_service import get_shared_db_service
        from services.time_utils import parse_utc
        db = get_shared_db_service()

        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT loop_id, net_value, started_at, completed_at,
                       termination_reason, actions_executed
                FROM cortex_iterations
                WHERE topic = ?
                  AND created_at > datetime('now', '-24 hours')
                  AND actions_executed IS NOT NULL
                  AND json_array_length(actions_executed) > 0
                ORDER BY created_at DESC
                LIMIT 50
            """, (topic,))
            raw_rows = cursor.fetchall()
            cursor.close()

        if len(raw_rows) < 2:
            return None

        # Group by loop_id
        loop_data: Dict[str, Dict] = {}
        for row in raw_rows:
            loop_id = row[0]
            net_value = row[1] or 0.0
            started_at = row[2]
            completed_at = row[3]
            actions_json = row[5]

            tool_types: set = set()
            if actions_json:
                try:
                    actions = json.loads(actions_json) if isinstance(actions_json, str) else actions_json
                    for a in actions:
                        if isinstance(a, dict) and 'action_type' in a:
                            tool_types.add(a['action_type'])
                except (json.JSONDecodeError, TypeError):
                    pass

            if loop_id not in loop_data:
                loop_data[loop_id] = {
                    'tool_types': set(),
                    'total_net_value': 0.0,
                    'iteration_count': 0,
                    'loop_start': started_at,
                    'loop_end': completed_at,
                }
            entry = loop_data[loop_id]
            entry['tool_types'] |= tool_types
            entry['total_net_value'] += net_value
            entry['iteration_count'] += 1
            if started_at and (entry['loop_start'] is None or started_at < entry['loop_start']):
                entry['loop_start'] = started_at
            if completed_at and (entry['loop_end'] is None or completed_at > entry['loop_end']):
                entry['loop_end'] = completed_at

        loops = list(loop_data.values())[:10]
        if len(loops) < 2:
            return None

        strategy_outcomes: Dict[frozenset, List[Dict]] = {}
        for loop in loops:
            tools = frozenset(loop['tool_types'])
            net_value = loop['total_net_value']
            iterations = loop['iteration_count']

            seconds = 0.0
            if loop['loop_start'] and loop['loop_end']:
                try:
                    t_start = parse_utc(loop['loop_start'])
                    t_end = parse_utc(loop['loop_end'])
                    seconds = (t_end - t_start).total_seconds()
                except Exception:
                    pass

            complexity = 'simple' if iterations <= 2 else ('moderate' if iterations <= 4 else 'complex')
            strategy_outcomes.setdefault(tools, []).append({
                'net_value': net_value,
                'iterations': iterations,
                'seconds': seconds,
                'complexity': complexity,
            })

        if len(strategy_outcomes) < 2:
            return None

        ranked = sorted(
            strategy_outcomes.items(),
            key=lambda x: sum(e['net_value'] for e in x[1]) / len(x[1]),
            reverse=True,
        )
        best = ranked[0]
        worst = ranked[-1]
        best_entries = best[1]
        worst_entries = worst[1]

        return {
            'best_strategy': ', '.join(sorted(best[0])),
            'best_avg_value': sum(e['net_value'] for e in best_entries) / len(best_entries),
            'best_avg_seconds': sum(e['seconds'] for e in best_entries) / len(best_entries),
            'best_complexity': max(
                set(e['complexity'] for e in best_entries),
                key=list(e['complexity'] for e in best_entries).count,
            ),
            'worst_strategy': ', '.join(sorted(worst[0])),
            'worst_avg_value': sum(e['net_value'] for e in worst_entries) / len(worst_entries),
            'worst_avg_seconds': sum(e['seconds'] for e in worst_entries) / len(worst_entries),
            'loops_analyzed': len(loops),
        }
    except Exception as e:
        logger.debug(f"{LOG_PREFIX} Strategy analysis failed: {e}")
        return None


# ── Context building and LLM synthesis ────────────────────────────────────


def _build_context_block(
    topic: str,
    query: str,
    iterations: List[Dict],
    episodes: List[Dict],
    concepts: List[Dict],
    strategy_insight: Optional[Dict],
) -> str:
    """Assemble all retrieved data into a structured context block for the LLM."""
    lines = [f"Topic: {topic}", f"Reflection query: {query}", ""]

    # ACT loop iterations
    if iterations:
        lines.append("## Recent ACT loop outcomes")
        for it in iterations:
            action_types = []
            for a in (it.get('actions') or []):
                if isinstance(a, dict):
                    action_types.append(a.get('action_type', '?'))
            actions_str = ', '.join(action_types) if action_types else 'none'
            lines.append(
                f"- loop={it['loop_id'][:8]}.. iter={it['iteration_number']} "
                f"mode={it['chosen_mode']} conf={it['chosen_confidence']} "
                f"net_value={it['net_value']} cost={it['iteration_cost']} "
                f"actions=[{actions_str}] "
                f"termination={it['termination_reason']}"
            )
        lines.append("")

    # Episodes
    if episodes:
        lines.append("## Relevant episodic memories")
        for ep in episodes:
            lines.append(
                f"- [{ep['topic']}] (score={ep['composite_score']:.2f}) "
                f"{ep['narrative']}"
            )
        lines.append("")

    # Concepts
    if concepts:
        lines.append("## Related knowledge concepts")
        for c in concepts:
            lines.append(
                f"- {c['concept_name']} (confidence={c['confidence']}) "
                f"{c['description']}"
            )
        lines.append("")

    # Strategy insight
    if strategy_insight:
        lines.append("## Strategy analysis (last 24h)")
        lines.append(
            f"- Best strategy: [{strategy_insight['best_strategy']}] "
            f"avg_value={strategy_insight['best_avg_value']:.2f} "
            f"avg_time={strategy_insight['best_avg_seconds']:.0f}s "
            f"complexity={strategy_insight['best_complexity']}"
        )
        lines.append(
            f"- Worst strategy: [{strategy_insight['worst_strategy']}] "
            f"avg_value={strategy_insight['worst_avg_value']:.2f} "
            f"avg_time={strategy_insight['worst_avg_seconds']:.0f}s"
        )
        lines.append(f"- Loops analyzed: {strategy_insight['loops_analyzed']}")
        lines.append("")

    return "\n".join(lines)


def _synthesize(context_block: str) -> str:
    """
    Send context to a lightweight LLM for synthesis.

    Falls back to returning the raw context block if the LLM call fails.
    """
    system_prompt = (
        "You are a reflective cognitive process. Given the context below "
        "(ACT loop outcomes, episodic memories, knowledge concepts, strategy analysis), "
        "synthesize a concise reflection covering:\n"
        "1. What worked well and why\n"
        "2. What didn't work or could improve\n"
        "3. Patterns noticed across recent actions\n"
        "4. Connections to existing knowledge\n\n"
        "Be specific and actionable. Reference concrete outcomes from the data. "
        "Keep the reflection to 3-6 sentences. Do not repeat the raw data."
    )

    if not context_block.strip():
        return "[REFLECT] No recent experience data available for reflection."

    try:
        from services.background_llm_queue import create_background_llm_proxy
        llm = create_background_llm_proxy("reflect-skill")
        response = llm.send_message(system_prompt, context_block)
        if response and response.text:
            return response.text.strip()
        # LLM returned nothing — fall through to fallback
        logger.warning(f"{LOG_PREFIX} LLM returned empty response, using fallback")
    except Exception as e:
        logger.warning(f"{LOG_PREFIX} LLM synthesis failed: {e}")

    # Fallback: return structured summary without synthesis
    return f"[REFLECT] Raw reflection context (LLM synthesis unavailable):\n{context_block}"


