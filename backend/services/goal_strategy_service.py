"""
Goal Strategy Service -- Lightweight strategy generation on actionable transition.

When a goal transitions to actionable (sufficient salience + confidence),
generates a 1-3 sentence strategy via LLM. The strategy:
  - Guides the proactive service on HOW to act
  - Feeds into persistent task scope when ACT-style execution triggers
  - Keeps the LLM call lightweight (~500 tokens prompt, ~100 tokens response)

Called exactly once per actionable transition.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)
LOG_PREFIX = "[GOAL STRATEGY]"

# Maximum characters for user context injected into the strategy prompt
_USER_CONTEXT_MAX_CHARS = 500

# Maximum characters stored as strategy (truncation guard)
_STRATEGY_MAX_CHARS = 500

_SYSTEM_PROMPT = (
    "Produce a 1-3 sentence strategy for helping the user with this goal. "
    "Be specific and actionable. Focus on HOW to help, not WHAT the goal is. "
    "Consider the user's context and evidence. "
    "Output ONLY the strategy sentences, nothing else."
)


def generate_strategy(goal: dict) -> Optional[str]:
    """
    Main entry: generate and persist a strategy for a goal that just became actionable.

    Args:
        goal: Dict with at least keys: id, description, type.

    Returns:
        The strategy string on success, None on failure (goal still works fine).
    """
    goal_id = goal.get('id', 'unknown')
    description = goal.get('description', '')
    goal_type = goal.get('type', 'emergent')

    logger.info(
        f"{LOG_PREFIX} Generating strategy for goal {goal_id[:8]}: "
        f"'{description[:60]}' (type={goal_type})"
    )

    try:
        evidence_context = _gather_evidence_context(goal_id)
        user_context = _get_user_context()
        strategy = _call_strategy_llm(description, goal_type, evidence_context, user_context)

        if not strategy:
            logger.warning(f"{LOG_PREFIX} LLM returned empty strategy for goal {goal_id[:8]}")
            return None

        _store_strategy(goal_id, strategy)

        logger.info(
            f"{LOG_PREFIX} Strategy stored for goal {goal_id[:8]}: "
            f"'{strategy[:80]}'"
        )
        return strategy

    except Exception as e:
        logger.warning(
            f"{LOG_PREFIX} Strategy generation failed for goal {goal_id[:8]} "
            f"(goal still works): {e}"
        )
        return None


def _gather_evidence_context(goal_id: str) -> str:
    """
    Fetch recent evidence rows for the goal and format as a context string.

    Args:
        goal_id: Goal UUID.

    Returns:
        Formatted string listing up to 5 recent evidence items.
    """
    try:
        from services.database_service import get_lightweight_db_service
        db = get_lightweight_db_service()

        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT signal_type, content, source, strength
                FROM goal_evidence
                WHERE goal_id = ?
                ORDER BY created_at DESC
                LIMIT 5
            """, (goal_id,))
            rows = cursor.fetchall()
            cursor.close()

        if not rows:
            return "No evidence recorded yet."

        lines = []
        for signal_type, content, source, strength in rows:
            source_part = f" (from {source})" if source else ""
            lines.append(
                f"- [{signal_type}]{source_part} {content} (strength={strength:.2f})"
            )

        return "\n".join(lines)

    except Exception as e:
        logger.debug(f"{LOG_PREFIX} Evidence gather failed: {e}")
        return "Evidence unavailable."


def _get_user_context() -> str:
    """
    Extract Values & Goals and Active Threads sections from the autobiography narrative.

    Returns:
        Truncated autobiography excerpt (~500 chars) or empty string on failure.
    """
    try:
        from services.database_service import get_lightweight_db_service
        db = get_lightweight_db_service()

        from services.autobiography_service import AutobiographyService
        auto_service = AutobiographyService(db)
        narrative_data = auto_service.get_current_narrative()

        if not narrative_data:
            return ""

        narrative = narrative_data.get('narrative', '')
        if not narrative:
            return ""

        # Extract relevant sections
        relevant_sections = []
        target_headers = ('## values', '## goals', '## active threads', '## focus')

        lines = narrative.splitlines()
        in_section = False

        for line in lines:
            line_lower = line.lower().strip()
            if line_lower.startswith('## '):
                in_section = any(line_lower.startswith(h) for h in target_headers)
            if in_section:
                relevant_sections.append(line)

        excerpt = "\n".join(relevant_sections).strip()

        # Truncate to keep the LLM prompt lightweight
        if len(excerpt) > _USER_CONTEXT_MAX_CHARS:
            excerpt = excerpt[:_USER_CONTEXT_MAX_CHARS] + "..."

        return excerpt

    except Exception as e:
        logger.debug(f"{LOG_PREFIX} User context fetch failed: {e}")
        return ""


def _call_strategy_llm(
    description: str,
    goal_type: str,
    evidence_context: str,
    user_context: str,
) -> Optional[str]:
    """
    Make a lightweight LLM call to generate a 1-3 sentence strategy.

    Args:
        description: Goal description text.
        goal_type: One of stated, inferred, emergent, developmental.
        evidence_context: Formatted evidence string from _gather_evidence_context.
        user_context: Autobiography excerpt from _get_user_context.

    Returns:
        Strategy text (truncated to 500 chars) or None on failure.
    """
    try:
        from services.background_llm_queue import create_background_llm_proxy
        proxy = create_background_llm_proxy("goal-strategy")

        user_message_parts = [
            f"Goal: {description}",
            f"Goal type: {goal_type}",
            "",
            "Evidence:",
            evidence_context,
        ]

        if user_context:
            user_message_parts += [
                "",
                "User context (from autobiography):",
                user_context,
            ]

        user_message = "\n".join(user_message_parts)

        response = proxy.send_message(_SYSTEM_PROMPT, user_message)

        if not response:
            return None

        strategy = response.text.strip()
        if len(strategy) > _STRATEGY_MAX_CHARS:
            strategy = strategy[:_STRATEGY_MAX_CHARS]

        return strategy if strategy else None

    except Exception as e:
        logger.warning(f"{LOG_PREFIX} LLM call failed: {e}")
        return None


def _store_strategy(goal_id: str, strategy: str) -> None:
    """
    Persist the generated strategy to the goals table.

    Args:
        goal_id: Goal UUID.
        strategy: Strategy text to store.
    """
    from services.database_service import get_lightweight_db_service
    from services.time_utils import utc_now

    db = get_lightweight_db_service()
    now = utc_now().isoformat()

    with db.connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE goals SET strategy = ?, updated_at = ? WHERE id = ?",
            (strategy, now, goal_id),
        )
        cursor.close()
