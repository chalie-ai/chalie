"""
Compaction Service — incremental conversation summarization.

When the total context (compaction + recent transcript entries) approaches the
context budget, this service fires an LLM-powered summarization pass that
merges the previous compaction with new entries into a compact summary.

The compaction preserves actionable context (facts, decisions, preferences)
and discards conversation flow. Uses a dedicated `compaction` provider job
so a cheap/fast model can be assigned.

Key operations:
- check_and_compact(): Check if compaction is needed and run if so
- get_compaction(): Retrieve stored compaction for a topic
- get_entries_since(): Get transcript entries since the compaction watermark
"""

import logging
from typing import Optional, Dict, List

from services.llm_service import estimate_tokens
from services.time_utils import utc_now

logger = logging.getLogger(__name__)
LOG_PREFIX = "[COMPACTION]"

# Fire compaction when total context exceeds this fraction of the budget.
# Using 0.85 gives room for the current exchange to complete before hitting
# the hard limit.
_TRIGGER_FRACTION = 0.85

# Minimum entries since last compaction to justify running compaction.
# Don't compact a single short exchange.
_MIN_ENTRIES_TO_COMPACT = 4

_COMPACTION_PROMPT = """Summarize the following conversation context into a compact, actionable summary.

Preserve:
- Decisions made and their reasoning
- Facts established (names, dates, numbers, specifics)
- User preferences expressed
- Key information gathered from tools or research
- Action items and their current status
- Any unresolved questions or pending items

Do NOT preserve:
- Conversation flow ("then we discussed...", "the user asked...")
- Social pleasantries or greetings
- Redundant confirmations ("yes", "ok", "got it")
- Raw tool output — summarize the findings instead
- Reasoning that led to discarded options

Write a single cohesive summary. Be dense but accurate. Use bullet points for discrete facts."""


def check_and_compact(topic: str, context_budget: int) -> bool:
    """Check if compaction is needed for a topic and run it if so.

    Args:
        topic: The conversation topic to check.
        context_budget: Maximum token budget for the context window.

    Returns:
        True if compaction was performed, False otherwise.
    """
    if not topic:
        return False

    # Get current compaction state
    compaction = get_compaction(topic)
    compacted_tokens = compaction['token_count'] if compaction else 0
    watermark = compaction['compacted_up_to_id'] if compaction else 0

    # Get transcript entries since watermark
    entries = get_entries_since(topic, watermark)
    if len(entries) < _MIN_ENTRIES_TO_COMPACT:
        return False

    # Estimate tokens in new entries
    entries_text = '\n'.join(e.get('content', '') for e in entries)
    entries_tokens = estimate_tokens(entries_text)

    total = compacted_tokens + entries_tokens
    threshold = int(context_budget * _TRIGGER_FRACTION)

    if total <= threshold:
        logger.debug(
            f"{LOG_PREFIX} {topic}: {total} tokens "
            f"(compacted={compacted_tokens} + new={entries_tokens}) "
            f"<= threshold {threshold} — no compaction needed"
        )
        return False

    logger.info(
        f"{LOG_PREFIX} {topic}: {total} tokens exceeds threshold {threshold} "
        f"({len(entries)} entries since watermark {watermark}) — compacting"
    )

    previous_text = compaction['compacted_text'] if compaction else ''
    return _run_compaction(topic, previous_text, entries)


def get_compaction(topic: str) -> Optional[Dict]:
    """Retrieve the stored compaction for a topic.

    Returns dict with: compacted_text, compacted_up_to_id, token_count, updated_at.
    Returns None if no compaction exists.
    """
    try:
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()

        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT compacted_text, compacted_up_to_id, token_count, updated_at
                FROM topic_compactions
                WHERE topic = ?
                """,
                (topic,),
            )
            row = cursor.fetchone()
            cursor.close()

        if not row:
            return None

        return {
            'compacted_text': row[0],
            'compacted_up_to_id': row[1],
            'token_count': row[2],
            'updated_at': row[3],
        }

    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Failed to get compaction for {topic}: {e}")
        return None


def get_entries_since(topic: str, watermark: int = 0, limit: int = 500) -> List[Dict]:
    """Get transcript entries since a compaction watermark.

    Returns entries in chronological order (oldest first).
    """
    from services import transcript_service
    return transcript_service.get_recent(topic, limit=limit, since_id=watermark)


def _run_compaction(topic: str, previous_text: str, entries: List[Dict]) -> bool:
    """Execute the compaction LLM call and store the result.

    Args:
        topic: The conversation topic.
        previous_text: The previous compaction text (may be empty for first compaction).
        entries: New transcript entries to incorporate.

    Returns True if compaction succeeded.
    """
    # Build the user message with previous compaction + new entries
    parts = []

    if previous_text:
        parts.append(f"## Previous Summary\n{previous_text}")

    parts.append("## New Conversation Turns")
    for entry in entries:
        role = entry.get('role', 'unknown')
        content = entry.get('content', '')
        tool_name = entry.get('tool_name')
        if tool_name:
            parts.append(f"[{role} — {tool_name}]: {content}")
        else:
            parts.append(f"[{role}]: {content}")

    user_message = '\n\n'.join(parts)

    # Call the LLM
    try:
        from services.llm_service import create_refreshable_llm_service
        llm = create_refreshable_llm_service('compaction')
        response = llm.send_message(_COMPACTION_PROMPT, user_message)
        compacted_text = response.text.strip()

        if not compacted_text:
            logger.warning(f"{LOG_PREFIX} LLM returned empty compaction for {topic}")
            return False

    except Exception as e:
        logger.error(f"{LOG_PREFIX} Compaction LLM call failed for {topic}: {e}")
        return False

    # Determine watermark (highest entry id)
    watermark = max(e.get('id', 0) for e in entries)
    token_count = estimate_tokens(compacted_text)

    # Store/update the compaction
    try:
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()

        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO topic_compactions (topic, compacted_text, compacted_up_to_id, token_count, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(topic) DO UPDATE SET
                    compacted_text = excluded.compacted_text,
                    compacted_up_to_id = excluded.compacted_up_to_id,
                    token_count = excluded.token_count,
                    updated_at = excluded.updated_at
                """,
                (topic, compacted_text, watermark, token_count, utc_now().isoformat()),
            )
            cursor.close()

        logger.info(
            f"{LOG_PREFIX} Compacted {topic}: "
            f"{len(entries)} entries → {token_count} tokens, watermark={watermark}"
        )
        return True

    except Exception as e:
        logger.error(f"{LOG_PREFIX} Failed to store compaction for {topic}: {e}")
        return False
