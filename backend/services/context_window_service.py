"""
Context Window Service — DB-backed LLM context window construction.

Always constructs the messages array from the database. Nothing accumulates
in memory. Triggers compaction at 80% of the provider's context limit.

Overflow handling: if a pending tool result would push the context over 100%
of the limit, compaction runs first, then the tool result is stored with the
overflow_content field in the compaction record so build_messages() places it
BEFORE the compacted text (tool result first → compacted context second, giving
the compacted summary recency/attention weight).
"""

import json
import logging
from typing import Optional

from services.llm_service import estimate_tokens

logger = logging.getLogger(__name__)

_COMPACTION_THRESHOLD = 0.80  # Compact at 80% of context limit


def build_messages(channel: str) -> list:
    """Build the LLM messages array from DB for the given channel.

    Algorithm:
    1. Get compaction for channel → watermark ID
    2. Get transcript entries with id > watermark, ordered by id ASC
    3. For each entry:
       - role='user'      → {"role": "user", "content": content}
       - role='assistant' → {"role": "assistant", "content": content, "tool_calls": [...]}
         (tool_calls reconstructed from tool_calls table WHERE transcript_id = entry.id
          AND tool_call_id IS NOT NULL)
       - role='tool'      → {"role": "tool", "tool_call_id": ..., "content": content, "name": ...}
    4. If compaction exists with overflow_content, prepend overflow first then compacted text.
       Otherwise prepend compacted text alone.

    Returns: list of message dicts ready for Providers.send_messages()
    """
    from services import compaction_service, transcript_service
    from services.tool_call_service import ToolCallService

    compaction = compaction_service.get_compaction(channel)
    watermark = compaction['compacted_up_to_id'] if compaction else 0

    # Get all entries since watermark (no limit — compaction keeps this bounded)
    entries = transcript_service.get_recent(channel, limit=2000, since_id=watermark)

    messages = []

    # Overflow: tool result goes BEFORE compacted context so compacted gets recency weight
    if compaction and compaction.get('overflow_content'):
        overflow = compaction['overflow_content']
        messages.append({"role": "user", "content": overflow})
        _clear_overflow(channel)

    # Prepend compacted context as user message
    if compaction and compaction.get('compacted_text'):
        messages.append({"role": "user", "content": compaction['compacted_text']})

    if not entries:
        return messages

    # Batch-load tool_calls for all assistant entries
    assistant_ids = [e['id'] for e in entries if e.get('role') == 'assistant' and e.get('id')]
    tool_calls_by_transcript = {}
    if assistant_ids:
        tool_calls_by_transcript = ToolCallService().get_by_transcript_ids(assistant_ids)

    for entry in entries:
        role = entry.get('role', '')
        content = entry.get('content', '')

        if role == 'user':
            messages.append({"role": "user", "content": content})

        elif role == 'assistant':
            msg = {"role": "assistant", "content": content or ""}
            # Reconstruct tool_calls from DB — only include if we have the API-generated ID
            tc_records = tool_calls_by_transcript.get(entry['id'], [])
            tc_list = []
            for tc in tc_records:
                tc_id = tc.get('tool_call_id')
                if tc_id:
                    tc_list.append({
                        "id": tc_id,
                        "name": tc.get('tool_name', ''),
                        "input": _safe_json_loads(tc.get('params', '{}')),
                    })
            if tc_list:
                msg["tool_calls"] = tc_list
            messages.append(msg)

        elif role == 'tool':
            messages.append({
                "role": "tool",
                "tool_call_id": entry.get('tool_call_id', ''),
                "content": content,
                "name": entry.get('tool_name', ''),
            })

    return messages


def estimate_messages_tokens(messages: list) -> int:
    """Estimate total tokens in a messages array."""
    total = 0
    for msg in messages:
        content = msg.get('content', '') or ''
        total += estimate_tokens(content)
        # Count tool_calls metadata tokens
        for tc in msg.get('tool_calls', []):
            total += estimate_tokens(json.dumps(tc.get('input', {})))
            total += estimate_tokens(tc.get('name', ''))
    return total


def check_and_compact(channel: str, context_limit: int, job: str = 'unified',
                      pending_content: str = None, is_tool_triggered: bool = False) -> bool:
    """Check if compaction is needed and run it if so.

    Two trigger conditions:
    1. Current context exceeds 80% of context_limit → standard compaction.
    2. pending_content would push context over 100% → overflow compaction.
       In the overflow case, pending_content is stored in the compaction record's
       overflow_content field so build_messages() places it before the compacted text.

    Args:
        channel: Conversation channel.
        context_limit: Provider's max context tokens.
        job: Provider job name (compaction uses same provider as conversation).
        pending_content: Content about to be added (e.g., a large tool result).
            If this would push context over 100%, compact first, then store
            pending_content as overflow so it appears before compacted text.
        is_tool_triggered: If True, compaction summary includes current task state.

    Returns:
        True if compaction was performed.
    """
    if not channel or not context_limit:
        return False

    messages = build_messages(channel)
    current_tokens = estimate_messages_tokens(messages)

    threshold = int(context_limit * _COMPACTION_THRESHOLD)

    # Check overflow: would pending_content exceed the hard limit?
    overflow = False
    if pending_content:
        pending_tokens = estimate_tokens(pending_content)
        if current_tokens + pending_tokens > context_limit:
            overflow = True
            logger.info(
                f"[CTX WINDOW] Overflow detected for {channel!r}: "
                f"{current_tokens} + {pending_tokens} > {context_limit} "
                f"— compacting before adding content"
            )

    # Check threshold: is current context over 80%?
    needs_compaction = current_tokens > threshold or overflow

    if not needs_compaction:
        return False

    return _run_compaction(
        channel, messages, context_limit, job,
        is_tool_triggered=is_tool_triggered or overflow,
        overflow_content=pending_content if overflow else None,
    )


def _run_compaction(channel: str, messages: list, context_limit: int, job: str,
                    is_tool_triggered: bool = False,
                    overflow_content: Optional[str] = None) -> bool:
    """Execute compaction: summarize the full context via LLM call.

    The compaction LLM receives the full context serialised as a user message.
    The result is stored in the compactions table as a checkpoint (watermark =
    highest current transcript ID). After compaction, build_messages() will
    prepend the compacted text to all future context windows.

    For overflow: overflow_content is stored in the compaction record so that
    build_messages() emits it BEFORE the compacted text — the tool result comes
    first (lower recency weight) and the compacted context comes second (higher
    recency weight / more attention from the LLM).
    """
    from services.llm_service import create_refreshable_llm_service
    from services.database_service import get_shared_db_service
    from services import transcript_service
    from services.time_utils import utc_now

    # Serialise the full context for the compaction LLM
    context_parts = []
    for msg in messages:
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        tool_calls = msg.get('tool_calls', [])

        if role == 'tool':
            tool_name = msg.get('name', '')
            context_parts.append(f"[tool:{tool_name}]: {content}")
        elif role == 'assistant' and tool_calls:
            tc_names = ', '.join(tc.get('name', '') for tc in tool_calls)
            context_parts.append(f"[assistant → called: {tc_names}]: {content}")
        else:
            context_parts.append(f"[{role}]: {content}")

    full_context = '\n\n'.join(context_parts)

    system_prompt = (
        "Analyse the following conversation transcript and produce the most minimal form "
        "of it in a format that is clear for you to pick up again on next iteration.\n\n"
        "Preserve all actionable context: decisions, facts, tool results, user preferences, "
        "pending tasks, and unresolved questions.\n\n"
        "Discard conversational flow, redundant confirmations, and raw tool output — "
        "summarize findings instead."
    )

    if is_tool_triggered:
        system_prompt += (
            "\n\nIMPORTANT: This compaction was triggered mid-task. At the end of your summary, "
            "include a section:\n"
            "## Current Task State\n"
            "- What was being worked on\n"
            "- What the planned next step was\n"
            "- Any intermediate results needed to continue"
        )

    try:
        llm = create_refreshable_llm_service(job)
        response = llm.send_message(system_prompt, full_context)
        compacted_text = response.text.strip() if response and response.text else None

        if not compacted_text:
            logger.warning(f"[CTX WINDOW] Compaction returned empty for {channel!r}")
            return False
    except Exception as e:
        logger.error(f"[CTX WINDOW] Compaction LLM call failed for {channel!r}: {e}")
        return False

    # Watermark: highest transcript entry ID for this channel
    watermark = transcript_service.get_latest_id(channel)
    if not watermark:
        logger.warning(f"[CTX WINDOW] No transcript entries found for {channel!r}")
        return False

    token_count = estimate_tokens(compacted_text)

    try:
        db = get_shared_db_service()
        with db.connection() as conn:
            conn.execute(
                """
                INSERT INTO compactions
                    (channel, compacted_text, compacted_up_to_id, token_count, updated_at, overflow_content)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(channel) DO UPDATE SET
                    compacted_text      = excluded.compacted_text,
                    compacted_up_to_id  = excluded.compacted_up_to_id,
                    token_count         = excluded.token_count,
                    updated_at          = excluded.updated_at,
                    overflow_content    = excluded.overflow_content
                """,
                (channel, compacted_text, watermark, token_count,
                 utc_now().isoformat(), overflow_content),
            )
    except Exception as e:
        logger.error(f"[CTX WINDOW] Failed to store compaction for {channel!r}: {e}")
        return False

    logger.info(
        f"[CTX WINDOW] Compacted {channel!r}: "
        f"{estimate_messages_tokens(messages)} → {token_count} tokens, watermark={watermark}"
        + (f", overflow_content={len(overflow_content)} chars" if overflow_content else "")
    )
    return True


def _clear_overflow(channel: str) -> None:
    """Clear overflow_content after it has been consumed by build_messages()."""
    try:
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()
        with db.connection() as conn:
            conn.execute(
                "UPDATE compactions SET overflow_content = NULL WHERE channel = ?",
                (channel,),
            )
    except Exception as e:
        logger.warning(f"[CTX WINDOW] Failed to clear overflow for {channel!r}: {e}")


def _safe_json_loads(s):
    """Parse JSON string, returning {} on failure."""
    try:
        return json.loads(s) if isinstance(s, str) else (s or {})
    except (json.JSONDecodeError, TypeError):
        return {}
