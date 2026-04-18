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
import re
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

# Fallback token ceiling used only when context_budget is unknown (0 or None).
# When context_budget is known, the fraction-based threshold is used directly.
_FALLBACK_MAX_TOKENS = 36_000

_COMPACTION_PROMPT = """Compress conversation into dense summary. No articles, no filler, no conversation flow.

Format:
- Decisions: [decision] — [reasoning if non-obvious]
- Facts: [name/date/number/specific]
- Preferences: [preference expressed]
- Tool findings: [key result] — [source tool if relevant]
- Action items: [item] — [status]
- Unresolved: [question/blocker]

Rules:
- Bullet points only. No prose paragraphs.
- Drop: greetings, confirmations, discarded options, raw tool output
- Keep: every fact, decision, preference, number, URL, code snippet, file path
- Fragments over sentences. Dense over readable."""


def _strip_entry(content: str, role: str) -> str:
    """Strip linguistic fluff from a transcript entry before compaction.

    Pure function — no side effects. Protects code blocks, inline code, URLs,
    and file paths from any stripping.
    """
    if role == 'internal':
        return content

    if role == 'tool':
        # Collapse runs of 3+ newlines to 2, strip HTML tags
        content = re.sub(r'<[^>]+>', '', content)
        content = re.sub(r'\n{3,}', '\n\n', content)
        return content

    # Unknown roles — return content unmodified
    if role not in ('user', 'assistant'):
        return content

    # --- user / assistant stripping ---

    # Sanitise null bytes that would collide with placeholder sentinels
    content = content.replace('\x00', '')

    # Step 1: extract protected regions (code blocks, inline code, URLs, paths)
    # Replace them with placeholders so subsequent regexes don't touch them.
    placeholders = []

    def protect(m):
        idx = len(placeholders)
        placeholders.append(m.group(0))
        return f'\x00PROTECT{idx}\x00'

    # Fenced code blocks (``` ... ```)
    content = re.sub(r'```[\s\S]*?```', protect, content)
    # Inline code (`...`)
    content = re.sub(r'`[^`]+`', protect, content)
    # URLs
    content = re.sub(r'https?://\S+', protect, content)
    # File paths: starts with /, ./, ../, or contains a / with no surrounding spaces
    # Match sequences that look like paths: optional leading ./ ../ or / then word chars/slashes/dots
    content = re.sub(r'(?<!\w)(?:\.\./|\.\/|/)[\w./\-]+', protect, content)

    # Step 2: strip filler phrases at start of sentence (after . or start of string)
    # Covers: "Sure thing, ", "Sure, ", "I'd be happy to ", "Let me ", "I'll ",
    # "Certainly, ", "Of course, ", "Great, ", "Alright, ", "Ok, ", "Got it, ",
    # "Thanks, ", "Thank you, ", "Absolutely, "
    filler_starts = (
        r"Sure thing,?\s+"
        r"|Sure,?\s+"
        r"|I'd be happy to\s+"
        r"|Let me\s+"
        r"|I'll\s+"
        r"|Certainly,?\s+"
        r"|Of course,?\s+"
        r"|Great,?\s+"
        r"|Alright,?\s+"
        r"|Ok,?\s+"
        r"|Got it,?\s+"
        r"|Thanks,?\s+"
        r"|Thank you,?\s+"
        r"|Absolutely,?\s+"
    )
    content = re.sub(
        r'(?:(?<=\.)\s+|(?:^))(?:' + filler_starts + r')',
        lambda m: re.match(r'^(\s*)', m.group(0)).group(1),
        content,
        flags=re.MULTILINE | re.IGNORECASE,
    )

    # Step 3: strip hedge phrases mid-sentence (case-insensitive)
    hedge_patterns = [
        r'\bI think[^\S\n]+',
        r'\bIt seems like[^\S\n]+',
        r'\bIt looks like[^\S\n]+',
        r'\bbasically[^\S\n]+',
        r'\bessentially[^\S\n]+',
        r'\bjust[^\S\n]+',
        r'\breally[^\S\n]+',
        r'\bactually[^\S\n]+',
        r'\bprobably[^\S\n]+',
    ]
    for pattern in hedge_patterns:
        content = re.sub(pattern, ' ', content, flags=re.IGNORECASE)

    # Step 4: strip articles (a/an/the) — natural language only, not in protected regions
    # Only strip when surrounded by word boundaries and not adjacent to a placeholder marker
    content = re.sub(r'(?<!\x00)\b(a|an|the)\s+(?!\x00)', ' ', content, flags=re.IGNORECASE)

    # Step 5: restore protected regions
    def restore(m):
        return placeholders[int(m.group(1))]

    content = re.sub(r'\x00PROTECT(\d+)\x00', restore, content)

    # Step 6: collapse multiple spaces and strip trailing whitespace per line
    content = re.sub(r'[^\S\n]+', ' ', content)
    content = '\n'.join(line.strip() for line in content.split('\n'))
    content = re.sub(r'\n{3,}', '\n\n', content)

    return content.strip()


def check_and_compact(channel: str, context_budget: int, _context=None) -> bool:
    """Check if compaction is needed for a channel and run it if so.

    Args:
        channel: The conversation channel to check.
        context_budget: Maximum token budget for the context window.

    Returns:
        True if compaction was performed, False otherwise.
    """
    if not channel:
        return False

    # Get current compaction state
    compaction = get_compaction(channel, _context=_context)
    compacted_tokens = compaction['token_count'] if compaction else 0
    watermark = compaction['compacted_up_to_id'] if compaction else 0

    # Get transcript entries since watermark
    entries = get_entries_since(channel, watermark)
    if len(entries) < _MIN_ENTRIES_TO_COMPACT:
        return False

    # Estimate tokens in new entries
    entries_text = '\n'.join(e.get('content', '') for e in entries)
    entries_tokens = estimate_tokens(entries_text)

    total = compacted_tokens + entries_tokens
    if context_budget:
        threshold = int(context_budget * _TRIGGER_FRACTION)
    else:
        threshold = _FALLBACK_MAX_TOKENS

    if total <= threshold:
        logger.debug(
            f"{LOG_PREFIX} {channel}: {total} tokens "
            f"(compacted={compacted_tokens} + new={entries_tokens}) "
            f"<= threshold {threshold} — no compaction needed"
        )
        return False

    logger.info(
        f"{LOG_PREFIX} {channel}: {total} tokens exceeds threshold {threshold} "
        f"({len(entries)} entries since watermark {watermark}) — compacting"
    )

    previous_text = compaction['compacted_text'] if compaction else ''
    return _run_compaction(channel, previous_text, entries, _context=_context)


def get_compaction(channel: str, _context=None) -> Optional[Dict]:
    """Retrieve the stored compaction for a channel.

    Returns dict with: compacted_text, compacted_up_to_id, token_count, updated_at, overflow_content.
    Returns None if no compaction exists.
    """
    try:
        from services.database_service import get_shared_db_service
        db = get_shared_db_service()

        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT compacted_text, compacted_up_to_id, token_count, updated_at, overflow_content
                FROM compactions
                WHERE channel = ?
                """,
                (channel,),
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
            'overflow_content': row[4],
        }

    except Exception as e:
        logger.warning(f"{LOG_PREFIX} Failed to get compaction for {channel}: {e}")
        return None


def get_entries_since(channel: str, watermark: int = 0, limit: int = 500) -> List[Dict]:
    """Get transcript entries since a compaction watermark.

    Returns entries in chronological order (oldest first).
    """
    from services import transcript_service
    return transcript_service.get_recent(channel, limit=limit, since_id=watermark)


def _run_compaction(channel: str, previous_text: str, entries: List[Dict], _context=None) -> bool:
    """Execute the compaction LLM call and store the result.

    Args:
        channel: The conversation channel.
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
        try:
            content = _strip_entry(entry.get('content', ''), role)
        except Exception:
            logger.warning(f"{LOG_PREFIX} Strip failed for entry {entry.get('id')}, using raw content")
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
            logger.warning(f"{LOG_PREFIX} LLM returned empty compaction for {channel}")
            return False

    except Exception as e:
        logger.error(f"{LOG_PREFIX} Compaction LLM call failed for {channel}: {e}")
        if _context is not None:
            _context.record_failure('compaction_llm', e)
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
                INSERT INTO compactions (channel, compacted_text, compacted_up_to_id, token_count, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(channel) DO UPDATE SET
                    compacted_text = excluded.compacted_text,
                    compacted_up_to_id = excluded.compacted_up_to_id,
                    token_count = excluded.token_count,
                    updated_at = excluded.updated_at
                """,
                (channel, compacted_text, watermark, token_count, utc_now().isoformat()),
            )
            cursor.close()

        logger.info(
            f"{LOG_PREFIX} Compacted {channel}: "
            f"{len(entries)} entries → {token_count} tokens, watermark={watermark}"
        )
        return True

    except Exception as e:
        logger.error(f"{LOG_PREFIX} Failed to store compaction for {channel}: {e}")
        return False
