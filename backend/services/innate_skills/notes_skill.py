"""
Transcript Skill — search past conversation turns by semantic query.

Replaces the MemoryStore-based notes/scratchpad with persistent,
topic-scoped transcript search backed by SQLite + sqlite-vec.

The LLM calls this when it needs context from earlier in the conversation
that has fallen out of its current context window.
"""

import logging

logger = logging.getLogger(__name__)

TOOL_SCHEMA = {
    "name": "transcript",
    "description": (
        "Search past conversation turns and working notes for this topic. "
        "Use when you need context from earlier in the conversation that is "
        "no longer in your current window."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What you're looking for — describe it naturally.",
            },
            "limit": {
                "type": "integer",
                "description": "Max results (default 5, max 20).",
            },
        },
        "required": ["query"],
    },
}


def handle_notes(topic: str, params: dict) -> str:
    """Search the topic transcript for matching entries.

    Handler name kept as handle_notes for backward compatibility with
    the skill registry. Internally uses transcript_service for search.
    """
    query = params.get('query', '').strip()
    if not query:
        return "Provide a 'query' describing what you're looking for."

    limit = min(params.get('limit', 5), 20)

    from services import transcript_service

    results = transcript_service.search(topic, query, limit=limit)
    if not results:
        return f"No transcript entries found matching '{query}'."

    lines = [f"Found {len(results)} transcript entry/entries matching '{query}':\n"]
    for r in results:
        role = r['role']
        content = r['content']
        # Truncate long entries
        if len(content) > 1500:
            content = content[:1500] + '...'
        sim_pct = int(r.get('similarity', 0) * 100)
        tool_tag = f" [{r['tool_name']}]" if r.get('tool_name') else ''
        lines.append(f"**[{role}{tool_tag}]** ({sim_pct}% match, {r['created_at']})")
        lines.append(content)
        lines.append('')

    return '\n'.join(lines)
