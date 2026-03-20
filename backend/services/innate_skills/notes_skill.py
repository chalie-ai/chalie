import json
import logging

logger = logging.getLogger(__name__)

TOOL_SCHEMA = {
    "name": "notes",
    "description": (
        "Query working notes from this session. Large tool results and older action history "
        "are compressed into notes automatically for on-demand retrieval. Use when act_history "
        "is long and you need to recall a result from many iterations ago, or when the prompt "
        "says 'older actions are stored in notes'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "read"],
                "description": "Operation: 'list' shows titles/summaries of all notes; 'read' retrieves specific note content.",
            },
            "query": {
                "type": "string",
                "description": "Keyword or phrase to search within notes. Used with 'read' action.",
            },
            "id": {
                "type": "string",
                "description": "Specific note ID to retrieve. Used with 'read' action.",
            },
        },
        "required": ["action"],
    },
}


def handle_notes(topic: str, params: dict) -> str:
    action = params.get('action', 'list')
    loop_id = params.get('loop_id', '')

    if not loop_id:
        return "No active scratchpad in this context."

    from services.memory_client import MemoryClientService
    store = MemoryClientService.create_connection()
    key = f"scratchpad:{loop_id}:entries"

    raw_entries = store.lrange(key, 0, -1)
    if not raw_entries:
        return "Notes are empty — no large results or pruned history yet."

    entries = []
    for raw in raw_entries:
        try:
            entries.append(json.loads(raw) if isinstance(raw, str) else json.loads(raw.decode()))
        except Exception:
            continue

    if action == 'list':
        lines = [f"## Working Notes ({len(entries)} entries)"]
        for e in entries:
            lines.append(
                f"- **{e.get('id', '?')}** [{e.get('source', '?')}] "
                f"iter {e.get('iteration', '?')}: {e.get('summary', '')[:120]}"
            )
        return '\n'.join(lines)

    if action == 'read':
        entry_id = params.get('id', '')
        query = params.get('query', '').lower()

        if entry_id:
            for e in entries:
                if e.get('id') == entry_id:
                    return f"## Note {entry_id}\n\n{e.get('full_content', e.get('summary', 'No content'))}"
            return f"Note {entry_id} not found."

        if query:
            matches = []
            for e in entries:
                searchable = (
                    f"{e.get('summary', '')} {e.get('query_hint', '')} {e.get('full_content', '')}"
                ).lower()
                if query in searchable:
                    matches.append(e)
            if not matches:
                return f"No notes matching '{query}'."
            lines = [f"## Notes matching '{query}' ({len(matches)} results)"]
            for e in matches[:3]:
                content = e.get('full_content', e.get('summary', ''))
                if len(content) > 2000:
                    content = content[:2000] + '...'
                lines.append(f"### {e.get('id', '?')} [{e.get('source', '?')}]\n{content}")
            return '\n'.join(lines)

        return "Provide 'id' or 'query' parameter for read action."

    return f"Unknown notes action: {action}. Use 'list' or 'read'."
