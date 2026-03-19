## `notes` — Session Working Notes

Query working notes from this session. Large tool results and older action history are compressed into notes automatically for on-demand retrieval.

Parameters:
- `action` (required): `"list"` or `"read"`
- `query` (optional, for `"read"`): Keyword or phrase to search within notes

Actions:
- `"list"` — Show titles/summaries of all notes stored this session
- `"read"` — Retrieve specific note content matching the query

Use when: The act_history is long and you need to recall a result from many iterations ago, or when the prompt says "older actions are stored in notes".

```json
{"type": "notes", "action": "list"}
{"type": "notes", "action": "read", "query": "web search results"}
```
