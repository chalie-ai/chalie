## `recall` — Unified Memory Retrieval
Search across ALL memory layers in one call.

Parameters:
- `query` (required): Search text
- `layers` (optional): Target specific layers — `["working_memory", "gists", "facts", "episodes", "concepts"]` or omit for all
- `limit` (optional): Max results per layer (default 3)

Use when: You need to find what the system knows about something.
