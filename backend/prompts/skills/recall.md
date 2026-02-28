## `recall` — Unified Memory Retrieval
Search across ALL memory layers in one call.

Parameters:
- `query` (required): Search text
- `layers` (optional): Target specific layers — `["working_memory", "gists", "facts", "episodes", "concepts", "user_traits"]` or omit for all
- `limit` (optional): Max results per layer (default 3)

Use when: You need to find what the system knows about something.

**Self-knowledge queries**: For "what do you know about me?" style questions, use `layers=["user_traits"]` with query `"user profile"`. Returns all stored traits organized by category with confidence labels and source metadata (explicit vs. inferred), so you can phrase them naturally rather than declaratively. Broad queries (capped at 15 traits) include a status message if more are available.
