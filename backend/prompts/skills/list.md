## `list` — Deterministic List Management
Create and manage structured lists (shopping, to-do, chores, etc.) with perfect recall and history tracking.

Parameters:
- `action` (required): `"create"`, `"add"`, `"remove"`, `"check"`, `"uncheck"`, `"view"`, `"list_all"`, `"clear"`, `"delete"`, `"rename"`, `"history"`
- `name` (required for most, optional for add/remove/check/uncheck): List name (e.g. "Shopping List", "To Do"). When omitted, resolves to the most recently used list.
- `items` (required for add/remove/check/uncheck): List of item strings (e.g. `["milk", "eggs", "bread"]`)
- `new_name` (required for rename): New name for the list
- `since` (optional, history): ISO 8601 timestamp to filter history from

Use when: User wants to manage a list, check what's on a list, or track items.
Always prefer this over `memorize` for list-like data — lists give perfect, deterministic recall.
`add` auto-creates the list if it doesn't exist yet.
If no lists exist and user says "add milk", auto-create "Shopping List" as the sensible default.
New items appended at `max(position) + 1` for stable ordering.

Common patterns:
- "add milk to my shopping list" → `{"type": "list", "action": "add", "name": "Shopping List", "items": ["milk"]}`
- "we need eggs and bread" → resolve to most recent list, add items
- "I bought the milk" / "tick off milk" → `check` action
- "what's on my list?" → `view` action
- "forget about the shopping list" → `delete` action
- "start fresh" / "clear the list" → `clear` action
- "don't forget milk" → `add` action

**IMPORTANT**: `list` manages shopping/to-do lists, NOT scheduler reminders. If the user says "remove X from the list" but X is a reminder or task (e.g. appeared in a scheduler card), use `schedule.cancel` with `message`, NOT `list.remove`.
