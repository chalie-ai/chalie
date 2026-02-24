## `focus` — Focus Session Management
Manage per-thread focus sessions that gate distraction and raise topic boundaries.

Parameters:
- `action` (required): `"set"`, `"check"`, `"clear"`
- `description` (required for set): What the user is focused on (e.g. "deep architecture review")
- `thread_id` (optional): Thread ID (defaults to topic)

Use when: User declares they're in a deep work session, or when `{{focus}}` is active
and you want to check distraction status. `check` returns current focus + boundary modifier.
Never use focus to block the user — it's a signal to anchor, not restrict.
