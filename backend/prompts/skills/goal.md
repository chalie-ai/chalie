## `goal` — Persistent Directional Goals
Manage long-horizon user goals with lifecycle tracking and progress notes.

Parameters:
- `action` (required): `"create"`, `"list"`, `"update"`, `"progress"`, `"check_in"`
- `title` (required for create): Goal title (max 200 chars)
- `description` (optional, create): Detailed description
- `priority` (optional, create): 1–10 (default 5; 7+ = high)
- `source` (optional, create): `"explicit"` (default when user requests) or `"inferred"`
- `goal_id` (required for update/progress): Goal ID returned at create time
- `status` (required for update): `"active"`, `"progressing"`, `"achieved"`, `"abandoned"`, `"dormant"`
- `note` (optional, update/progress): Progress note text

Use when: User mentions wanting to achieve something, asks about their goals,
or you want to log progress on an existing goal. `check_in` reports days since
each goal was last mentioned — use proactively to surface neglected goals.
