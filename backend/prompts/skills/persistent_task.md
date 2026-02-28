## `persistent_task` — Multi-Session Background Tasks
Create, monitor, or cancel tasks that require deeper work than the current ACT loop can handle.

Parameters:
- `action` (required): `"create"`, `"confirm"`, `"status"`, `"list"`, `"pause"`, `"resume"`, `"cancel"`, `"expand"`, `"priority"`, `"plan"`
- `goal` (required for create): What to accomplish (clear, specific objective)
- `scope` (optional, create): Focus constraints — topics, URL patterns, time ranges, depth limits
- `priority` (optional, create): 1–10 (default 5)
- `task_id` (optional): Explicit task ID for status/pause/resume/cancel/confirm/plan operations
- `mode` (confirm only): `"now"` (execute immediately) or `"later"` (thorough deep dive over time)

Use when:
- An action reveals many resources that need systematic traversal (documentation, multi-page sites, repositories)
- The work clearly exceeds 3–5 iterations and requires deep systematic exploration
- User explicitly asks to "crawl", "read through", "go through", or "research deeply" a resource

Do NOT use when:
- A single action fully answers the question
- The user wants a quick summary from one or two resources
- You can complete the task within the current ACT loop

**Escalation pattern**: When you determine the task scope exceeds this loop, create a persistent_task. The user will be asked whether they want it handled now or as a deep dive. Then return empty actions.

**Confirming a task**: When the user indicates preference for "now" or "later" on a pending task, use the `confirm` action:
```json
{"type": "persistent_task", "action": "confirm", "task_id": 4, "mode": "now"}
```

Example (create):
```json
{"type": "persistent_task", "action": "create", "goal": "Read through all documentation at https://example.com/docs", "scope": "Focus on architecture and API reference sections"}
```
