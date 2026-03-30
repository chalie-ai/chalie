## `goals` — Goal Management
Create, view, confirm, complete, dismiss, mute, adjust, or narrate tracked goals.

Parameters:
- `action` (required): `"create"`, `"list"`, `"view"`, `"confirm"`, `"complete"`, `"dismiss"`, `"adjust"`, `"mute"`, `"unmute"`, `"narrate"`, `"cluster_confirm"`, `"cluster_dismiss"`
  - `create`: Set a new goal from user's stated intention.
- `goal_id` (required for view/confirm/complete/dismiss/adjust/mute/unmute): Goal ID
- `urgency` (optional, adjust only): New urgency value 0.0-1.0
- `timescale` (optional, create/adjust): `"immediate"`, `"short_term"` (default for create), `"medium_term"`, `"long_term"`
- `action`: `"create"` — Create a stated goal from the user's explicit intention.
  - `description` (required): What the user wants to achieve
  - `timescale` (optional): `"immediate"`, `"short_term"` (default), `"medium_term"`, `"long_term"`
- `action`: `"narrate"` — No additional parameters needed. Generates a narrative synthesis of how goals have evolved.
- `action`: `"cluster_confirm"` — Confirm a detected cluster of related goals. Creates a parent goal and links the children.
  - `goal_ids` (required): List of goal IDs in the cluster
  - `description` (required): Description for the consolidated parent goal
- `action`: `"cluster_dismiss"` — Dismiss a suggested cluster grouping for 30 days.
  - `goal_ids` (required): List of goal IDs in the cluster

Use when:
- User explicitly states a goal ("I want to...", "my goal is...", "I need to...") — use `create`
- User asks "what are my goals?", "what am I working on?"
- User says "I finished X" or "I don't care about X anymore"
- User confirms or validates an inferred goal ("yes, that's important to me")
- User wants to adjust priority or timeline of a goal
- User says "stop reminding me about X" or "I'll handle X myself" — use `mute`
- User says "start tracking X again" after silencing it — use `unmute`
- User asks "tell me about my goals", "how have things evolved?", "what's the story?"
- User wants to understand the big picture of what they've been working on
- System detects a cluster of related goals and suggests consolidation
- User wants to group related goals under a parent goal

Notes:
- `mute` silences proactive actions for a goal without removing it; the goal stays
  in context so Chalie still knows about it. Use when the user wants to handle
  something themselves without proactive interruptions.
- `unmute` re-enables proactive triggering for a previously muted goal.
- `narrate` synthesizes a story of goal evolution — how goals formed, evolved, and
  connect to each other. Use when the user wants the big picture rather than a list.

Example invocations:
```json
{"type": "goals", "action": "create", "description": "Run a half marathon", "timescale": "medium_term"}
{"type": "goals", "action": "list"}
{"type": "goals", "action": "view", "goal_id": "abc123"}
{"type": "goals", "action": "complete", "goal_id": "abc123"}
{"type": "goals", "action": "adjust", "goal_id": "abc123", "urgency": 0.9, "timescale": "immediate"}
{"type": "goals", "action": "mute", "goal_id": "abc123"}
{"type": "goals", "action": "unmute", "goal_id": "abc123"}
{"type": "goals", "action": "narrate"}
{"type": "goals", "action": "cluster_confirm", "goal_ids": ["abc", "def", "ghi"], "description": "Master Mediterranean cuisine"}
{"type": "goals", "action": "cluster_dismiss", "goal_ids": ["abc", "def", "ghi"]}
```
