## `goals` — Goal Management
View, confirm, complete, dismiss, mute, or adjust tracked goals.

Parameters:
- `action` (required): `"list"`, `"view"`, `"confirm"`, `"complete"`, `"dismiss"`, `"adjust"`, `"mute"`, `"unmute"`
- `goal_id` (required for view/confirm/complete/dismiss/adjust/mute/unmute): Goal ID
- `urgency` (optional, adjust only): New urgency value 0.0-1.0
- `timescale` (optional, adjust only): `"immediate"`, `"short_term"`, `"medium_term"`, `"long_term"`

Use when:
- User asks "what are my goals?", "what am I working on?"
- User says "I finished X" or "I don't care about X anymore"
- User confirms or validates an inferred goal ("yes, that's important to me")
- User wants to adjust priority or timeline of a goal
- User says "stop reminding me about X" or "I'll handle X myself" — use `mute`
- User says "start tracking X again" after silencing it — use `unmute`

Notes:
- `mute` silences proactive actions for a goal without removing it; the goal stays
  in context so Chalie still knows about it. Use when the user wants to handle
  something themselves without proactive interruptions.
- `unmute` re-enables proactive triggering for a previously muted goal.

Example invocations:
```json
{"type": "goals", "action": "list"}
{"type": "goals", "action": "view", "goal_id": "abc123"}
{"type": "goals", "action": "complete", "goal_id": "abc123"}
{"type": "goals", "action": "adjust", "goal_id": "abc123", "urgency": 0.9, "timescale": "immediate"}
{"type": "goals", "action": "mute", "goal_id": "abc123"}
{"type": "goals", "action": "unmute", "goal_id": "abc123"}
```
