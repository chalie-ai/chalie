## `schedule` — Reminders & Scheduled Prompts
Create, list, or cancel scheduled notifications and prompts stored in Chalie's own memory.

Parameters:
- `action` (required): `"create"`, `"list"`, or `"cancel"`
- `message` (required for create; optional for cancel): What to remind or prompt (max 1000 chars). For cancel, used as a fuzzy content match when `item_id` is unknown.
- `due_at` (required for create): ISO 8601 with timezone (e.g. `"2026-02-21T09:00:00+01:00"`)
- `item_type` (optional, create): `"notification"` (default) or `"prompt"`
  - `"notification"` — text is surfaced directly as a timed reminder (the message appears as-is)
  - `"prompt"` — text is fed to Chalie as a conversational turn at the scheduled time (Chalie thinks about it and responds)
- `recurrence` (optional, create): `"daily"`, `"weekly"`, `"monthly"`, `"weekdays"`, `"hourly"`, or `"interval:N"` (every N minutes, 1–1440) — omit for one-time
- `window_start` / `window_end` (optional, create): HH:MM strings for hourly window (e.g. `"09:00"` / `"17:00"`)
- `item_id` (optional for cancel): Exact ID returned at create time. Prefer this when known.
- `time_range` (optional, list): `"today"`, `"tomorrow"`, `"this_week"`, `"next_hour"`, `"soon"` (next 6h), or `"all"` (default) — filter list results by time window

Use when: User asks to be reminded of something, schedule a recurring check, or manage reminders.
For time-scoped queries ("what's on today?", "what's coming up?"), always set `time_range` to the appropriate value.
Always normalise natural time expressions to ISO 8601 before calling create. `due_at` must be in the future.
For "every hour between 09:00 and 17:00": use `recurrence: "hourly"`, `window_start: "09:00"`, `window_end: "17:00"`, `due_at` set to today's window_start.
For "every 30 minutes" / "every X minutes": use `recurrence: "interval:30"` (replace 30 with the desired number of minutes, 1–1440). No window support for interval.

**Choosing item_type**: Use `"notification"` for simple reminders ("remind me to take medicine at 9am", "alert me when it's time to leave"). Use `"prompt"` when the user wants Chalie to think about or act on something at a scheduled time ("check on my progress every morning", "summarize what I should focus on each week", "remind me to reflect on my goals").

**IMPORTANT — cancel by content (no ID needed)**: When the user removes/deletes/cancels a reminder by describing its content ("remove the order food reminder", "cancel my dentist reminder", "I already ordered food, remove it"), use `schedule.cancel` with `message` set to the key words from the reminder text. Do NOT use `list.remove` for scheduler items — they are not on a list, they are scheduled items.
