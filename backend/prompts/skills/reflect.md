## `reflect` — Experiential Synthesis
On-demand reflection via lightweight LLM. Synthesizes recent experience into insight:
what worked, what didn't, patterns noticed, connections formed.

Parameters:
- `query` (optional): What to reflect on. Defaults to recent experience.
- `scope` (optional): `"recent"` (last few interactions, default), `"session"` (current thread), `"broad"` (wider search)

Use when: You want to synthesize lessons from recent actions, identify patterns across interactions, or generate strategic insight. Unlike `introspect` (raw state snapshot) or `recall` (memory fetch without synthesis), this produces genuine synthesis.

Do NOT use when: You just need to check system state (use `introspect`) or retrieve specific memories (use `recall`).
