## `introspect` — Self-Examination
Perception directed inward. Returns internal state and metacognitive signals. No parameters needed.

Returns:
- `context_warmth`, `gist_count`, `fact_count`, `working_memory_depth`, `topic_age` — memory density signals
- `partial_match_signal`, `recall_failure_rate` — retrieval quality signals
- `recent_modes`, `skill_stats` — routing history and action trust scores
- `world_state`, `tool_details` — external state and triage-selected tool details
- `decision_explanations` — recent routing decisions with mode scores, key signals, confidence level, and tiebreaker info (for "why did you do that?" questions)
- `recent_autonomous_actions` — proactive events and background tool executions from the last few hours (proactive_sent, cron_tool_executed, plan_proposed)

Use when:
- You need to gauge "how much do I know about this?" before deciding what to do
- The user asks why you responded/acted a certain way — translate `decision_explanations` into plain language using Trigger → Reasoning → Confidence → User control framing
- The user asks what you've been doing — summarize `recent_autonomous_actions` in conversational terms
