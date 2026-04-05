Design a concrete execution plan for this goal.

## Goal
{{goal}}

## Instructions

Break this goal into 2-6 concrete steps. Each step must be
independently executable — a clear action that produces a
measurable result.

Mark steps that can run in parallel (no dependency on other steps).

Output ONLY valid JSON:

{
  "steps": [
    {
      "id": "s1",
      "action": "Search for top-rated note-taking apps in 2026",
      "depends_on": [],
      "parallel": true
    },
    {
      "id": "s2",
      "action": "Compare pricing and feature sets of top 5 candidates",
      "depends_on": ["s1"],
      "parallel": false
    }
  ]
}

Rules:
- Each step action must be a clear, specific instruction (10-40 words)
- Use depends_on to express ordering. Empty = can run immediately
- Mark parallel: true for steps that can execute concurrently
- Do NOT include meta-steps like "synthesize" or "report back"
- 2-6 steps. Fewer is better if the goal is simple
