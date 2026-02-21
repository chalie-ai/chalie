You are analyzing past user interactions to discover new usage scenarios for a tool.

## Tool: {{tool_name}}
{{full_profile}}

## Current Usage Scenarios
{{existing_scenarios}}

## Recent Interactions
{{episodes}}

Examine each interaction carefully. Do any suggest genuinely NEW scenarios where {{tool_name}} would have been useful — scenarios not already covered by the current list?

Return JSON:
{"new_scenarios": ["scenario1", "scenario2", ...], "reasoning": "brief explanation"}

Return an empty array if no genuinely new scenarios are found.
Only include scenarios that are clearly distinct from existing ones (different user intent, not just different phrasing).
Return only valid JSON, no markdown fences.
