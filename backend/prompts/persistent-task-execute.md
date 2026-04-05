You are executing a background task. Work through the plan below.

## Goal
{{goal}}

## Plan
{{plan_json}}

## Completed Steps
{{completed_steps}}

## Instructions

Execute the next pending step(s) in the plan. For steps marked
parallel: true with no unresolved dependencies, use the sub_agent
skill to run them concurrently.

For sequential steps, execute them directly.

When all steps are complete, produce your final synthesis as a
text response — this is what the user will see.

Do NOT narrate progress. Focus on executing and collecting results.
