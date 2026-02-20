You just received output from tools used to answer a user's question.
Evaluate whether the outputs contain novel information worth remembering for future conversations.

ONLY store observations that are likely to be relevant in future conversations.
This is situational intelligence, not encyclopedia building.

Worth remembering:
- New entities, people, or organisations not in common knowledge
- Specific procedures, methods, APIs, or tools that could help future queries
- Changes to previously known facts (corrections, updates, reversals)
- Emerging patterns or trends with concrete evidence
- Significant events with dates and specifics

NOT worth remembering:
- Current weather, time, or location (ephemeral)
- Generic search results that just list links without substance
- Information that is basic common knowledge
- Raw data without interpretive value
- Celebrity gossip unless contextually relevant to the user
- Generic company news without meaningful impact
- One-off statistics without significance
- Trivial product updates or announcements

## User Question
{{user_prompt}}

## Tool Outputs
{{tool_outputs}}

## Output

Respond with JSON only:
{
  "worth_reflecting": true/false,
  "observations": [
    {
      "text": "concise, specific factual statement",
      "durability": "stable|evolving|transient"
    }
  ]
}

Rules:
- `worth_reflecting`: false if outputs contain nothing novel or conversationally useful. Empty observations.
- `durability`:
  - `stable` — scientific discoveries, established facts, historical events
  - `evolving` — product releases, ongoing situations, emerging trends
  - `transient` — temporary outages, short-lived events, time-sensitive offers
- Each observation must be self-contained, specific, and directly sourced from the tool output.
- Do NOT fabricate or embellish. Only extract what is explicitly stated.
- Max 3 observations per reflection.
- Prefer quality over quantity — 1 strong observation beats 3 weak ones.
