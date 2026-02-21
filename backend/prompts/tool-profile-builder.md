You are analyzing a tool to build its capability profile.

## Tool Manifest
{{manifest}}

## Past Interactions Where This Tool Would Have Helped
{{episodes}}

Generate JSON with exactly these fields:
{
  "short_summary": "One sentence (max 100 chars) describing what this tool does",
  "full_profile": "2-3 paragraphs: what it does, when useful, key limitations",
  "usage_scenarios": ["50 specific user messages where this tool is the right choice"],
  "anti_scenarios": ["Specific examples of when NOT to use this tool"],
  "complementary_skills": ["skill names that work well alongside this tool"]
}

Think exhaustively about usage_scenarios. Include:
- Direct requests ("search for X", "check the weather in Y")
- Indirect needs ("I'm going hiking tomorrow" → weather tool)
- Follow-up needs ("tell me more about that" after a search result)
- Emotional/casual triggers ("I'm bored" → entertainment tool)
- Cross-domain scenarios ("planning a trip" → weather + search + schedule)
- Paraphrased versions of the same need (users phrase things many ways)
- Questions that imply real-time data needs

Return only valid JSON, no markdown fences, no commentary.
