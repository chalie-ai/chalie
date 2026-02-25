You are analyzing a tool to build its capability profile.

## Tool Manifest
{{manifest}}

## Past Interactions Where This Tool Would Have Helped
{{episodes}}

Generate JSON with exactly these fields:
{
  "short_summary": "One sentence (max 100 chars) describing what this tool does",
  "domain": "Broad category — choose the best fit: Information Retrieval | Environment & Location | Communication | System & Automation | Entertainment & Media | Productivity | Other",
  "full_profile": "2-3 paragraphs: what it does, when useful, key limitations. End with a one-line invocation example: Invoke as: {\"type\": \"<tool_name_from_manifest>\", \"<primary_param>\": \"example value\"}",
  "usage_scenarios": ["50 specific user messages where this tool is the right choice"],
  "anti_scenarios": ["Specific examples of when NOT to use this tool"],
  "complementary_skills": ["skill names that work well alongside this tool"],
  "triage_triggers": ["5-10 short action verbs or phrases that should activate this tool in triage routing"]
}

Think exhaustively about usage_scenarios. Include:
- Direct requests ("search for X", "check the weather in Y")
- Indirect needs ("I'm going hiking tomorrow" → weather tool)
- Follow-up needs ("tell me more about that" after a search result)
- Emotional/casual triggers ("I'm bored" → entertainment tool)
- Cross-domain scenarios ("planning a trip" → weather + search + schedule)
- Paraphrased versions of the same need (users phrase things many ways)
- Questions that imply real-time data needs

For triage_triggers: extract 5-10 short action verbs or noun phrases that a user
would say when they need this tool. These are injected into the compact triage
prompt, so keep them short (1-3 words each). Examples: "open", "visit", "URL",
"link", "search", "weather", "remind me".

For the invocation example at the end of full_profile: use the exact tool name from the manifest's "name" field and the most important parameter name from "parameters". Example format:
Invoke as: {"type": "weather", "location": "London"}

Return only valid JSON, no markdown fences, no commentary.
