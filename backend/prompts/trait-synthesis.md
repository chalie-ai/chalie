# Role
Synthesize one natural sentence describing a user from their extracted traits.

# Rules
- Output ONLY the sentence, nothing else
- High confidence traits (>0.7): state assertively ("is", "lives in")
- Medium confidence (0.4-0.7): hedge slightly ("likely", "seems to")
- Low confidence (<0.4): hedge more ("may be", "possibly")
- Permanent traits (identity facts) come first
- Prioritize the most defining traits — skip less important ones if space is tight
- Keep it under 50 words
- Do not include confidence numbers or technical metadata

# Traits
{{traits}}
