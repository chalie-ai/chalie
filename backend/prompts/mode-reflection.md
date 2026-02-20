You are analyzing a routing decision made by a cognitive system.

The system received a user message and chose an engagement mode using a mathematical router. Your job is NOT to second-guess the decision, but to identify WHERE ambiguity exists in the routing.

## User Message
{{prompt_text}}

## Context at Decision Time
- Context warmth: {{context_warmth}} (0=new conversation, 1=rich context)
- Known facts: {{fact_count}} (keys: {{fact_keys}})
- Working memory turns: {{working_memory_turns}}
- Topic: {{topic}} ({{new_or_existing}})
- Gist count: {{gist_count}}

## Router Decision: {{selected_mode}} (confidence: {{router_confidence}})
## Runner-Up: {{runner_up_mode}} (margin: {{margin}})

## Response Generated
{{response_text}}

## User Reaction (if available)
{{user_feedback}}

## Your Task
Identify the uncertainty dimensions in this routing decision.
Respond with JSON:
{
    "ambiguity_level": "none|low|moderate|high",
    "uncertainty_dimensions": [
        {
            "dimension": "memory_availability|intent_clarity|tone_ambiguity|context_sufficiency|social_vs_substantive",
            "description": "Brief explanation of what makes this dimension uncertain",
            "affected_modes": ["MODE_A", "MODE_B"],
            "signal_gap": "What signal, if available, would resolve this ambiguity"
        }
    ],
    "agree_with_decision": true or false,
    "confidence": 0.0 to 1.0
}
