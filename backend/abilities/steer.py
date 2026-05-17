"""
SteerAbility — mid-loop user steering injection.

This ability is registered in AbilityRegistry so ActDispatcherService can
dispatch it by name, but it is intentionally NOT listed in any
MessageProcessor's ALWAYS_AVAILABLE or DISCOVERABLE lists.

The LLM never sees this tool. It is dispatched exclusively by the POST /chat
endpoint via ActDispatcherService when a user message arrives while a UMP ACT
loop is already running. The handler returns the steer text verbatim so the
LLM sees it as a tool result in the next iteration's context.
"""

from abilities._base import Ability


class SteerAbility(Ability):
    NAME = "steer"
    SUMMARY = (
        "User steering — injects a mid-loop user message into the active ACT iteration."
    )
    EXAMPLES = [
        "steer: please focus on the budget aspect",
        "steer: actually, ignore that — answer in French",
        "steer: stop researching and just give me a summary",
        "steer: add the item to the shopping list while you search",
        "steer: actually use the subagent for this",
        "steer: be more concise in your answer",
        "steer: also check the calendar for conflicts",
        "steer: wait, I meant next Monday not this Monday",
    ]
    INPUT_SCHEMA = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The user steering text to inject into the active ACT loop.",
            },
        },
        "required": ["text"],
    }
    TIMEOUT = 5

    def execute(self, channel: str, params: dict, telemetry: dict | None) -> str:
        """Return the steer text verbatim so the LLM sees it in the next iteration."""
        return (params.get("text") or "").strip()
