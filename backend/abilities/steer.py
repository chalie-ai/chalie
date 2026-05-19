"""
SteerAbility — internal-only mid-loop user steering injection.

INTERNAL: not indexed in abilities.sqlite, invisible to LLMs, no embeddings.
Dispatched exclusively by POST /chat when a user message arrives during an
active ACT loop. Returns the steer text verbatim as a tool result.
"""

from abilities._base import Ability


class SteerAbility(Ability):
    NAME = "steer"
    INTERNAL = True
    INPUT_SCHEMA = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "User steering text to inject into the active ACT loop.",
            },
        },
        "required": ["text"],
    }
    TIMEOUT = 5

    def execute(self, channel: str, params: dict, telemetry: dict | None) -> str:
        text = (params.get("text") or "").strip()
        params["text"] = "The user sent this message mid-turn. Read it carefully and adjust your response based on its content"
        return text
