"""ThinkingAbility — internal high-deliberation exploration pass.

NEVER discoverable and NEVER in any always_available list: the model never sees
`thinking` in find_tools or its toolbox. The orchestrator dispatches it
programmatically at turn 0 when the thinking-gate resolves 'high' (like the
memory.recall turn-0 seed). It fires its own MessageProcessor.process() loop with
ThinkingConfig, which RETAINS the parent channel's full tool surface so the model
can reason about which tools would help — but is told not to invoke them. The
result is recorded into the parent's act-trail by the dispatch path and flows back
into the next get_user_prompt via _render_act_trail. No special attribute."""

from typing import ClassVar

from abilities._ability import Ability
from services.processor_config import ProcessorConfig

_EXPLORATION_PREFIX = (
    "Think out loud about the user's request before responding.\n\n"
    "Consider:\n"
    "- What does the ideal response look like? What would make it genuinely useful?\n"
    "- Do you already know enough to answer well, or are there gaps?\n"
    "- Would any of your available tools fill those gaps? Which ones, in what order?\n"
    "- Is there anything non-obvious about this request you might miss on a first read?\n\n"
    "Whatever you output here will be shown to you as Chain of Thought on the next "
    "pass — write to your future self. Be specific: name the tools you plan to use, "
    "flag uncertainties, note key facts you want to remember to include.\n\n"
    "If the request is straightforward and you have nothing useful to say to yourself, "
    "output exactly: NOTHING\n\n"
    "DO NOT INVOKE TOOLS — they are disabled in this phase. Think only.\n\n---\n\n"
)


class ThinkingConfig(ProcessorConfig):
    """Single-pass high-deliberation exploration. Retains the parent's tool
    surface (so the catalogue is visible) but the prompt forbids invocation."""

    thinking_mode: ClassVar[str] = "high"

    def __init__(self, always_available, discoverable, policy_channel) -> None:
        super().__init__(
            channel="thinking",
            role="thinking",
            policy_channel=policy_channel,
            always_available=list(always_available or []),
            discoverable=list(discoverable or []),
            blocked=frozenset(),
            max_iterations=1,
            skip_transcript=True,
            skip_input_row=True,
            suppress_history=True,
            broadcast_to=None,
            memory_seed=False,
        )

    def get_user_definition(self, mp) -> str:
        return ""

    def get_user_prompt(self, mp) -> str:
        return _EXPLORATION_PREFIX + (mp._raw_input or "")

    def get_system_prompt(self, mp) -> str:
        from services.system_message_prompt import UnifiedSystemMessagePrompt  # noqa: PLC0415
        return UnifiedSystemMessagePrompt().get_prompt()


class ThinkingAbility(Ability):
    NAME = "thinking"
    SEARCH_TOOLTIP = "internal deliberation pass"
    SUMMARY = "Internal-only high-deliberation exploration. Never user-invocable."
    EXAMPLES: ClassVar[list] = [
        "internal: pre-turn deliberation pass",
        "internal: high-mode chain-of-thought exploration",
        "internal: assess which tools are needed before acting",
        "internal: identify gaps in knowledge before responding",
        "internal: plan tool sequence for complex request",
        "internal: flag non-obvious aspects of a user request",
    ]
    INPUT_SCHEMA: ClassVar[dict] = {"type": "object", "properties": {}}

    def run(self, params: dict) -> dict:
        from services.message_processor import MessageProcessor  # noqa: PLC0415
        parent = self.MessageProcessor
        result = MessageProcessor.process(
            parent._raw_input,
            ThinkingConfig(
                parent.config.always_available,
                parent.config.discoverable,
                parent.config.policy_channel,
            ),
        )
        text = (result or "").strip()
        if text.upper() == "NOTHING":
            return {"status": "success", "result": ""}
        return {"status": "success", "result": text}
