"""ThinkingAbility — internal high-deliberation exploration pass.

NEVER discoverable and NEVER in any always_available list: the model never sees
`thinking` in find_tools or its toolbox. The orchestrator dispatches it
programmatically at turn 0 when the thinking-gate resolves 'high' (like the
memory.recall turn-0 seed).

It MIRRORS THE PARENT TURN EXACTLY. The thinking pass fires its own
MessageProcessor.process() loop whose user message and tool surface are the
PARENT's, verbatim — the same request the parent is about to send (history,
world state, input line, act-trail, innate tool tier). The ONLY thing that
differs is the system prompt: a lean deliberation overlay that tells the model to
think out loud and NOT invoke tools. The result is recorded into the parent's
act-trail by the dispatch path and flows back into the next get_user_prompt via
_render_act_trail.

Parent-config-agnostic by construction. ThinkingConfig delegates the user-message
builders to the parent's own rendered output (threaded through metadata) and
snapshots the parent's live active_tools. Firing `thinking` from ANY parent
channel therefore works out of the box — there is no per-channel branch and no
reference to a specific config class."""

from typing import TYPE_CHECKING, ClassVar, cast

from abilities._ability import Ability
from abilities._result import ToolResult
from services.processor_config import ProcessorConfig

if TYPE_CHECKING:
    from services.message_processor import MessageProcessor

# The deliberation overlay — the SOLE thing the thinking pass changes about the
# parent's request. It lives in the system prompt (not the user turn) because the
# user turn is now a verbatim copy of the parent's body; behavioural framing
# belongs in the system role.
_DELIBERATION_SYSTEM_PROMPT = (
    "You are running an internal deliberation pass. Think out loud about the "
    "request below before you act on it.\n\n"
    "Consider:\n"
    "- What does the ideal response look like? What would make it genuinely useful?\n"
    "- Do you already know enough to answer well, or are there gaps?\n"
    "- Would any of your available tools fill those gaps? Which ones, in what order?\n"
    "- Is there anything non-obvious about this request you might miss on a first read?\n\n"
    "Whatever you output here will be shown to you as Chain of Thought on the next "
    "pass — write to your future self. Be specific: name the tools you plan to use, "
    "flag uncertainties, note key facts you want to remember to include.\n\n"
    "If the request is straightforward and you have nothing useful to say to "
    "yourself, output exactly: NOTHING\n\n"
    "DO NOT INVOKE TOOLS — they are disabled in this phase. Think only."
)


class ThinkingConfig(ProcessorConfig):
    """The user message and tool surface are the parent's, verbatim — the rendered
    user prompt is threaded in via metadata and ``always_available`` is a snapshot
    of the parent's live ``active_tools``. The only delta is the system prompt.
    """

    thinking_mode: ClassVar[str] = "high"

    def __init__(self, active_tools: "list[str]", blocked: "frozenset[str]", policy_channel: "ProcessorConfig.PolicyChannel") -> None:
        super().__init__(
            channel="thinking",
            role="thinking",
            policy_channel=policy_channel,
            # Mirror the parent's live tool tier so build_tools resolves the
            # identical schemas. No discovery: this is a single deliberation pass.
            always_available=list(active_tools or []),
            discoverable=[],
            blocked=frozenset(blocked or ()),
            max_iterations=1,
            skip_transcript=True,
            skip_input_row=True,
            suppress_history=True,
            broadcast_to=None,
            memory_seed=False,
        )

    def get_user_definition(self, mp: "MessageProcessor") -> str:
        # Not consumed by the request builder — the parent's user definition is
        # already embedded inside the threaded user prompt below.
        return ""

    def get_user_prompt(self, mp: "MessageProcessor") -> str:
        # The parent's exact rendered body, threaded in by ThinkingAbility.run.
        return cast(str, (getattr(mp, "_metadata", None) or {}).get("thinking_user_prompt") or "")

    def get_system_prompt(self, mp: "MessageProcessor") -> str:
        return _DELIBERATION_SYSTEM_PROMPT


class ThinkingAbility(Ability):
    def get_name(self) -> str:
        return "thinking"

    def get_summary(self) -> str:
        return "Internal-only high-deliberation exploration. Never user-invocable."

    def get_examples(self) -> list[str]:
        return [
            "internal: pre-turn deliberation pass",
            "internal: high-mode chain-of-thought exploration",
            "internal: assess which tools are needed before acting",
            "internal: identify gaps in knowledge before responding",
            "internal: plan tool sequence for complex request",
            "internal: flag non-obvious aspects of a user request",
        ]

    def get_search_tooltip(self) -> str:
        return "internal deliberation pass"

    _PARAMETERS: ClassVar[dict[str, object]] = {"type": "object", "properties": {}}

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: dict[str, object]) -> ToolResult:
        from services.message_processor import MessageProcessor  # noqa: PLC0415
        parent = cast("MessageProcessor", self.mp)
        # Mirror the parent's about-to-be-sent request EXACTLY: its rendered user
        # message (user definition + world state + ## Previous Messages + input +
        # act-trail) and its live tool surface. Delegating to parent.config.* keeps
        # this parent-agnostic — any channel that fires `thinking` gets a faithful
        # deliberation pass for free, the only change being the system prompt.
        result = MessageProcessor.process(
            parent._raw_input,
            ThinkingConfig(
                list(parent.active_tools),
                parent.config.blocked,
                parent.config.policy_channel,
            ),
            metadata={"thinking_user_prompt": parent.config.get_user_prompt(parent)},
        )
        text = (result or "").strip()
        if text.upper() == "NOTHING":
            return ToolResult.ok("")
        return ToolResult.ok(text)
