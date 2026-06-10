"""ToolChainCompactor — internal act-trail compaction, fired by dispatch.

NEVER discoverable and NEVER in any always_available list: the model never sees
`tool_chain_compactor` in find_tools or its toolbox. The orchestrator dispatches
it programmatically alongside ChatHistoryCompactor when a compaction limit is
reached (``_dispatch_compaction``). It reads the current turn's act-trail from the
bound ``mp`` and, when there is something to compact, fires its own
MessageProcessor.process() loop with ToolChainCompactionConfig to produce a dense
handover.

The dispatch chain records the handover as the ``tool_chain_compactor`` tool_calls
row — that row is the new trail boundary: ``_from_last_compaction`` slices from it,
so pre-compacted tool calls drop out of the rendered trail without a DELETE. When
the trail is empty the ability silently returns nothing (no boundary, no LLM call)."""

from typing import ClassVar

from abilities._ability import Ability
from abilities._result import ToolResult
from services.processor_config import ProcessorConfig


class ToolChainCompactionConfig(ProcessorConfig):
    """Single-pass act-trail compaction. No tools, no transcript writes.
    Thinking is forced high so no fact a later step needs is dropped."""

    thinking_mode: ClassVar[str] = "high"

    def __init__(self) -> None:
        super().__init__(
            channel="compaction",
            role="compaction",
            policy_channel=ProcessorConfig.POLICY_CHANNEL.SUBCONSCIOUS,
            always_available=[],
            discoverable=[],
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
        return mp._raw_input

    def get_system_prompt(self, mp) -> str:
        from services.system_message_prompt import ToolChainCompactionSystemPrompt  # noqa: PLC0415
        return ToolChainCompactionSystemPrompt().get_prompt()


class ToolChainCompactor(Ability):
    def get_name(self) -> str:
        return "tool_chain_compactor"

    def get_summary(self) -> str:
        return "Internal-only act-trail (tool-chain) compaction. Never user-invocable."

    def get_examples(self) -> list[str]:
        return [
            "internal: compact the current turn's tool-call trail into a handover",
            "internal: summarise tool results before the trail overflows the window",
            "internal: collapse a long act-trail mid-turn",
            "internal: hand over what tools ran and returned this turn",
            "internal: bound the act-trail so pre-compacted calls drop out",
            "internal: preserve tool findings while shrinking the trail",
        ]

    def get_search_tooltip(self) -> str:
        return "internal act-trail compaction"

    _PARAMETERS: ClassVar[dict] = {"type": "object", "properties": {}}

    def get_parameters(self) -> dict:
        return self._PARAMETERS

    def run(self, params: dict) -> ToolResult:
        from services.message_processor import MessageProcessor  # noqa: PLC0415

        parent = self.mp
        if not parent._has_trail():
            # No non-compactor trail since the last boundary — nothing to do.
            return ToolResult.ok("")

        trail_text = parent._render_act_trail(for_compaction=True)
        if not trail_text.strip():
            return ToolResult.ok("")

        handover = (MessageProcessor.process(trail_text, ToolChainCompactionConfig()) or "").strip()
        # The dispatch chain records this result as the tool_chain_compactor row;
        # a non-empty result becomes the new trail boundary (see
        # _from_last_compaction). An empty result records a no-op row that is
        # neither a boundary nor rendered.
        return ToolResult.ok(handover)
