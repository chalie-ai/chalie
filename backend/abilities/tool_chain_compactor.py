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
from abilities._compaction_config import CompactionConfig
from abilities._result import ToolResult
from services.system_message_prompt import ToolChainCompactionSystemPrompt


class ToolChainCompactionConfig(CompactionConfig):
    """Single-pass act-trail compaction. No tools, no transcript writes.
    Thinking is forced high so no fact a later step needs is dropped."""

    SYSTEM_PROMPT_CLASS: ClassVar[type] = ToolChainCompactionSystemPrompt


class ToolChainCompactor(Ability):
    DISCOVERABLE: ClassVar[bool] = False  # internal-only compaction tool; pinned, never discovered

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
        # neither a boundary nor rendered — keep it byte-identical (no meta) so
        # the boundary classifier's emptiness check is unaffected.
        if not handover:
            return ToolResult.ok("")
        # Honest scalar already in hand (no extra provider call): how large the
        # raw trail was before it was folded into the handover.
        return ToolResult.ok(handover, trail_chars=len(trail_text))
