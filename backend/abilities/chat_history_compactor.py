"""ChatHistoryCompactor — internal chat-history compaction, fired by dispatch.

NEVER discoverable and NEVER in any always_available list: the model never sees
`chat_history_compactor` in find_tools or its toolbox. The orchestrator dispatches
it programmatically when a compaction limit is reached (``_dispatch_compaction``),
exactly like the memory.recall turn-0 seed and the thinking pass. It fires its own
MessageProcessor.process() loop with ChatHistoryCompactionConfig, reads the parent
channel's ``get_previous_messages()`` from the bound ``mp``, and writes the model's
output VERBATIM to the transcript as ``role='compaction'`` — that row's own id is
the new watermark, so the next ``_previous_rows()`` read returns nothing through it.

No tags, no parser: whatever the model writes IS the new checkpoint. The watermark
ALWAYS advances on a non-empty history, so compaction can never silently no-op into
an infinite loop."""

import logging
from typing import ClassVar

from abilities._ability import Ability
from abilities._compaction_config import CompactionConfig
from abilities._result import ToolResult
from services.system_message_prompt import ChatHistoryCompactionSystemPrompt

logger = logging.getLogger(__name__)


class ChatHistoryCompactionConfig(CompactionConfig):
    """Single-pass chat-history compaction. No tools, no transcript writes of its
    own (the ability writes the durable watermark row explicitly). Thinking is
    forced high so the model reasons hard about what continuity to preserve."""

    SYSTEM_PROMPT_CLASS: ClassVar[type] = ChatHistoryCompactionSystemPrompt


class ChatHistoryCompactor(Ability):
    DISCOVERABLE: ClassVar[bool] = False  # internal-only compaction tool; pinned, never discovered

    def get_name(self) -> str:
        return "chat_history_compactor"

    def get_summary(self) -> str:
        return "Internal-only chat-history compaction. Never user-invocable."

    def get_examples(self) -> list[str]:
        return [
            "internal: compact conversation history into the living checkpoint",
            "internal: advance the compaction watermark for a channel",
            "internal: summarise previous messages when the context window fills",
            "internal: carry forward continuity memory across turns",
            "internal: replace older turns with a dense memory snapshot",
            "internal: fold the recent transcript tail into the checkpoint",
        ]

    def get_search_tooltip(self) -> str:
        return "internal chat-history compaction"

    _PARAMETERS: ClassVar[dict] = {"type": "object", "properties": {}}

    def get_parameters(self) -> dict:
        return self._PARAMETERS

    def run(self, params: dict) -> ToolResult:
        from services.message_processor import MessageProcessor  # noqa: PLC0415
        from services import compaction_persistence, transcript_service  # noqa: PLC0415

        parent = self.mp
        channel = parent.config.channel

        # Carry forward the prior checkpoint so continuity chains across
        # compactions instead of restarting from the recent tail each time.
        prior_row = compaction_persistence.get_compaction(channel)
        prior = (prior_row.get("compacted_text") or "").strip() if prior_row else ""

        combined = self._fit_compaction_input(parent, prior)
        if combined is None:
            # Nothing to compact — suppress_history channel or no rows past the
            # watermark. Do NOT write a watermark (there is no backlog to fold).
            return ToolResult.ok("")

        # _fit_compaction_input surfaced the count of transcript rows actually
        # folded into the checkpoint (kept after the rare drop-oldest fallback) —
        # an honest scalar already computed there, no extra provider call.
        rows_compacted = int(getattr(parent, "_compaction_kept_rows", 0))

        summary = (MessageProcessor.process(combined, ChatHistoryCompactionConfig()) or "").strip()
        if not summary:
            logger.warning(
                "[chat_history_compactor] empty summary on channel=%s; advancing watermark anyway",
                channel,
            )

        # The model's output IS the checkpoint — write it verbatim. The new
        # transcript row's own id becomes the watermark (advances unconditionally
        # on a non-empty backlog → no silent no-write, no infinite loop).
        transcript_service.write_input_row(channel, "compaction", summary)
        return ToolResult.ok("Chat history compacted.", rows_compacted=rows_compacted)

    @staticmethod
    def _fit_compaction_input(parent, prior: str):
        """Build the bare compaction request body and shrink it to fit the cap.

        Canonical design step 4.1/4.2: the compaction request includes ONLY the
        system prompt, the prior checkpoint, and ``get_previous_messages`` — no
        tools, no act-trail. That bare request almost always fits, so the drop
        loop is the EXTREMELY RARE fallback: while the {system + combined} body
        exceeds the context cap, drop the OLDEST message from get_previous_messages
        one at a time (typically 1–2) until it fits. A floor of one surviving
        message prevents dropping everything. Returns the combined text to
        summarise, or None when there is nothing left to compact.

        Uses parent.providers.measure(dto) for sizing — no raw provider object.

        Surfaces the count of transcript rows actually folded — ``total - drop``,
        the kept count after the rare drop-oldest fallback — onto
        ``parent._compaction_kept_rows`` so ``run()`` can attach an honest
        ``rows_compacted`` to the success result without recomputing or paying a
        second provider call. It is 0 when there is nothing to compact.
        """
        from services.provider_api import ProviderApiRequest, ThinkingLevel  # noqa: PLC0415

        system = ChatHistoryCompactionSystemPrompt().get_prompt()
        window = parent.providers.get_context_limit()
        cap = window - max(int(0.10 * window), 8000) if window else 0
        total = len(parent._previous_rows())

        drop = 0
        while True:
            prev = parent.get_previous_messages(drop_oldest=drop)
            if not prev.strip():
                parent._compaction_kept_rows = 0
                return None
            combined = prev if not prior else f"## Previous Summary\n\n{prior}\n\n## New Turns\n\n{prev}"
            if cap <= 0 or drop >= total - 1:
                parent._compaction_kept_rows = max(total - drop, 0)
                return combined   # cannot shrink further (no window, or one row left)
            candidate_dto = ProviderApiRequest(
                system=system,
                messages=[{"role": "user", "content": combined}],
                tools=None,
                thinking_mode=ThinkingLevel.LOW,
                cache_prefix=False,
            )
            if parent.providers.measure(candidate_dto) <= cap:
                parent._compaction_kept_rows = max(total - drop, 0)
                return combined
            drop += 1
