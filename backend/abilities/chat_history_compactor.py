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
from services.processor_config import ProcessorConfig

logger = logging.getLogger(__name__)


class ChatHistoryCompactionConfig(ProcessorConfig):
    """Single-pass chat-history compaction. No tools, no transcript writes of its
    own (the ability writes the durable watermark row explicitly). Thinking is
    forced high so the model reasons hard about what continuity to preserve."""

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
        from services.system_message_prompt import ChatHistoryCompactionSystemPrompt  # noqa: PLC0415
        return ChatHistoryCompactionSystemPrompt().get_prompt()


class ChatHistoryCompactor(Ability):
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

    def run(self, params: dict) -> dict:
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
            return {"status": "success", "result": ""}

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
        return {"status": "success", "result": "Chat history compacted."}

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
        """
        from services.llm_service import estimate_tokens  # noqa: PLC0415
        from services.system_message_prompt import ChatHistoryCompactionSystemPrompt  # noqa: PLC0415

        system = ChatHistoryCompactionSystemPrompt().get_prompt()
        provider = parent.providers.selected_provider()
        window = parent.providers.get_context_limit()
        cap = window - max(int(0.10 * window), 8000) if window else 0
        total = len(parent._previous_rows())

        drop = 0
        while True:
            prev = parent.get_previous_messages(drop_oldest=drop)
            if not prev.strip():
                return None
            combined = prev if not prior else f"## Previous Summary\n\n{prior}\n\n## New Turns\n\n{prev}"
            if cap <= 0 or drop >= total - 1:
                return combined   # cannot shrink further (no window, or one row left)
            body = provider.build_request_body(system, [{"role": "user", "content": combined}], [])
            if estimate_tokens(body) <= cap:
                return combined
            drop += 1
