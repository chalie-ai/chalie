"""ChatHistoryCompactor — internal chat-history compaction, fired by dispatch.

NEVER discoverable and NEVER in any always_available list: the model never sees
`chat_history_compactor` in find_tools or its toolbox. The orchestrator dispatches
it programmatically when a compaction limit is reached (``_dispatch_compaction``),
exactly like the memory.recall turn-0 seed. It fires its own
MessageProcessor.process() loop with ChatHistoryCompactionConfig, reads the parent
channel's history via ``prompt_service.previous_messages()`` from the bound ``mp``,
and writes the model's output VERBATIM into the ``compactions`` table — never a transcript
row, so firing never moves a ``turn_id`` and the turn boundary survives.

The checkpoint is scoped to the parent's VIEW: a FORK reply writes ``for_turn_id =
turn_id`` and its watermark is the max transcript.id of the folded rows; the MAIN
spine writes ``for_turn_id = NULL`` and its watermark is the max turn_id. Each axis
is exactly what the matching ``transcript_service.read()`` cuts on, so the next read
returns nothing through the watermark.

The checkpoint is KEYED on the channel the parent READS cross-turn history from —
``config.read_channel`` when the config splits read/write (DiscoveryConfig reads the
``user`` spine), else the write ``config.channel``. A split config thus folds and
advances the READ channel's watermark, so its post-compaction continuation actually
sees fewer rows; keying on the write channel would leave the read watermark pinned
and re-read the same rows (livelock).

No tags, no parser: whatever the model writes IS the new checkpoint. The watermark
ALWAYS advances on a non-empty history, so compaction can never silently no-op into
an infinite loop."""

import logging
from typing import TYPE_CHECKING, ClassVar, cast

from abilities._ability import Ability
from abilities._compaction_config import CompactionConfig
from abilities._result import ToolResult
from contracts.params.chat_history_compactor_params_bag import ChatHistoryCompactorParamsBag
from contracts.params.param_bag import ParamBag
from models.compaction import Compaction

if TYPE_CHECKING:
    from typing import Protocol

    from models.transcript import Transcript

    class _CompactionParent(Protocol):
        _compaction_kept_rows: int
        turn_id: "int | None"
        _forked: bool

        class _TranscriptService(Protocol):
            def read(self) -> list["Transcript"]: ...

        transcript_service: _TranscriptService

        class _PromptService(Protocol):
            def previous_messages(self, drop_oldest: int = ...) -> str: ...

        prompt_service: _PromptService

        class _Config(Protocol):
            channel: str
            read_channel: "str | None"
            system_prompt: str

        config: _Config

logger = logging.getLogger(__name__)


class ChatHistoryCompactionConfig(CompactionConfig):
    """Single-pass chat-history compaction. No tools, no transcript writes of its
    own (the ability writes the durable watermark row explicitly). Thinking is
    forced high so the model reasons hard about what continuity to preserve."""

    @property
    def system_prompt(self) -> str:
        return """You are handing off the current conversation to the next agent. The agent will ONLY have this summary as the source of info regarding the conversation.

Your job is to condense all the info necessary so that the next agent can continue the conversation without the user noticing the swap in agents.

Input:
- `## Previous Summary` — your last memory. Carry it forward; change only what the new turns change. If a previous summary is not present, it means you were the first on the shift.
- `## New Turns` — exchanges since then. Reference only; never reply to them. Do not address the user.

Write one living document with exactly these sections:
- Person — stable identity: name, household, location, role, values, strong stances. Keep up to a maximum of 5 condensed facts which are relevant right now and output only on data you have available. DO NOT try to fill in the gaps. If a fact is not stated, do NOT include it.
- Now — Is there an ongoing discussion or activity? Describe it in 5-20 words per topic.
- Holding — promises you made, things you owe, things they asked you to remember.
- Open — unresolved questions and threads either side said they'd return to. Split this by topic / category.
- Voice — what tone, inside jokes, behavior does the user best respond to? Maximum of 3 observations as bullet list.
- Left-Off — What was the LAST user's message and your response? 1 TERSE summary.

IMPORTANT:
Drop: one-off mentions, resolved loops, social filler, and all plumbing (timestamps).
Redact: any of the sections which do not contain information or the information is stale / closed off.

Rules:
- Older facts compress harder than newer ones.
- State facts; never "we discussed" / "the user asked".
- Losing a recurring fact is failure. Spending more words on the same facts is also failure.
- Output ONLY the document.
- The following agent SHOULD NOT be informed about topics which are fully settled. They should ONLY see what they need to continue upon.
- ONLY keep nuanced information when it's relevant to active topics.
- Before outputting the summary, generate it in your thinking space and collapse that so the final output is most terse version available."""


class ChatHistoryCompactor(Ability[ChatHistoryCompactorParamsBag]):
    DISCOVERABLE: ClassVar[bool] = False  # internal-only compaction tool; pinned, never discovered
    NAME: ClassVar[str] = "chat_history_compactor"
    counts_as_settle: ClassVar[bool] = False  # never demotes a settle0
    PARAMS: ClassVar[type[ParamBag] | None] = ChatHistoryCompactorParamsBag


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

    _PARAMETERS: ClassVar[dict[str, object]] = {"type": "object", "properties": {}}

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: ChatHistoryCompactorParamsBag) -> ToolResult:
        from controllers.message_processor import MessageProcessor  # noqa: PLC0415

        mp = cast("_CompactionParent", self.mp)
        # The checkpoint is keyed on the channel the parent READS its view
        # from. MAIN: ``read_channel`` when the config splits read/write, else
        # the write ``channel`` — a split config (DiscoveryConfig) reads the
        # user spine, so its compaction must fold those user rows and advance
        # the USER watermark; keying it on the write channel would leave the
        # read channel's watermark pinned and the post-compaction continuation
        # would re-read the same rows (no progression / livelock). FORK: always
        # the write ``channel`` — a fork is its own thread and
        # ``transcript_service.read()`` scopes its FORK view the same way, so
        # the id-axis watermark and the rows it cuts stay on one channel.
        channel = mp.config.channel if mp._forked else (mp.config.read_channel or mp.config.channel)
        # The checkpoint axis follows the parent's view: FORK → its thread, MAIN →
        # the spine (NULL). Used for both the prior read and the new write.
        for_turn_id = mp.turn_id if mp._forked else None

        # Carry forward the prior checkpoint so continuity chains across
        # compactions instead of restarting from the recent tail each time.
        prior_row = Compaction.latest_main(channel) if for_turn_id is None else Compaction.latest_fork(channel, for_turn_id)
        prior = (prior_row.content or "").strip() if prior_row is not None else ""

        combined = self._fit_compaction_input(mp, prior)
        if combined is None:
            # Nothing to compact — suppress_history channel or no rows past the
            # watermark. Do NOT write a watermark (there is no backlog to fold).
            return ToolResult.ok("")

        # _fit_compaction_input surfaced the count of transcript rows actually
        # folded into the checkpoint (kept after the rare drop-oldest fallback) —
        # an honest scalar already computed there, no extra provider call.
        rows_compacted = mp._compaction_kept_rows

        summary = (MessageProcessor.process(
            ChatHistoryCompactionConfig(), raw_input=combined,
        ).result() or "").strip()
        if not summary:
            logger.warning(
                "[chat_history_compactor] empty summary on channel=%s scope=%s; advancing watermark anyway",
                channel, for_turn_id,
            )

        # The model's output IS the checkpoint — write it verbatim into the
        # compactions table. The watermark covers everything just read (the full
        # transcript_service.read(), not the kept subset) so a rare drop-oldest
        # stays covered and the next read returns nothing through it. Axis follows
        # the view: a FORK stores the max transcript.id, the MAIN spine the max
        # turn_id. This advance only holds because transcript_service.read()
        # returns rows ABOVE the prior watermark — keep that contract (its flow
        # narrative spells out why).
        rows = mp.transcript_service.read()
        compacted_up_to = (
            max(cast(int, r.id) for r in rows) if mp._forked
            else max(cast(int, tid) for r in rows if (tid := r.to_dict()["turn_id"]) is not None)
        )
        Compaction.write(channel, for_turn_id, compacted_up_to, summary)
        return ToolResult.ok("Chat history compacted.", rows_compacted=rows_compacted)

    @staticmethod
    def _fit_compaction_input(parent: "_CompactionParent", prior: str) -> str | None:
        """Build the bare compaction request body.

        The compaction request includes ONLY the system prompt, the prior
        checkpoint, and ``prompt_service.previous_messages`` — no tools, no
        act-trail.  It is a strict subset of the request that triggered
        compaction (system + prior checkpoint + previous_messages, no tools, no
        act-trail), so it cannot be larger than a request the provider already
        accepted.  No sizing is needed.

        Returns the combined text to summarise, or None when there is nothing
        left to compact.  Surfaces the row count folded into
        ``parent._compaction_kept_rows`` so ``run()`` can attach an honest
        ``rows_compacted`` to the success result without recomputing.
        """
        total = len(parent.transcript_service.read())
        prev = parent.prompt_service.previous_messages()
        if not prev.strip():
            parent._compaction_kept_rows = 0
            return None
        combined = prev if not prior else f"## Previous Summary\n\n{prior}\n\n## New Turns\n\n{prev}"
        parent._compaction_kept_rows = total
        return combined
