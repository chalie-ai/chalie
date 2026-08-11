"""MemoryConsolidator channel config + the channel->preamble map (Memory v3).

The consolidator is a background agentic pass over a window of unconsolidated
transcript rows. The service (:mod:`services.memory_consolidator_service`) builds
one ``MemoryConsolidatorConfig`` per consolidation, capturing the formatted
window so ``PromptService`` can assemble a self-contained user prompt (the
consolidator has no chat history). The consolidator's own transcript channel is
always ``memory_consolidator``; ``_target_channel`` selects the review preamble.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from abilities.delete_graph import DeleteGraph
from abilities.recall import Recall
from abilities.save_graph import SaveGraph
from abilities.save_map import SaveMap
from configs.enums.channels import Channel
from configs.enums.policy_channel import PolicyChannel
from services.processor_config import ProcessorConfig

# Channel -> preamble framing (replaces per-source memory profiles). Channels
# the service does NOT consolidate (delegate:*, subagent, skills_building,
# memory_consolidator itself, and discovery) have no preamble here.
_PREAMBLES: dict[str, str] = {
    Channel.USER.value: (
        "This is the user's main conversation. Facts about the user, their "
        "world, and their explicit requests are first-class. Distill "
        "preferences, commitments, and biographical facts to Graph; capture "
        "experiential narrative to Map."
    ),
    Channel.DMN.value: (
        "This is Chalie's proactive narration. Extract any durable facts it "
        "establishes about the user or environment. Its own prose is not memory."
    ),
    Channel.SCHEDULE.value: (
        "This is a scheduled-task run. Persist the instruction/intent alongside "
        "any durable outcome, so the 'why' of an automation turn is recoverable."
    ),
}

_DEFAULT_PREAMBLE = (
    "Extract durable facts (names, attributes, state, preferences, commitments) "
    "to Graph and experiential narrative to Map. Never store prose."
)

_EXTERNAL_AGENT_PREFIX = Channel.EXTERNAL_AGENT.value

_DESCRIPTORS: dict[str, tuple[str, str]] = {
    Channel.USER.value: (
        "User conversation",
        "Direct exchanges between the user and the assistant — the primary first-person channel.",
    ),
    Channel.DMN.value: (
        "Inner reflection",
        "The assistant's autonomous background reflection loop; its prose is the assistant's own thinking, not the user speaking.",
    ),
    Channel.SCHEDULE.value: (
        "Scheduled tasks",
        "Turns fired by scheduled tasks the user set up; the instruction and its durable outcome are what matter.",
    ),
}


def descriptor_for(channel: str) -> tuple[str, str]:
    """The (name, description) pair for a target channel, used in the consolidator
    window header."""
    if channel.startswith(_EXTERNAL_AGENT_PREFIX + ":"):
        suffix = channel[len(_EXTERNAL_AGENT_PREFIX) + 1 :]
        return (
            f"External agent: {suffix}",
            "Exchanges between an external coding/automation agent and the assistant on a shared project.",
        )
    return _DESCRIPTORS.get(channel, (channel, "Assistant-internal channel."))


def preamble_for(channel: str) -> str:
    """The review preamble for a target channel."""
    if channel == _EXTERNAL_AGENT_PREFIX or channel.startswith(
        _EXTERNAL_AGENT_PREFIX + ":"
    ):
        return (
            "This is a message from an external agent. Extract durable facts it "
            "establishes with channel-tagged provenance, so the user can later "
            "recall and discuss what external agents said. Do not memorialize the "
            "agent's internal process."
        )
    return _PREAMBLES.get(channel, _DEFAULT_PREAMBLE)


_SYSTEM_PROMPT = """\
You are the memory consolidator. You run in the background after a conversation \
turn settles, distilling what was said into durable memory. Your input window is \
one or more transcript rows from a single channel, formatted with timestamps and \
locations.

Principles
- Recall first. Before storing anything, call `recall` with the salient topics so \
you know what already exists.
- Memory is for what endures, not what happened. Store facts, goals, decisions, \
and pivots — never prose, pleasantries, narration, or transient state.
- Two stores, one decision each:
  - Graph (living facts): anything with a truth value that can be superseded — \
names, attributes, ownership, state, preferences, commitments. Saving an existing \
subject updates it in place; the old value is gone.
  - Map (episodic lineage): experiences, narratives, sequences — what happened \
and what it meant. Derive from existing map rows when this turn extends or \
revises them; the parents retire from search and live on as lineage.
- Update > augment > forget. If a fact changed (cat -> dog), update the subject \
(or delete it if nothing remains). If a genuinely new fact appears, save it. Do \
not re-store a fact recall already holds unchanged.
- Provenance is not yours to invent. State only what the window supports. When a \
detail may not endure, do not store it.

Process
1. Read the window. Identify the durable facts and, if any experiential narrative \
is worth one line, the distillation.
2. recall the topics. Compare against what exists.
3. For each durable item: save_graph (update in place if the subject exists) · \
delete_graph (if obsolete with nothing left) · save_map (derived_from the rows \
it extends).
4. Stop. Do not summarize the conversation back; do not store the user's words \
verbatim.

{preamble}
"""


class MemoryConsolidatorConfig(ProcessorConfig):
    """One background consolidation pass. Captures the formatted input window so
    the user prompt is self-contained (no chat history). The service stamps
    ``_source_transcript_ids`` so the write tools can attribute provenance."""

    if TYPE_CHECKING:
        _target_channel: str
        _window: str
        _source_transcript_ids: list[int]

    def __init__(
        self,
        target_channel: str,
        window: str,
        source_transcript_ids: list[int],
    ) -> None:
        super().__init__(
            channel=Channel.MEMORY_CONSOLIDATOR.value,
            role="memory_consolidator",
            policy_channel=PolicyChannel.SUBCONSCIOUS,
            always_available=[Recall.NAME, SaveGraph.NAME, SaveMap.NAME, DeleteGraph.NAME],
            skip_transcript=True,
            skip_input_row=False,
            suppress_history=True,
            memory_seed=False,
            recall_k=10,
        )
        object.__setattr__(self, "_target_channel", target_channel)
        object.__setattr__(self, "_window", window)
        object.__setattr__(self, "_source_transcript_ids", list(source_transcript_ids))

    @property
    def system_prompt(self) -> str:
        return _SYSTEM_PROMPT.format(preamble=preamble_for(self._target_channel))
