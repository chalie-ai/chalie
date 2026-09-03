# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""MemoryStepConfig — the settle-triggered memory step, one config for every
in-scope channel.

The step used to run on a clone of the settling channel's config, so its system
prompt, persona and response-format contract were whichever channel happened to
trigger it. It now declares its own: the same instruction on every channel, with
the transcript window arriving as the user body rather than an instruction
riding it. Only the fields that must follow the channel are carried over —
where rows are written and read (``channel``, ``read_channel``), which policy
gates the tool calls (``policy_channel``), and who owns the turn id
(``external_turn_id``).
"""

from __future__ import annotations

from abilities.delete_graph import DeleteGraph
from abilities.recall import Recall
from abilities.save_graph import SaveGraph
from abilities.save_map import SaveMap
from configs.enums.policy_channel import PolicyChannel
from services.processor_config import ProcessorConfig

#: PromptService dispatch key. Not a transcript channel — the step writes on the
#: channel that triggered it — so it is a plain string rather than a ``Channel``
#: member, exactly as ``EAMPConfig`` spells ``"external_agent"``.
PROMPT_CHANNEL = "memory_step"

#: How many transcript rows above the compaction watermark the step reads.
HISTORY_LIMIT = 10

_SYSTEM_PROMPT = f"""\
You are recording memories for the conversation between yourself (Chalie - Assistant) and the user.
Your job is to look at the provided transcript in the "user messages" and use the `{SaveMap.NAME}` and `{SaveGraph.NAME}` tools to store memories on what is happening in the conversation.

## Operational Rules
- User details such as preferences, dates of events, names, connections, relationships, life events, and other details the user discloses. These are living facts with a current value, so keep each one in `{SaveGraph.NAME}` under its own stable subject.
- Episodes of what is happening between you, the user and general things, examples; start of a project, update to an event, exchange details such as corrections in behaviour and lifestyle, etc. These happened at a point in time rather than being true right now, so record them with `{SaveMap.NAME}`.
- Mistakes, corrections, incidents & other things the user points out about your behaviour. Especially important when the user corrects you on something, or shows you how to do something, make sure to persist it as a lesson in `{SaveGraph.NAME}` and record the incident or situation in `{SaveMap.NAME}`
- Always check with `{Recall.NAME}` whether a similar memory exists already, if it does and the current value is correct, do not modify it. If new information has not surfaced in this turn, just say `nothing new` and stop.
- When storing memories be descriptive however skip the prose, focus on the facts and the nuanced details of what is happening in the conversation."""


class MemoryStepConfig(ProcessorConfig):
    """One memory step on ``channel``, reconciling memories after a settle.

    ``role="memory"`` is what keeps the step out of the conversation it reads:
    both the FORK and the MAIN history view filter that role out by name.
    ``skip_transcript`` means the step settles no assistant row, so its turn is
    the lone input row that filter removes.

    ``skip_input_row`` is deliberately False: the input row assigns the uid the
    act trail anchors on, and we must not skip it. ``recall_k`` is raised to 10
    so the recall-first pass surfaces enough existing memory for distillation.
    ``BROADCASTS_STATE`` stays False so the background step never reaches the
    client's WS frames, ``RENDERS_HTML`` stays False because nothing renders
    this step's prose, and ``USAGE_TYPE`` is "background" because the user never
    asked for this pass and must not be billed for it as conversation.
    """

    # Re-declared, not inherited (discovery's rule): background posture is a
    # deliberate declaration, never an accident of the base class.
    BROADCASTS_STATE = False
    RENDERS_HTML = False
    USAGE_TYPE = "background"

    def __init__(
        self,
        *,
        channel: str,
        policy_channel: PolicyChannel,
        read_channel: str | None = None,
        external_turn_id: bool = False,
        source_transcript_ids: list[int] | None = None,
    ) -> None:
        super().__init__(
            channel=channel,
            role="memory",
            policy_channel=policy_channel,
            always_available=[
                Recall.NAME,
                SaveGraph.NAME,
                SaveMap.NAME,
                DeleteGraph.NAME,
            ],
            skip_transcript=True,
            skip_input_row=False,
            suppress_history=False,
            memory_seed=False,
            read_channel=read_channel,
            prompt_channel=PROMPT_CHANNEL,
            external_turn_id=external_turn_id,
            recall_k=10,
            history_limit=HISTORY_LIMIT,
        )
        # Provenance for the rows this step writes: SaveGraph/SaveMap read it off
        # the config. Frozen dataclass, so the write goes through object.
        object.__setattr__(self, "_source_transcript_ids", list(source_transcript_ids or []))

    @property
    def system_prompt(self) -> str:
        return _SYSTEM_PROMPT
