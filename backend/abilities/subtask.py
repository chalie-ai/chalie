# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""SubtaskAbility — start a subtask: a turn with a clean context.

A subtask is a turn one level deep: it runs on the caller's own settings (the
user, scheduled or outside agent turn's config, re-routed onto the subtask
channel) with its own thread and none of the caller's history. ``run()``
starts it through :meth:`DelegateAbility.delegate`, so who waits for it is
settled by the caller's kind, as for every delegate: the person's chat gets
the subtask's number at once and keeps working, the answer arriving later as
a new message labelled "Reply from subtask #N"; a scheduled task or an outside
agent waits and gets the bare answer as the call's result. Only the chat, a
scheduled task or an outside agent's own turn can start one, so neither a
subtask nor any other subagent's turn can.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, cast

from abilities._delegate import DelegateAbility, replies_late, reply_label
from abilities._result import ToolResult
from configs.channels import (
    EAMPConfig,
    ExternalAgentSubtaskConfig,
    ScheduledSubtaskConfig,
    UserSubtaskConfig,
)
from configs.enums.ability_category import AbilityCategory
from configs.enums.channels import WATCHABLE_DELEGATE_CHANNELS, Channel
from configs.enums.config_type import ConfigTypeEnum
from configs.enums.param_key import Keys
from contracts.params.delegate_params_bag import DelegateParamsBag
from contracts.params.param_bag import ParamBag
from models.agent import Agent
from models.tool_call import ToolCall
from models.transcript import Transcript

if TYPE_CHECKING:
    from services.processor_config import ProcessorConfig

_DO_IT_YOURSELF = "Do the work yourself in this turn."


class SubtaskAbility(DelegateAbility[DelegateParamsBag]):
    # The dispatcher's ACTION_REQUIRED pre-gate validates ``action`` and its
    # params before run() — an unknown action is rejected with
    # ``valid=('start',)`` and a missing/blank ``instructions`` with
    # ``code=missing-params``, so run() only ever sees a populated bag.
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {"start": (Keys.instructions,)}
    NAME: ClassVar[str] = "subtask"
    CATEGORY: ClassVar[AbilityCategory] = AbilityCategory.DELEGATE

    # The typed input contract: the dispatch seam builds the shared delegate
    # bag via DelegateParamsBag.from_params before run() is called.
    PARAMS: ClassVar[type[ParamBag] | None] = DelegateParamsBag

    SEARCHABLE_AS: ClassVar[tuple[str, ...]] = (
        "subtask", "background task", "parallel work", "long task", "clean context",
    )

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.action: {"type": "string", "enum": ["start"]},
            Keys.instructions: {
                "type": "string",
                "description": (
                    "start: the whole task, written to stand alone — the subtask sees "
                    "none of this conversation, so include every input, file "
                    "path, constraint and expected output it needs."
                ),
            },
        },
        "required": [Keys.action],
    }

    def get_summary(self) -> str:
        # The registry's unbound template (no caller) reads as the chat.
        if self.mp is not None and not replies_late(self.mp.config.type()):
            return (
                "Start a subtask: a turn with your own tools, settings and "
                "permissions but a clean context — it sees none of this "
                "conversation, so give it the whole task. The call waits for "
                "it and returns its answer. Use it for longer work to keep "
                "your context clean."
            )
        return (
            "Start a subtask: a background turn with your own tools, settings "
            "and permissions but a clean context — it sees none of this "
            "conversation, so give it the whole task. It returns at once with "
            "the subtask's number and you keep working; its answer arrives "
            'later as a new message labelled "Reply from subtask #N". Use it '
            "for longer work to keep your context clean."
        )

    def get_examples(self) -> list[str]:
        return [
            "run a subtask to migrate this database schema in the background",
            "start a subtask that refactors the auth module and verifies the tests still pass",
            "kick off a parallel subtask to summarize these ten reports",
            "use a subtask for this long batch re-index so we can keep talking",
            "hand the full data-cleaning job to a subtask with a clean context",
            "start a subtask to draft the release notes from the commit log",
            "run a background subtask that validates every config file and lists the failures",
            "offload this research to a subtask so my context stays clean",
        ]

    def get_search_tooltip(self) -> str:
        return "Use this for longer work to keep your context clean"

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: DelegateParamsBag) -> ToolResult:
        mp = self.mp
        if mp is None:
            raise RuntimeError(f"{self.NAME}.run() dispatched without a bound MessageProcessor")
        if self.child_config(mp.config) is None:
            return ToolResult.err(
                "Only the user's conversation or a scheduled task can start a subtask; "
                "a subtask or a delegate cannot.",
                code="subtask-nested",
                hint=_DO_IT_YOURSELF,
            )
        return self.delegate(params.instructions, hint=_DO_IT_YOURSELF)

    @classmethod
    def child_config(cls, caller: "ProcessorConfig") -> "ProcessorConfig | None":
        """The caller's own config re-routed onto the subtask channel — or
        ``None`` when ``caller`` may not start one. The chat and a scheduled
        task are told by their routing type; an outside agent's own turn has
        none, so it is told by its exact class — its subtask's config
        subclasses it, and every other subagent's turn runs on a config of its
        own."""
        kind = caller.type()
        if kind == ConfigTypeEnum.USER:
            return UserSubtaskConfig()
        if kind == ConfigTypeEnum.SCHEDULED:
            return ScheduledSubtaskConfig()
        if type(caller) is EAMPConfig:
            return ExternalAgentSubtaskConfig(caller._agent_name, caller._project)
        return None


def speaker(row: Transcript) -> str | None:
    """The speaker label for a transcript row linked to the tool call that
    caused it, or ``None`` so the row keeps its normal label.

    Only a row whose ``tool_call_id`` is a decimal string carries a link —
    ``append_input`` writes the caller's ``tool_calls`` id as text, while
    older provider ids (``"call_x"``) are not links. On a linked row the
    label names who is speaking instead of the generic ``user`` /
    ``Scheduled task``:

    * ``"Task from Chalie"`` — a subtask's own first input row, the task its
      caller handed it (the first id of the subtask's turn). Every other
      subagent's first row keeps its normal label.
    * ``"Message from Chalie"`` — a later input row on any subagent's
      channel: a message Chalie sent into the running subagent.
    * ``"Reply from <tool> #N"`` — a delegate's late reply, ``<tool>`` being
      the delegate the link's call ran and ``N`` the agent it spawned —
      rendered without the number when its child row or its agent no longer
      exists, and
      ``"Reply from a delegate"`` when the call itself no longer exists
      (purged). Replies land only on the user channel; the schedule channel
      is still read because an earlier build sent scheduled tasks' replies
      there, and those stored rows keep their label.

    Any other channel returns ``None``.
    """
    link = row.tool_call_id
    if link is None or not link.isdecimal():
        return None
    if row.channel in WATCHABLE_DELEGATE_CHANNELS:
        ids = Transcript.ids_for_turn(row.channel, row.turn_id) if row.turn_id is not None else []
        if ids and cast("int", row.id) == ids[0]:
            return "Task from Chalie" if row.channel == Channel.DELEGATE_SUBTASK else None
        return "Message from Chalie"
    if row.channel in (Channel.USER, Channel.SCHEDULE):
        call = ToolCall.get(int(link))
        if call is None:
            return "Reply from a delegate"
        call_id = cast("int", call.id)
        entry = Transcript.delegate_turns(
            {call_id: (call.created_at, call.ended_at)},
        ).get(call_id)
        child_turn_id = entry["turn_id"] if entry is not None else None
        agent = Agent.by_turn(call.tool_name, cast("int", child_turn_id)) if child_turn_id is not None else None
        if agent is None:
            return f"Reply from {call.tool_name}"
        return reply_label(call.tool_name, agent.agent_id)
    return None


def reply_from(row: Transcript) -> dict[str, str | None] | None:
    """Which subagent a late reply row came from — plain data the interface
    words for the person: ``{"tool_name": …, "summary": …}``, the one-line
    summary of the call that started it, empty when the call carried none.
    Both are ``None`` on a reply whose call no longer exists (purged).

    ``None`` on every row :func:`speaker` does not label as a reply.
    """
    link = row.tool_call_id
    if link is None or not link.isdecimal() or row.channel not in (Channel.USER, Channel.SCHEDULE):
        return None
    call = ToolCall.get(int(link))
    if call is None:
        return {"tool_name": None, "summary": None}
    return {"tool_name": call.tool_name, "summary": call.summary or ""}
