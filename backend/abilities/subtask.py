# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""SubtaskAbility — start a subtask: a background turn with a clean context.

A subtask is a background turn one level deep: it runs on the caller's own
settings (the user or scheduled turn's config, re-routed onto the subtask
channel) with its own thread and none of the caller's history, and its
answer arrives on the caller's channel as a new message labelled
"Reply from subtask #N". Unlike a delegate it is not waited for — ``run()``
opens the child through :meth:`DelegateAbility.spawn` and returns at once
with the subtask's number, so the caller keeps working. Only a user or
scheduled turn can start one, so neither a subtask nor a delegate's child can.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar, cast

from abilities._delegate import DelegateAbility
from abilities._result import ToolResult
from configs.channels import ScheduledSubtaskConfig, UserSubtaskConfig, config_for
from configs.enums.ability_category import AbilityCategory
from configs.enums.channels import Channel
from configs.enums.config_type import ConfigTypeEnum
from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag
from contracts.params.subtask_params_bag import (
    SubtaskMessageParams,
    SubtaskParamsBag,
    SubtaskStartParams,
)
from models.tool_call import ToolCall
from models.transcript import Transcript
from models.turn_execution import TurnExecution
from services.turn_execution_service import (
    STOP_REASON_PARENT_STOPPED,
    STOP_REASON_PROCESS_DEATH,
)

if TYPE_CHECKING:
    from controllers.message_processor import MessageProcessor

logger = logging.getLogger(__name__)


class SubtaskAbility(DelegateAbility[SubtaskParamsBag]):
    # Multi-action router: the dispatcher's ACTION_REQUIRED pre-gate validates
    # ``action`` and its params before run() — an unknown action is rejected
    # with ``valid=('start', 'message')`` and a missing/blank ``instructions``
    # or ``message`` with ``code=missing-params``, so run() only ever sees a
    # populated leaf.
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {
        "start": (Keys.instructions,),
        "message": (Keys.subtask, Keys.message),
    }
    NAME: ClassVar[str] = "subtask"
    CATEGORY: ClassVar[AbilityCategory] = AbilityCategory.DELEGATE

    # The typed input contract: the dispatch seam builds the bag via
    # SubtaskParamsBag.from_params before run() is called.
    PARAMS: ClassVar[type[ParamBag] | None] = SubtaskParamsBag

    SEARCHABLE_AS: ClassVar[tuple[str, ...]] = (
        "subtask", "background task", "parallel work", "long task", "clean context",
    )

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.action: {"type": "string", "enum": ["start", "message"]},
            Keys.instructions: {
                "type": "string",
                "description": (
                    "start: the whole task, written to stand alone — the subtask sees "
                    "none of this conversation, so include every input, file "
                    "path, constraint and expected output it needs."
                ),
            },
            Keys.subtask: {
                "type": "integer",
                "minimum": 1,
                "description": 'message: the subtask\'s number (the N in "Subtask #N").',
            },
            Keys.message: {
                "type": "string",
                "description": "message: what to tell the subtask.",
            },
        },
        "required": [Keys.action],
    }

    def get_summary(self) -> str:
        return (
            "Start a subtask: a background turn with your own tools, settings "
            "and permissions but a clean context — it sees none of this "
            "conversation, so give it the whole task. It returns at once with "
            "the subtask's number and you keep working; its answer arrives "
            'later as a new message labelled "Reply from subtask #N". Use it '
            "for longer work to keep your context clean. You can send it a "
            "follow-up with action 'message'."
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

    def run(self, params: SubtaskParamsBag) -> ToolResult:
        mp = self.mp
        if mp is None:
            raise RuntimeError(f"{self.NAME}.run() dispatched without a bound MessageProcessor")
        # The kind gate holds for both actions: only a user or scheduled turn
        # may start a subtask or message one, so neither a subtask nor a
        # delegate's child can.
        kind = mp.config.type()
        if kind not in (ConfigTypeEnum.USER, ConfigTypeEnum.SCHEDULED):
            return ToolResult.err(
                "Only the user's conversation or a scheduled task can start a subtask; "
                "a subtask or a delegate cannot.",
                code="subtask-nested",
                hint="Do the work yourself in this turn.",
            )
        config = ScheduledSubtaskConfig() if kind == ConfigTypeEnum.SCHEDULED else UserSubtaskConfig()
        if isinstance(params, SubtaskStartParams):
            child = self.spawn(config, params.instructions)
            return ToolResult.ok(
                f"Subtask #{child.turn_id} started. It works in the background; its answer will "
                f"arrive as a new message labelled \"Reply from subtask #{child.turn_id}\". "
                "Carry on — do not wait for it or start it again."
            )
        if isinstance(params, SubtaskMessageParams):
            n = params.subtask
            if not self._subtask_is_ours(mp, kind, n):
                return ToolResult.err(
                    f"No subtask #{n} belongs to this conversation.",
                    code="subtask-not-found",
                    hint='Use the number from "Subtask #N started".',
                )
            # Lazy: the controller imports the ability registry, which imports this module.
            from controllers.message_processor import MessageProcessor  # noqa: PLC0415

            result = MessageProcessor.send(
                config,
                params.message,
                metadata={"origin": mp.origin, "tool_call_id": self.tool_call_id},
                turn_id=n,
            )
            mp.tool_call_service.reemit(self.tool_call_id)
            if result is None:
                raise RuntimeError(f"{self.NAME}.run(): the message to subtask #{n} failed to land")
            return ToolResult.ok(
                f"Message sent to subtask #{n}. Its answer will arrive as a new message "
                f'labelled "Reply from subtask #{n}". Carry on — do not wait for it.'
            )
        raise TypeError(f"{self.NAME}.run() got an unrouted bag: {type(params).__name__}")

    @staticmethod
    def _subtask_is_ours(mp: "MessageProcessor", kind: ConfigTypeEnum | None, n: int) -> bool:
        """Whether subtask ``n`` belongs to this conversation — the ONE address
        check action "message" runs before sending: the subtask's first input
        row carries the caller's ``tool_calls`` id as decimal text, which walks
        back to the caller's row; that row must sit on this channel, and — the
        schedule channel is shared by every schedule — on it, on this turn.
        Any miss (no rows, no link, no call, no caller, wrong owner) is the
        same "not ours" answer."""
        ids = Transcript.ids_for_turn(Channel.DELEGATE_SUBTASK.value, n)
        if not ids:
            return False
        first = Transcript.get(ids[0])
        if first is None or first.tool_call_id is None or not first.tool_call_id.isdecimal():
            return False
        call = ToolCall.get(int(first.tool_call_id))
        if call is None:
            return False
        caller = Transcript.get(call.transcript_id)
        if caller is None or caller.turn_id is None:
            return False
        if caller.channel != mp.channel:
            return False
        return kind != ConfigTypeEnum.SCHEDULED or caller.turn_id == mp.turn_id


def speaker(row: Transcript) -> str | None:
    """The speaker label for a transcript row linked to the tool call that
    caused it, or ``None`` so the row keeps its normal label.

    Only a row whose ``tool_call_id`` is a decimal string carries a link —
    ``append_input`` writes the caller's ``tool_calls`` id as text, while
    older provider ids (``"call_x"``) are not links. On a linked row the
    label names who is speaking instead of the generic ``user`` /
    ``Scheduled task``:

    * ``"Task from Chalie"`` — the subtask's own first input row, the task
      its caller handed it (the first id of the subtask's turn).
    * ``"Message from Chalie"`` — a later input row on the subtask channel:
      a message Chalie sent to the working subtask.
    * ``"Reply from subtask #N"`` — a reply row landing on the user or
      schedule channel, ``N`` being the child turn the link's call spawned —
      rendered ``"Reply from subtask"`` without the number when the call or
      its child row no longer exists (purged).

    Any other channel returns ``None``: other delegates' input rows carry
    call links too and keep their normal label.
    """
    link = row.tool_call_id
    if link is None or not link.isdecimal():
        return None
    if row.channel == Channel.DELEGATE_SUBTASK:
        turn_id = row.turn_id
        if turn_id is None:
            return "Message from Chalie"
        ids = Transcript.ids_for_turn(row.channel, turn_id)
        first = ids[0] if ids else None
        return (
            "Task from Chalie"
            if first is not None and cast("int", row.id) == first
            else "Message from Chalie"
        )
    if row.channel in (Channel.USER, Channel.SCHEDULE):
        label = "Reply from subtask"
        call = ToolCall.get(int(link))
        if call is not None:
            call_id = cast("int", call.id)
            entry = Transcript.delegate_turns(
                {call_id: (call.created_at, call.ended_at)},
            ).get(call_id)
            child_turn_id = entry["turn_id"] if entry is not None else None
            if child_turn_id is not None:
                label = f"Reply from subtask #{child_turn_id}"
        return label
    return None


def is_subtask_turn(channel: str, turn_id: int) -> bool:
    """Whether ``(channel, turn_id)`` is a subtask's own turn — the only kind
    of turn whose end is reported to a caller, asked the same way by the
    end-of-turn hook and by the boot sweep, which has only the swept
    execution row to go on.

    The channel alone does not settle it: the memory step runs on the channel
    it settled on, in a turn of its own, so the subtask channel carries the
    memory step's turns too. The row that opened the turn tells them apart —
    the memory step's input row carries its ``memory`` role, a subtask's its
    caller's. A turn with no row at all counts as a subtask's, so
    :func:`reply_to_caller` fails on it loudly instead of skipping it.
    """
    if channel != Channel.DELEGATE_SUBTASK:
        return False
    opener = Transcript.filter("channel", channel).filter("turn_id", turn_id).order_by("id ASC").first()
    return opener is None or opener.role != "memory"


def reply_to_caller(turn_id: int, answer: str) -> None:
    """Send a finished subtask's outcome to the turn that started it.

    The outcome is read from the subtask's own execution row, never from a
    MessageProcessor — a cancel lands on another thread, and the boot sweep
    will later call this with no MP at all. The reply joins the caller's turn
    while it works; otherwise a user's main exchange gets a new turn, and a
    thread reply or a schedule's thread is continued — always through
    :meth:`MessageProcessor.send`, the one input function every channel uses.
    """
    # Lazy: the controller imports the ability registry, which imports this module.
    from controllers.message_processor import MessageProcessor  # noqa: PLC0415

    execution = TurnExecution.latest(Channel.DELEGATE_SUBTASK.value, turn_id)
    if execution is None or execution.ended_at is None:
        raise RuntimeError(f"subtask #{turn_id} ended without a closed turn execution")
    if execution.state == TurnExecution.COMPLETED:
        text = answer if answer.strip() else "This subtask finished without an answer."
    elif execution.state == TurnExecution.CANCELLED:
        if execution.stop_reason == STOP_REASON_PARENT_STOPPED:
            # Stopped by its caller's own stop: the caller never gets an
            # answer, so there is no one to reply to.
            logger.info("[SUBTASK] subtask %d: stopped with its caller, no reply", turn_id)
            return
        text = "This subtask was stopped before it finished."
    elif execution.state == TurnExecution.CRASHED:
        if execution.stop_reason == STOP_REASON_PROCESS_DEATH:
            text = "This subtask was cut off by a restart before it finished."
        else:
            text = f"This subtask crashed: {execution.stop_reason}"
    else:
        raise RuntimeError(f"subtask #{turn_id} ended in unexpected state {execution.state!r}")

    # The subtask's input rows carry the sender's tool_calls id (the stamp
    # DelegateAbility.spawn writes when it opens the turn) — walk back from
    # the LAST linked row to the row that asked for the call, then to the
    # caller's turn; a follow-up message re-points the reply at the turn that
    # sent it.
    rows = Transcript.by_turn(Channel.DELEGATE_SUBTASK.value, turn_id)
    link = next((row for row in reversed(rows) if row["tool_call_id"] is not None), None)
    if link is None:
        raise RuntimeError(
            f"subtask #{turn_id}: no row of its turn carries the caller's tool_call_id"
        )
    call = ToolCall.get(int(cast("str", link["tool_call_id"])))
    if call is None:
        raise RuntimeError(f"subtask #{turn_id}: its caller's tool call no longer exists")
    caller = Transcript.get(call.transcript_id)
    if caller is None or caller.turn_id is None:
        raise RuntimeError(
            f"subtask #{turn_id}: the caller's transcript row {call.transcript_id} "
            "no longer exists or carries no turn"
        )

    caller_execution = TurnExecution.latest(caller.channel, caller.turn_id)
    if caller_execution is None:
        raise RuntimeError(
            f"subtask #{turn_id}: caller {caller.channel} turn {caller.turn_id} "
            "has no turn execution to reply into"
        )
    if caller_execution.type is None:
        raise RuntimeError(
            f"subtask #{turn_id}: caller {caller.channel} turn {caller.turn_id} "
            "has an execution with no routing type to resolve a config for"
        )
    config = config_for(caller_execution.type)

    # The caller's row sits in the turn's main exchange (no settle yet, or at
    # or before its first settled assistant row); a user reply there must land
    # on the spine, never inside a running follow-up reply.
    settle0 = Transcript.settle0(caller.channel, caller.turn_id)
    opening = settle0 is None or cast("int", caller.id) <= settle0
    spine_only = opening and config.type() == ConfigTypeEnum.USER

    result = MessageProcessor.send(
        config, text, metadata={"tool_call_id": call.id}, turn_id=caller.turn_id, spine_only=spine_only,
    )
    if result is None:
        raise RuntimeError(
            f"subtask #{turn_id}: the reply to {caller.channel} turn {caller.turn_id} failed to open"
        )
    if result.turn_id == caller.turn_id and result.id == caller_execution.id:
        logger.info(
            "[SUBTASK] subtask %d: reply sent to %s turn %d (joined its working turn)",
            turn_id, caller.channel, caller.turn_id,
        )
    else:
        logger.info(
            "[SUBTASK] subtask %d: reply started on %s turn %d (caller turn %d)",
            turn_id, caller.channel, result.turn_id, caller.turn_id,
        )
