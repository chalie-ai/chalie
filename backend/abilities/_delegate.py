# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Shared infrastructure for delegate tools (subagent-as-tools).

A delegate tool is a standalone Ability that runs a child turn on its OWN
``ProcessorConfig`` subclass — there is no MessageProcessor subclass, no
SUBAGENT_TYPES registry, and no make_subagent_config() factory.

The framework provides one base class — :class:`DelegateAbility` — that every
delegate tool running its child turn through :meth:`DelegateAbility.spawn`
inherits. That runner opens the child linked to the parent's tool call and
returns it at once. Who waits for the child is settled by the caller's kind
alone (:func:`replies_late`): the person's chat never waits — the call returns
the child's number, and the child's outcome arrives later in the chat as a
message of its own (:func:`reply_to_caller`), sent at the end of the child's
turn however it ended; every other caller — a scheduled task, a subagent's own
turn, discovery, an outside agent — waits for the answer and gets it as the
call's result (:meth:`DelegateAbility.wait_for`). A DelegateAbility's call is
exactly one whose child turn can be watched. Each ``run()`` still owns its pre-flight
(``pim`` checks the mail connection, ``code_agent`` creates its workspace).
``vision`` is a delegate by category only: it describes through
``ImageDescription``, writes no transcript, and so is a plain Ability.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, cast

from abilities._ability import FOLLOW_UP_BLOCK, Ability, B
from abilities._result import ToolResult
from configs.enums.channels import WATCHABLE_DELEGATE_CHANNELS
from configs.enums.config_type import ConfigTypeEnum
from models.agent import Agent
from models.tool_call import ToolCall
from models.transcript import Transcript
from models.turn_execution import TurnExecution

if TYPE_CHECKING:
    from controllers.message_processor import MessageProcessor
    from services.processor_config import ProcessorConfig

logger = logging.getLogger(__name__)


def replies_late(kind: "ConfigTypeEnum | str | None") -> bool:
    """Whether a delegate called from a turn of this kind runs in the
    background and replies later, instead of keeping the caller waiting — the
    one rule both sides of a call read: the call itself (the caller's config
    type) and the child's end (the type stored on the caller's turn
    execution). Only the person's chat — a user turn, its main exchange or a
    thread reply — is replied to late; every other caller waits. It is settled
    by the caller's kind alone, never by anything the model sets."""
    return kind == ConfigTypeEnum.USER


def reply_label(name: str, n: int) -> str:
    """How a reply from ``name``'s subagent ``n`` (its ``agent_id``) is labelled
    on the caller's channel — the label its started result promises and its
    speaker shows."""
    return f"Reply from {name} #{n}"


class DelegateAbility(Ability[B], ABC):
    """Base class for the delegate tools that run their child turn through
    :meth:`spawn`. Abstract (lists ``ABC`` directly, like the other base
    mix-ins) so it is never registered as a tool in its own right; concrete
    delegates fill in the getters and their own ``run()``, so the base is
    generic over the subclass's bag (``DelegateAbility[ItsBag]``)."""

    #: Tools this delegate's follow-up instruction names: active, along with
    #: the delegate itself, on the turn that answers its late reply.
    FOLLOW_UP_TOOLS: ClassVar[tuple[str, ...]] = ()

    @classmethod
    @abstractmethod
    def child_config(cls, caller: "ProcessorConfig") -> "ProcessorConfig | None":
        """The config this delegate's child turn runs on when ``caller``
        starts it, or ``None`` when ``caller`` may not start one — the one
        place it is chosen, read both when the child is spawned and when a
        message is later sent into its running turn."""

    def spawn(self, config: "ProcessorConfig", instructions: str) -> "MessageProcessor":
        """Open ``instructions`` as a child turn on ``config`` and return it
        without waiting. The child's input row carries this call's
        ``tool_calls`` id, written in the same transaction that opens the turn,
        so the parent's pill can be followed to the child's transcript; its
        ``agents`` row — the caller's channel and turn, and the plain task —
        is written in that transaction too. The
        parent's frame is sent again once that row exists — the earlier
        ``started`` frame went out before there was a child turn to point at."""
        from controllers.message_processor import MessageProcessor  # noqa: PLC0415

        mp = self.mp
        if mp is None:
            raise RuntimeError(f"{self.NAME}.run() dispatched without a bound MessageProcessor")

        # A gated tool inside the delegate prompts on the CALLER's turn — the
        # delegate's own turn has no surface a human could answer from.
        child = MessageProcessor.process(
            config,
            raw_input=instructions,
            metadata={
                "origin": mp.origin,
                "tool_call_id": self.tool_call_id,
                "agent": Agent(
                    type=self.NAME, prompt=instructions,
                    spawned_by_channel=mp.channel, spawned_in_turn=mp.turn_id,
                ),
            },
        )
        mp.tool_call_service.reemit(self.tool_call_id)
        return child

    def delegate(self, instructions: str, *, hint: str) -> ToolResult:
        """Run ``instructions`` as a child turn on :meth:`child_config` of
        the caller's config, through :meth:`spawn`.

        A caller that is replied to late (:func:`replies_late`) gets the
        child's number at once (:meth:`started`); any other caller waits for
        the answer (:meth:`wait_for`)."""
        mp = self.mp
        if mp is None:
            raise RuntimeError(f"{self.NAME}.run() dispatched without a bound MessageProcessor")
        config = self.child_config(mp.config)
        if config is None:
            raise RuntimeError(f"{self.NAME}.run() reached delegate() for a caller that may not start one")
        child = self.spawn(config, instructions)
        if replies_late(mp.config.type()):
            return self.started(child)
        return self.wait_for(child, hint=hint)

    def wait_for(self, child: "MessageProcessor", *, hint: str) -> ToolResult:
        """Wait for ``child``'s turn to end and return its outcome as the
        call's result. A child the user stopped from its transcript panel ends
        CANCELLED; that is reported as ``delegate-stopped`` — read from the
        child's execution row, never inferred from its text — so the caller
        does not re-run work the user chose to stop. Otherwise the answer is
        shaped by :func:`delegate_result`, ``hint`` steering an empty one."""
        answer = child.result()
        if child.execution is not None and child.execution.state == TurnExecution.CANCELLED:
            return ToolResult.err(
                "The user stopped this subagent before it finished. Do not retry it or "
                "delegate the same task again; answer with what you already have, or tell "
                "the user it was stopped.",
                code="delegate-stopped",
            )
        return delegate_result(answer, hint=hint)

    def started(self, child: "MessageProcessor") -> ToolResult:
        """The result of a call whose child runs in the background: the
        subagent and its number (its ``agent_id``), written as
        :func:`reply_label` writes them, and the label its answer will arrive
        under."""
        if child.agent is None:
            raise RuntimeError(f"{self.NAME}'s child turn {child.turn_id} opened without its agent row")
        n = child.agent.agent_id
        return ToolResult.ok(
            f"Subagent {self.NAME} #{n} has been spawned. It will reply to you once it's done as "
            f"\"{reply_label(self.NAME, n)}\". Acknowledge to the user that the agent has been "
            "spawned and you'll be invoked automatically once it replies."
        )

    def follow_up(self, tr: ToolResult, action: "str | None") -> str:
        """None on a call that returned at once — it carries no answer yet; the
        turn answering the child's late reply gets the follow-up after the
        reply instead (:func:`reply_follow_up`)."""
        if self.mp is not None and replies_late(self.mp.config.type()):
            return ""
        return super().follow_up(tr, action)


def delegate_result(result: str, *, hint: str) -> ToolResult:
    """An empty body is NOT success — mapped to ``code=delegate-no-answer`` so a
    weak outer model self-corrects instead of trusting the silence."""
    if not result.strip():
        return ToolResult.err(
            "The delegate finished without producing an answer "
            "(it exhausted its iteration budget or was cancelled).",
            code="delegate-no-answer",
            hint=hint,
        )
    return ToolResult.ok(result)


def replying_tools(tool_name: str) -> tuple[str, ...]:
    """The tools a turn answering a late reply from ``tool_name`` needs active
    beyond its pinned ones: those the delegate's follow-up instruction names.
    The delegate itself is already pinned on the chat, the only kind replied to
    late. Empty for any other tool."""
    # Lazy: the registry imports this module.
    from abilities._registry import AbilityRegistry  # noqa: PLC0415

    if not AbilityRegistry.is_watchable_delegate(tool_name):
        return ()
    return cast("DelegateAbility", AbilityRegistry.get(tool_name)).FOLLOW_UP_TOOLS


def reply_to_caller(channel: str, turn_id: int, answer: str) -> None:
    """Send the outcome of the delegate child turn ``(channel, turn_id)`` to the
    turn whose tool call started it, when that caller did not wait for it.

    Asked of every turn once it has ended — by the end-of-turn hook, and by the
    boot sweep for a turn a restart cut off — so it first settles, from stored
    rows alone, whether there is anyone to reply to: only a watchable delegate's
    turn has a caller; the memory step's turn on a delegate channel (its first
    row carries the ``memory`` role) has none; and a caller that waited
    (:func:`replies_late`) got the answer as its call's result. So the caller
    replied to is always the person's chat.

    The outcome is read from the child's own execution row, never from a
    MessageProcessor — a cancel lands on another thread, and the boot sweep has
    no MP at all. The reply's text is exactly the child's answer, or a notice
    of how it ended; the follow-up an answer carries reaches the model only
    while the reply is being answered (:func:`reply_follow_up`). The reply
    joins the caller's turn while it works; otherwise a reply to the main
    exchange gets a new turn, and a thread reply is continued — always through
    :meth:`MessageProcessor.send`, the one input function every channel uses.
    """
    # Lazy: the channel configs and the controller import this module (through
    # the abilities they pin or load).
    from configs.channels import UserConfig  # noqa: PLC0415
    from controllers.message_processor import MessageProcessor  # noqa: PLC0415
    from services.turn_execution_service import (  # noqa: PLC0415
        STOP_REASON_PARENT_STOPPED,
        STOP_REASON_PROCESS_DEATH,
    )

    if channel not in WATCHABLE_DELEGATE_CHANNELS:
        return
    rows = Transcript.by_turn(channel, turn_id)
    if rows and rows[0]["role"] == "memory":
        return

    # The child's input row carries the tool_calls id of the call that spawned
    # it (DelegateAbility.spawn stamps it as the turn opens), and a message
    # later sent into the running child carries that same id — walk back from
    # it to the row that made the call, then to the caller's turn.
    link = next((row for row in rows if row["tool_call_id"] is not None), None)
    if link is None:
        raise RuntimeError(f"{channel} turn {turn_id}: no row of its turn carries the caller's tool_call_id")
    call = ToolCall.get(int(cast("str", link["tool_call_id"])))
    if call is None:
        raise RuntimeError(f"{channel} turn {turn_id}: its caller's tool call no longer exists")
    caller = Transcript.get(call.transcript_id)
    if caller is None or caller.turn_id is None:
        raise RuntimeError(
            f"{channel} turn {turn_id}: the caller's transcript row {call.transcript_id} "
            "no longer exists or carries no turn"
        )
    caller_execution = TurnExecution.latest(caller.channel, caller.turn_id)
    if caller_execution is None:
        raise RuntimeError(
            f"{channel} turn {turn_id}: caller {caller.channel} turn {caller.turn_id} "
            "has no turn execution to reply into"
        )
    if not replies_late(caller_execution.type):
        return

    name = call.tool_name
    execution = TurnExecution.latest(channel, turn_id)
    if execution is None or execution.ended_at is None:
        raise RuntimeError(f"{name} #{turn_id} ended without a closed turn execution")
    if execution.state == TurnExecution.COMPLETED:
        text = answer if answer.strip() else f"This {name} finished without an answer."
    elif execution.state == TurnExecution.CANCELLED:
        if execution.stop_reason == STOP_REASON_PARENT_STOPPED:
            # Stopped by its caller's own stop: the caller never gets an
            # answer, so there is no one to reply to.
            logger.info("[DELEGATE] %s #%d: stopped with its caller, no reply", name, turn_id)
            return
        text = f"This {name} was stopped before it finished."
    elif execution.state == TurnExecution.CRASHED:
        if execution.stop_reason == STOP_REASON_PROCESS_DEATH:
            text = f"This {name} was cut off by a restart before it finished."
        else:
            text = f"This {name} crashed: {execution.stop_reason}"
    else:
        raise RuntimeError(f"{name} #{turn_id} ended in unexpected state {execution.state!r}")

    # The caller's row sits in the turn's main exchange (no settle yet, or at
    # or before its first settled assistant row); a reply there must land on
    # the spine, never inside a running follow-up reply.
    settle0 = Transcript.settle0(caller.channel, caller.turn_id)
    spine_only = settle0 is None or cast("int", caller.id) <= settle0

    result = MessageProcessor.send(
        UserConfig(), text, metadata={"tool_call_id": call.id}, turn_id=caller.turn_id, spine_only=spine_only,
    )
    if result is None:
        raise RuntimeError(
            f"{name} #{turn_id}: the reply to {caller.channel} turn {caller.turn_id} failed to open"
        )
    logger.info(
        "[DELEGATE] %s #%d: reply sent to %s turn %d (caller turn %d, execution %s)",
        name, turn_id, caller.channel, result.turn_id, caller.turn_id, result.id,
    )


def reply_follow_up(row: Transcript) -> str:
    """What follows a delegate's late reply ``row`` in the prompt of the
    exchange answering it: a newline and the follow-up block its tool's result
    would have carried (:meth:`Ability.follow_up`), or ``''`` for any other row.

    Only a reply carrying an answer has one — its child turn's latest execution
    COMPLETED on a non-empty answer; a notice (no answer, stopped, crashed, cut
    off by a restart) has none. Derived from stored rows on every read, never
    stored, so the reply's own content stays exactly what the delegate said and
    every later exchange reads it bare."""
    # Lazy: the registry imports this module.
    from abilities._registry import AbilityRegistry  # noqa: PLC0415

    link = row.tool_call_id
    if link is None or not link.isdecimal():
        return ""
    call = ToolCall.get(int(link))
    if call is None or not AbilityRegistry.is_watchable_delegate(call.tool_name):
        return ""
    call_id = cast("int", call.id)
    child = Transcript.delegate_turns({call_id: (call.created_at, call.ended_at)}).get(call_id)
    if child is None:
        return ""
    channel, turn_id = cast("str", child["channel"]), cast("int", child["turn_id"])
    execution = TurnExecution.latest(channel, turn_id)
    if execution is None or execution.state != TurnExecution.COMPLETED:
        return ""
    answer = Transcript.last_answer(channel, turn_id) or ""
    if not answer.strip():
        return ""
    follow_up = AbilityRegistry.get(call.tool_name).follow_up(ToolResult.ok(answer), None)
    return f"\n{FOLLOW_UP_BLOCK.format(text=follow_up)}" if follow_up else ""
