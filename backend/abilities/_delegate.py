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
the child's number, and the outcome of each of the child's turns arrives
later in the chat as a message of its own (:func:`reply_to_caller`), sent at
the end of that turn however it ended; every other caller — a scheduled task, a subagent's own
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
from configs.enums.config_type import ConfigTypeEnum
from models.agent import Agent
from models.tool_call import ToolCall
from models.transcript import Transcript
from models.turn_execution import TurnExecution
from services.agent_service import CANCELLED_TEXT, CRASHED_TEXT, AgentService

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
        place it is chosen, read when the child is spawned and each time a
        message opens its next turn (:func:`agent_config`)."""

    def spawn(self, config: "ProcessorConfig", instructions: str) -> "MessageProcessor":
        """Open ``instructions`` as a child turn on ``config`` and return it
        without waiting. The child's input row carries this call's
        ``tool_calls`` id, written in the same transaction that opens the turn,
        so the parent's pill can be followed to the child's transcript; its
        ``agents`` row — the caller's channel and turn, the plain task, and
        the caller's input row as the task's context (``context_id``, the
        caller's ``mp.uid``: ``None`` when the caller's turn wrote no input
        row, a hidden-input turn) — is written in that transaction too. Only
        the plain task is stored as the child's input row; its first turn
        reads it inside the first-turn template, with that context, as the
        prompt is built (:func:`~abilities.message_parent.first_turn`). The
        child is passed the model and thinking level the caller's turn runs
        on; a child on the chat lane starts its thread on them. The
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
                    context_id=mp.uid,
                ),
                "provider_id": mp.provider_id,
                "thinking_level": mp.thinking_override or "auto",
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
        """Empty on a call that returned at once — it carries no answer yet;
        each late reply carrying one of the child's answers is followed by the
        follow-up of that answer instead, in the prompt of the turn the reply
        opens (:func:`reply_follow_up`)."""
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
    late. A message a subagent sent with ``message_parent`` needs the tool
    that replies to it. Empty for any other tool."""
    # Lazy: the registry imports this module.
    from abilities._registry import AbilityRegistry  # noqa: PLC0415

    # Lazy: these abilities import this module.
    from abilities.message_parent import MessageParentAbility  # noqa: PLC0415
    from abilities.message_subagent import MessageSubagentAbility  # noqa: PLC0415

    if tool_name == MessageParentAbility.NAME:
        return (MessageSubagentAbility.NAME,)
    if not AbilityRegistry.is_watchable_delegate(tool_name):
        return ()
    return cast("DelegateAbility", AbilityRegistry.get(tool_name)).FOLLOW_UP_TOOLS


def agent_config(agent: Agent) -> "ProcessorConfig":
    """The config every later turn of ``agent`` runs on: its delegate's
    :meth:`DelegateAbility.child_config` for the chat — the one caller that
    messages a subagent — on the subagent's own channel."""
    # Lazy: the registry and the channel configs import this module (through
    # the abilities they load or pin).
    from abilities._registry import AbilityRegistry  # noqa: PLC0415
    from configs.channels import UserConfig  # noqa: PLC0415

    delegate = AbilityRegistry.get(agent.type)
    if not isinstance(delegate, DelegateAbility):
        raise RuntimeError(f"subagent #{agent.agent_id}: {agent.type} is not a delegate")
    config = delegate.child_config(UserConfig())
    if config is None:
        raise RuntimeError(f"subagent #{agent.agent_id}: {agent.type} runs no turn for the chat")
    return config.on_channel(agent.channel)


def reply_to_caller(channel: str, turn_id: int, answer: str) -> None:
    """Send the outcome of the subagent turn ``(channel, turn_id)`` to the
    channel that spawned it, when that caller did not wait for it — only the
    person's chat (:func:`~abilities.message_parent.holds`). The reply of
    each of the subagent's turns is linked to the call that spawned it
    (:meth:`AgentService.spawn_call_id`), so each lands labelled with the
    subagent's number; the memory step's turn on its channel has no caller.

    Asked of every turn once it has ended — by the end-of-turn hook, and by the
    boot sweep for a turn a restart cut off. The outcome is read from the
    turn's own execution row, never from a MessageProcessor — a cancel lands
    on another thread, and the boot sweep has no MP at all. The reply's text is
    exactly the turn's answer, or a notice of how it ended: the same text
    :meth:`AgentService.last_output` gives for a subagent cancelled or
    crashed. Nothing is posted for a turn stopped by its caller, nor for one
    that completed with a delivered ``message_parent`` as its last tool call
    (:func:`~abilities.message_parent.ended_on_message_parent`) — its
    question has already reached the chat; one that ended on a refused or
    failed ``message_parent`` sent nothing, so its answer is posted as usual. A reply carrying an answer also
    hands over that answer's row — this turn's newest settled row, read off
    the turn itself, never the agent's ``response_id``, which its next turn
    overwrites — so the chat turn the reply opens follows it with that
    answer's follow-up block (:func:`reply_follow_up`); a notice hands over
    none. It is sent through :meth:`MessageProcessor.deliver`: a chat turn
    of its own, now or queued.
    """
    # Lazy: the turn-execution service imports the channel configs, which
    # import this module (through the abilities they pin or load); so do the
    # controller and message_parent.
    from abilities.message_parent import ended_on_message_parent, holds  # noqa: PLC0415
    from configs.channels import UserConfig  # noqa: PLC0415
    from controllers.message_processor import MessageProcessor  # noqa: PLC0415
    from services.turn_execution_service import STOP_REASON_PARENT_STOPPED  # noqa: PLC0415

    agent = AgentService.of_channel(channel)
    if agent is None or not holds(agent):
        return
    rows = Transcript.by_turn(channel, turn_id)
    if rows and rows[0]["role"] == "memory":
        return

    name = agent.type
    execution = TurnExecution.latest(channel, turn_id)
    if execution is None or execution.ended_at is None:
        raise RuntimeError(f"{name} #{agent.agent_id} turn {turn_id} ended without a closed turn execution")
    response_id: int | None = None
    if execution.state == TurnExecution.COMPLETED:
        if ended_on_message_parent(channel, turn_id):
            logger.info("[DELEGATE] %s #%d turn %d: ended on a message to its caller, no reply",
                        name, agent.agent_id, turn_id)
            return
        if answer.strip():
            text = answer
            response_id = Transcript.last_answer_id(channel, turn_id)
            if response_id is None:
                raise RuntimeError(f"{name} #{agent.agent_id} turn {turn_id} answered without a settled row")
        else:
            text = f"This {name} finished without an answer."
    elif execution.state == TurnExecution.CANCELLED:
        if execution.stop_reason == STOP_REASON_PARENT_STOPPED:
            # Stopped by its caller: there is no one to reply to.
            logger.info("[DELEGATE] %s #%d turn %d: stopped by its caller, no reply", name, agent.agent_id, turn_id)
            return
        text = CANCELLED_TEXT
    elif execution.state == TurnExecution.CRASHED:
        text = CRASHED_TEXT
    else:
        raise RuntimeError(f"{name} #{agent.agent_id} turn {turn_id} ended in unexpected state {execution.state!r}")

    call_id = AgentService.spawn_call_id(agent)
    if call_id is None:
        raise RuntimeError(f"{name} #{agent.agent_id}: the call that spawned it no longer exists")
    opened = MessageProcessor.deliver(UserConfig(), text, call_id, response_id=response_id)
    logger.info(
        "[DELEGATE] %s #%d turn %d: reply sent to %s (%s)", name, agent.agent_id, turn_id,
        agent.spawned_by_channel, f"turn {opened.turn_id}" if opened is not None else "queued",
    )


def reply_follow_up(row: Transcript, response_id: int | None) -> str:
    """What follows a delegate's late reply ``row`` in the prompt of the turn
    it opened: a newline and the follow-up block its tool's result would have
    carried (:meth:`Ability.follow_up`), built from the content of
    ``response_id`` — the row of the answer the reply carries, which the turn
    holds in its metadata (:meth:`MessageProcessor.deliver`) — or ``''`` when
    the turn holds none: a notice (no answer, stopped, crashed) or any
    message that is not a reply. The block follows that answer whichever turn
    the subagent is on by now. ``row`` only finds the delegate call that
    spawned the subagent, and with it the tool whose follow-up applies; a
    reply carrying an answer always has one, so a row that does not raises.
    Shown only to the turn the reply opened, never stored, so the reply's own
    content stays exactly what the delegate said and every later turn reads
    it bare."""
    # Lazy: the registry imports this module.
    from abilities._registry import AbilityRegistry  # noqa: PLC0415

    if response_id is None:
        return ""
    link = row.tool_call_id
    call = ToolCall.get(int(link)) if link is not None and link.isdecimal() else None
    if call is None or not AbilityRegistry.is_watchable_delegate(call.tool_name):
        raise RuntimeError(f"transcript row {row.id} carries answer row {response_id} but no delegate call")
    answer = Transcript.get(response_id)
    if answer is None:
        raise RuntimeError(f"transcript row {row.id} carries answer row {response_id}, which does not exist")
    follow_up = AbilityRegistry.get(call.tool_name).follow_up(ToolResult.ok(answer.content), None)
    return f"\n{FOLLOW_UP_BLOCK.format(text=follow_up)}" if follow_up else ""
