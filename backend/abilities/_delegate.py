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
delegate tool running its child turn through :meth:`DelegateAbility.delegate`
inherits. That runner links the child to the parent's tool call and shapes the
answer through ``delegate_result``, so a DelegateAbility's call is exactly one
whose child turn can be watched. Each ``run()`` still owns its pre-flight
(``pim`` checks the mail connection, ``code_agent`` creates its workspace).
``vision`` is a delegate by category only: it describes through
``ImageDescription``, writes no transcript, and so is a plain Ability.
"""

from __future__ import annotations

import logging
from abc import ABC
from typing import TYPE_CHECKING, cast

from abilities._ability import Ability, B
from abilities._result import ToolResult
from models.turn_execution import TurnExecution

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Protocol

    from configs.enums.policy_channel import PolicyChannel
    from services.processor_config import ProcessorConfig

    class _ActTrailRenderer(Protocol):
        def _render_act_trail(self) -> str | None: ...


logger = logging.getLogger(__name__)


class DelegateAbility(Ability[B], ABC):
    """Base class for the delegate tools that run their child turn through
    :meth:`delegate`. Abstract (lists ``ABC`` directly, like the other base
    mix-ins) so it is never registered as a tool in its own right; concrete
    delegates fill in the getters and their own ``run()``, so the base is
    generic over the subclass's bag (``DelegateAbility[ItsBag]``)."""

    def delegate(
        self, config: "Callable[[PolicyChannel], ProcessorConfig]", instructions: str, *, hint: str,
    ) -> ToolResult:
        """Run ``instructions`` as a child turn on ``config`` and return its
        answer. The child's input row carries this call's ``tool_calls`` id,
        written in the same transaction that opens the turn, so the parent's
        pill can be followed to the child's transcript. The parent's frame is
        sent again once that row exists — the earlier ``started`` frame went out
        before there was a child turn to point at.

        A child the user stopped from its transcript panel ends CANCELLED; that
        is reported as ``delegate-stopped`` — read from the child's execution
        row, never inferred from its text — so the caller does not re-run work
        the user chose to stop."""
        from controllers.message_processor import MessageProcessor  # noqa: PLC0415

        mp = self.mp
        if mp is None:
            raise RuntimeError(f"{self.NAME}.run() dispatched without a bound MessageProcessor")

        # A gated tool inside the delegate prompts on the CALLER's turn — the
        # delegate's own turn has no surface a human could answer from.
        child = MessageProcessor.process(
            config(mp.config.policy_channel),
            raw_input=instructions,
            metadata={"origin": mp.origin, "tool_call_id": self.tool_call_id},
        )
        mp.tool_call_service.reemit(self.tool_call_id)
        answer = child.result()
        if child.execution is not None and child.execution.state == TurnExecution.CANCELLED:
            return ToolResult.err(
                "The user stopped this subagent before it finished. Do not retry it or "
                "delegate the same task again; answer with what you already have, or tell "
                "the user it was stopped.",
                code="delegate-stopped",
            )
        return delegate_result(answer, hint=hint)


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


def render_trail(mp: object) -> str:
    """Render the current act-trail for a delegate's user prompt, or '' on miss."""
    try:
        trail = cast("_ActTrailRenderer", mp)._render_act_trail()
        return trail or ""
    except Exception:
        logger.warning("[DELEGATE] act-trail render failed", exc_info=True)
        return ""
