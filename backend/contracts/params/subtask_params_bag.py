"""SubtaskParamsBag — the typed input contract of the ``subtask`` ability.

The multi-action shape (see :mod:`contracts.params.plan_params_bag`): a plain
router reads ``action`` once and fans out to one frozen leaf per action.

* ``start`` carries ``instructions`` — the whole task, written to stand
  alone: the subtask reads none of the caller's conversation, so anything
  the task needs must already be inside the string.
* ``message`` carries ``subtask`` — the subtask's number (a digit string is
  accepted) — and ``message``, the follow-up sent into that subtask's turn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from abilities._result import ToolResult
from configs.enums.param_key import Keys
from contracts.params.param_bag import ParamBag


class SubtaskParamsBag(ParamBag):
    """Router and boundary type: ``SubtaskAbility.run()`` is annotated with
    this class, but every instance that reaches it is one of the leaves below."""

    __slots__ = ()

    @classmethod
    def from_params(cls, params: dict[str, object]) -> SubtaskParamsBag | ToolResult:
        match params.get(Keys.action):
            case "start":
                return SubtaskStartParams.from_params(params)
            case "message":
                return SubtaskMessageParams.from_params(params)
            case unknown:
                return ToolResult.err(
                    f"Unknown subtask action: {unknown!r}.",
                    code="unknown-action",
                    valid=("start", "message"),
                )


@dataclass(frozen=True, slots=True)
class SubtaskStartParams(SubtaskParamsBag):
    """``instructions`` — the whole task, self-contained: the subtask sees
    none of the caller's conversation, only this string."""

    instructions: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        instructions = cls.require_str(params, Keys.instructions)
        if isinstance(instructions, ToolResult):
            return instructions
        return cls(instructions=instructions)


@dataclass(frozen=True, slots=True)
class SubtaskMessageParams(SubtaskParamsBag):
    """``subtask`` — the number of the subtask to message; ``message`` — the
    follow-up it receives, joined into its turn while it works."""

    subtask: int
    message: str

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        raw_subtask = params.get(Keys.subtask)
        if isinstance(raw_subtask, str) and raw_subtask.strip().isdigit():
            raw_subtask = int(raw_subtask)
        if not isinstance(raw_subtask, int) or isinstance(raw_subtask, bool) or raw_subtask < 1:
            return ToolResult.err(
                "'subtask' must be the subtask's number (1, 2, 3…).",
                code="invalid-param",
                hint='pass "subtask" as the number from "Subtask #N started".',
            )
        message = cls.require_str(params, Keys.message)
        if isinstance(message, ToolResult):
            return message
        return cls(subtask=raw_subtask, message=message)
