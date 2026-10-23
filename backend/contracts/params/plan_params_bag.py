"""PlanParamsBag — the typed input contract of the ``plan`` ability.

The multi-action shape (see :mod:`contracts.params.list_params_bag`): a plain
router reads ``action`` once and fans out to one frozen leaf per action.

* ``create`` / ``add`` carry ``tasks`` — a non-empty array of
  ``{"title", "task"}`` objects, both non-blank. A weak model often serialises
  the array, so a JSON-array STRING is parsed. ``create`` holds 1 to
  :data:`MAX_PLAN_TASKS` tasks; ``add`` appends any number.
* ``update`` carries ``task`` — the 1-based task number (a digit string is
  accepted) — and ``status``, one of :class:`PlanTaskStatus`; ``-`` and spaces
  fold to ``_`` so ``in-progress`` lands.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Self

from abilities._result import ToolResult
from configs.enums.param_key import Keys
from configs.enums.plan_task_status import PlanTaskStatus
from contracts.params.param_bag import ParamBag

MAX_PLAN_TASKS = 10

_ACTIONS = ("create", "add", "update")
_STATUSES = tuple(status.value for status in PlanTaskStatus)
_TASKS_HINT = (
    "pass 'tasks' as a JSON array of objects, e.g. "
    '[{"title":"Find the flight","task":"Search the inbox for the booking email"}].'
)


@dataclass(frozen=True, slots=True)
class PlanTaskDraft:
    """One task as the model wrote it, before it has a number."""

    title: str
    task: str


def _tasks(params: dict[str, object]) -> list[PlanTaskDraft] | ToolResult:
    """``tasks`` as drafts: a non-empty array whose every element carries a
    non-blank ``title`` and ``task``; the first bad element is named."""
    raw = params.get(Keys.tasks)
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            pass
    if not isinstance(raw, list) or not raw:
        return ToolResult.err(
            "'tasks' must be a non-empty array.", code="invalid-tasks", hint=_TASKS_HINT,
        )
    drafts: list[PlanTaskDraft] = []
    for i, item in enumerate(raw, start=1):
        title = item.get(Keys.title_) if isinstance(item, dict) else None
        task = item.get(Keys.task) if isinstance(item, dict) else None
        if not isinstance(title, str) or not title.strip() or not isinstance(task, str) or not task.strip():
            return ToolResult.err(
                f"Task {i} needs a non-blank 'title' and 'task'.",
                code="invalid-tasks",
                hint=_TASKS_HINT,
            )
        drafts.append(PlanTaskDraft(title=title.strip(), task=task.strip()))
    return drafts


class PlanParamsBag(ParamBag):
    """Router and boundary type: ``PlanAbility.run()`` is annotated with this
    class, but every instance that reaches it is one of the leaves below."""

    __slots__ = ()

    @classmethod
    def from_params(cls, params: dict[str, object]) -> PlanParamsBag | ToolResult:
        match params.get(Keys.action):
            case "create":
                return PlanCreateParams.from_params(params)
            case "add":
                return PlanAddParams.from_params(params)
            case "update":
                return PlanUpdateParams.from_params(params)
            case unknown:
                return ToolResult.err(
                    f"Unknown plan action: {unknown!r}.", code="unknown-action", valid=_ACTIONS,
                )


@dataclass(frozen=True, slots=True)
class PlanCreateParams(PlanParamsBag):
    """``tasks`` — the whole plan, 1 to :data:`MAX_PLAN_TASKS` tasks in order."""

    tasks: list[PlanTaskDraft]

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        if isinstance(tasks := _tasks(params), ToolResult):
            return tasks
        if len(tasks) > MAX_PLAN_TASKS:
            return ToolResult.err(
                f"A plan holds at most {MAX_PLAN_TASKS} tasks; got {len(tasks)}.",
                code="too-many-tasks",
                hint="merge related steps, or add more tasks later with action 'add'.",
            )
        return cls(tasks=tasks)


@dataclass(frozen=True, slots=True)
class PlanAddParams(PlanParamsBag):
    """``tasks`` — new tasks, appended after the plan's last task."""

    tasks: list[PlanTaskDraft]

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        if isinstance(tasks := _tasks(params), ToolResult):
            return tasks
        return cls(tasks=tasks)


@dataclass(frozen=True, slots=True)
class PlanUpdateParams(PlanParamsBag):
    """``task`` — the 1-based task number; ``status`` — its new status."""

    task: int
    status: PlanTaskStatus

    @classmethod
    def from_params(cls, params: dict[str, object]) -> Self | ToolResult:
        raw_task = params.get(Keys.task)
        if isinstance(raw_task, str) and raw_task.strip().isdigit():
            raw_task = int(raw_task)
        if not isinstance(raw_task, int) or isinstance(raw_task, bool) or raw_task < 1:
            return ToolResult.err(
                "'task' must be the task's number (1, 2, 3…).",
                code="invalid-param",
                hint="pass 'task' as the number shown before the task in the plan.",
            )
        raw_status = params.get(Keys.status)
        folded = raw_status.strip().lower().replace("-", "_").replace(" ", "_") if isinstance(raw_status, str) else ""
        if folded not in _STATUSES:
            return ToolResult.err(
                f"Unknown status: {raw_status!r}.", code="invalid-status", valid=_STATUSES,
            )
        return cls(task=raw_task, status=PlanTaskStatus(folded))
