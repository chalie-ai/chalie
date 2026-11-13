"""Plan — the to-do list the agent keeps while it works one message.

Each message gets one plan, stored against its input transcript row
(``mp.uid``): ``create`` writes 1 to 10 tasks once, ``add`` appends tasks for
work the plan missed (a message joined mid-work, a discovery), ``update`` moves
one task's status. The current plan is shown back to the model by the prompt
on every step until the message's exchange settles, so the result here is a
short acknowledgement — the prompt is the one display of state.
"""

from __future__ import annotations

from typing import ClassVar

from abilities._ability import Ability
from abilities._result import ToolResult
from configs.enums.param_key import Keys
from configs.enums.plan_task_status import PlanTaskStatus
from contracts.params.param_bag import ParamBag
from contracts.params.plan_params_bag import (
    MAX_PLAN_TASKS,
    PlanAddParams,
    PlanCreateParams,
    PlanParamsBag,
    PlanTaskDraft,
    PlanUpdateParams,
)
from models.plan_task import PlanTask
from services.database import Database


class PlanAbility(Ability[PlanParamsBag]):
    DISCOVERABLE: ClassVar[bool] = False
    NAME: ClassVar[str] = "plan"

    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {
        "create": (Keys.tasks,),
        "add": (Keys.tasks,),
        "update": (Keys.task, Keys.status),
    }
    PARAMS: ClassVar[type[ParamBag] | None] = PlanParamsBag

    def get_summary(self) -> str:
        return (
            "Your to-do list for the current message. action='create' writes the "
            f"plan once: 1 to {MAX_PLAN_TASKS} tasks in order, each a short `title` "
            "and the specific `task` to perform. action='add' appends tasks for work "
            "the plan missed. action='update' sets one task's `status` by its "
            "number: in_progress when you start it, done when finished, cancelled "
            "when it is no longer needed."
        )

    def get_examples(self) -> list[str]:
        return [
            "plan the steps to answer this message",
            "make a to-do list before starting the work",
            "mark task 1 as in progress",
            "mark task 2 as done",
            "cancel task 3, it is no longer needed",
            "add a task to reply to the follow-up question",
        ]

    def get_search_tooltip(self) -> str:
        return "to-do list for the current message"

    _TASKS_SCHEMA: ClassVar[dict[str, object]] = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                Keys.title_: {"type": "string", "description": "A few words naming the task."},
                Keys.task: {"type": "string", "description": "The specific work to perform."},
            },
            "required": [Keys.title_, Keys.task],
        },
    }

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.action: {"type": "string", "enum": ["create", "add", "update"]},
            Keys.tasks: {**_TASKS_SCHEMA, "description": "create / add: the tasks, in order."},
            Keys.task: {"type": "integer", "minimum": 1, "description": "update: the task's number."},
            Keys.status: {
                "type": "string",
                "enum": [status.value for status in PlanTaskStatus],
                "description": "update: the task's new status.",
            },
        },
        "required": [Keys.action],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: PlanParamsBag) -> ToolResult:
        message_id = self._message_id()
        with Database.transaction():
            if isinstance(params, PlanCreateParams):
                return self._create(message_id, params.tasks)
            if isinstance(params, PlanAddParams):
                return self._add(message_id, params.tasks)
            if isinstance(params, PlanUpdateParams):
                return self._update(message_id, params)
        raise TypeError(f"{self.NAME}.run() got an unrouted bag: {type(params).__name__}")

    def _message_id(self) -> int:
        """The input transcript row the plan belongs to. Only configs with an
        input row pin this tool, so a missing one is a wiring bug — fail loud."""
        uid = self.mp.uid if self.mp is not None else None
        if uid is None:
            raise RuntimeError(f"{self.NAME} requires a bound MessageProcessor with an input row")
        return uid

    @staticmethod
    def _create(message_id: int, drafts: list[PlanTaskDraft]) -> ToolResult:
        if PlanTask.next_position(message_id) > 1:
            return ToolResult.err(
                "This message already has a plan.",
                code="plan-exists",
                hint="use action 'add' for new tasks, or 'update' to change a task's status.",
            )
        PlanAbility._insert(message_id, 1, drafts)
        return ToolResult.ok(f"Plan saved: {len(drafts)} tasks.")

    @staticmethod
    def _add(message_id: int, drafts: list[PlanTaskDraft]) -> ToolResult:
        first = PlanTask.next_position(message_id)
        if first == 1:
            return ToolResult.err(
                "This message has no plan yet.",
                code="no-plan",
                hint="create the plan first with action 'create'.",
            )
        PlanAbility._insert(message_id, first, drafts)
        last = first + len(drafts) - 1
        return ToolResult.ok(f"Added task {first}." if first == last else f"Added tasks {first}–{last}.")

    @staticmethod
    def _update(message_id: int, params: PlanUpdateParams) -> ToolResult:
        row = PlanTask.at(message_id, params.task)
        if row is None:
            numbers = tuple(str(task.position) for task in PlanTask.for_transcript(message_id))
            return ToolResult.err(
                f"There is no task {params.task}.",
                code="unknown-task",
                hint="create the plan first with action 'create'." if not numbers else None,
                valid=numbers,
            )
        row.status = params.status.value
        row.save()
        return ToolResult.ok(f"Task {params.task} is {params.status.value}.")

    @staticmethod
    def _insert(message_id: int, first: int, drafts: list[PlanTaskDraft]) -> None:
        for position, draft in enumerate(drafts, start=first):
            PlanTask(
                transcript_id=message_id,
                position=position,
                title=draft.title,
                task=draft.task,
                status=PlanTaskStatus.TODO.value,
            ).save()
