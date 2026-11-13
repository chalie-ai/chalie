"""PlanTask — one ``plan_tasks`` row: one task of the plan the agent keeps while
it works one message.

Active-record row-model. INTEGER-autoincrement PK, so the base
``Model.save()``/``delete()`` apply unchanged. ``transcript_id`` is the
message's input (user) transcript row; ``position`` is the 1-based task number
the model reads and refers to. ``status`` holds a
:class:`~configs.enums.plan_task_status.PlanTaskStatus` value. Holds no mp,
calls no service.
"""

from __future__ import annotations

from typing import ClassVar, Self

from models.model import Model


class PlanTask(Model):
    """One plan task: field storage + the per-plan reads the plan ability, the
    prompt and the turn serializer need."""

    __columns__: ClassVar[tuple[str, ...]] = (
        "id", "transcript_id", "position", "title", "task", "status",
    )

    @classmethod
    def get_table(cls) -> str:
        return "plan_tasks"

    transcript_id: int
    position: int
    title: str
    task: str
    status: str

    @classmethod
    def for_transcript(cls, transcript_id: int) -> list[Self]:
        """One message's plan, in task order. Empty = no plan yet."""
        return cls.filter("transcript_id", transcript_id).order_by("position").get()

    @classmethod
    def for_transcripts(cls, transcript_ids: list[int]) -> list[Self]:
        """Every task of every listed message's plan, in one query (the turn
        serializer's batch read). Ordered by id, which rises with position
        inside one plan. Empty input is a clean no-op — no query runs."""
        return cls._select_where_in_json("transcript_id", transcript_ids, order_by="id")

    @classmethod
    def next_position(cls, transcript_id: int) -> int:
        """The number the next appended task takes — 1 for an empty plan. Raw
        SQL: the ``MAX()`` aggregate can't be expressed by the builder."""
        row = cls._bound_connection().execute(
            "SELECT COALESCE(MAX(position), 0) + 1 FROM plan_tasks WHERE transcript_id = ?",
            (transcript_id,),
        ).fetchone()
        return int(row[0]) if row is not None else 1

    @classmethod
    def at(cls, transcript_id: int, position: int) -> Self | None:
        """The one task numbered ``position`` in a message's plan, or ``None``."""
        return cls.filter("transcript_id", transcript_id).filter("position", position).first()
