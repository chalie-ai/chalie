"""DTOs for the skills resource — read/create/update HTTP contract.

One class per file is the namespace convention. Validation (length, non-empty)
lives on the ``pydantic.Field`` constraints; the create DTO strips inbound
strings so the DB stores trimmed values (the old handler's ``.strip()``).
``SkillUpdate`` preserves the old PUT semantic where a blank ``use_for``/
``content`` keeps the existing value — a ``before`` validator strips and
collapses empty to ``None``. ``tags`` is different: an explicit empty string
clears it, so it has no such validator.
"""

from __future__ import annotations

from pydantic import Field, field_validator

from .base import DTO


class Skill(DTO):
    """Read shape for a skill (mirrors the skills.sqlite row)."""

    id: int
    title: str
    use_for: str
    content: str
    tags: str = ""
    version: int
    source: str
    enabled: bool
    based_on: int | None


class SkillCreate(DTO):
    """Inbound body to create a user skill."""

    title: str = Field(..., min_length=1, max_length=200)
    use_for: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    tags: str = ""

    @field_validator("title", "use_for", "content", "tags", mode="before")
    @classmethod
    def _strip(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class SkillUpdate(DTO):
    """Partial update of a user skill by id; title is never updatable.

    A blank ``use_for``/``content`` collapses to ``None`` (keep existing); a
    blank ``tags`` stays ``""`` (clears). Only present fields are applied.
    """

    use_for: str | None = Field(default=None, min_length=1)
    content: str | None = Field(default=None, min_length=1)
    tags: str | None = None

    @field_validator("use_for", "content", mode="before")
    @classmethod
    def _blank_keeps_old(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        return stripped or None
