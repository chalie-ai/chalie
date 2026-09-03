"""Response DTOs for the skills module.

Mirrors the field shape of the legacy DTOs (``backend/api/dto/skill.py``,
``skill_toggle.py``) so the wire format stays identical.
"""

from __future__ import annotations

from typing import cast

from models.skill import Skill
from .response import Response


class SkillResponse(Response):
    """Read shape for a skill — matches the legacy Skill DTO fields."""

    id: int
    title: str
    use_for: str
    content: str
    tags: str
    version: int
    source: str
    enabled: bool
    based_on: int | None

    @classmethod
    def from_model(cls, skill: Skill) -> "SkillResponse":
        """Project a persisted :class:`~models.skill.Skill` row into this read DTO.

        ``enabled`` and ``based_on`` are SQL defaults: an instance built for
        INSERT never carries them, so callers pass a row read back from the
        database, never the instance they just saved.
        """
        return cls(
            id=cast(int, skill.id),
            title=skill.title,
            use_for=skill.use_for,
            content=skill.content,
            tags=skill.tags or "",
            version=skill.version,
            source=skill.source,
            enabled=bool(skill.enabled),
            based_on=skill.based_on,
        )


class SkillToggleResponse(Response):
    """Result of flipping a skill's enabled state."""

    skill_id: int
    enabled: bool
