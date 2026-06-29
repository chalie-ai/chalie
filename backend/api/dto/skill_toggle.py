"""DTO for the skill-toggle action result (not the full Skill resource)."""

from __future__ import annotations

from .base import DTO


class SkillToggleResult(DTO):
    """Result of PUT /api/skills/<id>/toggle — the toggled id + new enabled flag."""

    skill_id: int
    enabled: bool
