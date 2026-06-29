"""DTO for the skills GET envelope — the paired skills + associations result."""

from __future__ import annotations

from .base import DTO
from .skill import Skill
from .skill_association import SkillAssociation


class SkillListResult(DTO):
    """GET /api/skills returns two co-loaded collections, not a bare list."""

    skills: list[Skill]
    associations: list[SkillAssociation]
