"""Response DTOs for the skills module.

Mirrors the field shape of the legacy DTOs (``backend/api/dto/skill.py``,
``skill_association.py``, ``skill_toggle.py``) so the wire format stays
identical.
"""

from __future__ import annotations

from datetime import datetime

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


class SkillAssociationResponse(Response):
    """Read shape for one pattern -> skill association rule."""

    skill_id: int
    skill_title: str
    pattern_name: str
    rule: str
    created_at: datetime


class SkillToggleResponse(Response):
    """Result of flipping a skill's enabled state."""

    skill_id: int
    enabled: bool
