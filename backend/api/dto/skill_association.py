"""DTO for a skill-association read shape (the GET /api/skills co-loaded rows)."""

from __future__ import annotations

from datetime import datetime

from .base import DTO


class SkillAssociation(DTO):
    """One skill↔pattern association row; ``created_at`` serializes as ISO-8601 UTC."""

    skill_id: int
    skill_title: str
    pattern_name: str
    rule: str
    created_at: datetime
