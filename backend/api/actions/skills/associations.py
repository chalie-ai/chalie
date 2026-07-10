"""Skill associations action — nested resource under /api/skills/associations.

Covers:
- GET /api/skills/associations  → get (every pattern -> skill rule)

Only the sentinel/id-less form is meaningful — an id-addressed call 404s,
since associations are not addressed by a single skill id.
"""

from __future__ import annotations

from typing import cast

from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse, NotFoundError
from api.response.skills import SkillAssociationResponse
from models.skill_association import SkillAssociation as SkillAssociationModel
from services.file_mapper_service import FileMapperService
from services.time_utils import parse_utc


class SkillAssociations(Action):
    """Action listing every pattern -> skill association rule."""

    response_dto = {"get": DocumentedResponse(SkillAssociationResponse, listing=True)}

    def slug(self) -> str:
        return "skills"

    def verb(self) -> str:
        return "associations"

    def get(self, id: int | str) -> ResponseReturnValue:
        if not self.is_create(id):
            raise NotFoundError("Not found")

        if not FileMapperService.get_skills_db_path().exists():
            return SkillAssociationResponse.listing([], 1, 1, 0)

        items = [self._to_response(row) for row in SkillAssociationModel.with_skill_titles()]
        return SkillAssociationResponse.listing(items, 1, max(len(items), 1), len(items))

    def _to_response(self, row: dict[str, object]) -> SkillAssociationResponse:
        return SkillAssociationResponse(
            skill_id=cast(int, row["skill_id"]),
            skill_title=cast(str, row["skill_title"]),
            pattern_name=cast(str, row["pattern_name"]),
            rule=cast(str, row["rule"]),
            created_at=parse_utc(cast(str, row["created_at"])),
        )
