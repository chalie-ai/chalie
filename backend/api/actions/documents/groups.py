"""Document classification groups action — nested resource under /api/documents/groups.

Covers:
- GET /api/documents/groups/<field>  → unique classification groupings for a field
"""

from __future__ import annotations

from typing import ClassVar, cast

from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse
from api.response.document import ClassificationGroup
from exceptions import NotFoundError
from services.document_service import DocumentService


class DocumentsGroups(Action):
    """Action returning unique classification groupings for a field."""

    id_type: ClassVar[type[int] | type[str]] = str
    response_dto = {"get": DocumentedResponse(ClassificationGroup, listing=True)}

    def slug(self) -> str:
        return "documents"

    def verb(self) -> str:
        return "groups"

    def get(self, id: int | str) -> ResponseReturnValue:
        if self.is_create(id):
            raise NotFoundError("Not found")
        field = cast(str, id)
        items = [
            ClassificationGroup(value=cast(str, g["value"]), count=cast(int, g["count"]))
            for g in DocumentService().get_classification_groups(field)
        ]
        return ClassificationGroup.listing(
            items, page=1, limit=len(items) or 1, total=len(items)
        )
