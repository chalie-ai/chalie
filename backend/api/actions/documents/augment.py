"""Document augmentation action — nested resource under /api/documents/augment.

Covers:
- POST /api/documents/augment/<id>  → add user context and confirm

The base auto-documents 404 only for get/delete handlers, so this post
declares its 404 and 400 explicitly via ``extras`` to keep swagger truthful.
"""

from __future__ import annotations

from typing import ClassVar, cast

from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse
from api.request import Request
from api.request.document import AugmentRequest
from exceptions import EndpointError, NotFoundError
from services.document_service import DocumentService


class DocumentsAugment(Action):
    """Action to add user context to a document awaiting confirmation."""

    id_type: ClassVar[type[int] | type[str]] = str
    request_dto = AugmentRequest
    response_dto = {
        "post": DocumentedResponse(
            extras=((404, "Not found"), (400, "Invalid state")),
        )
    }

    def slug(self) -> str:
        return "documents"

    def verb(self) -> str:
        return "augment"

    def post(self, id: int | str, data: Request | None) -> ResponseReturnValue:
        if self.is_create(id):
            raise NotFoundError("Not found")
        dto = cast(AugmentRequest, data)
        svc = DocumentService()
        doc = svc.get_document(cast(str, id))
        if not doc:
            raise NotFoundError("Not found")
        if doc["status"] not in ("awaiting_confirmation", "ready"):
            raise EndpointError("Document cannot be augmented in its current state")

        metadata = cast("dict[str, object]", doc.get("extracted_metadata") or {})
        metadata["_user_context"] = dto.context

        svc.update_extracted_metadata(
            cast(str, id),
            metadata=metadata,
            summary=cast(str, doc.get("summary", "")),
            summary_embedding=cast("list[float] | None", doc.get("summary_embedding")),
        )

        if doc["status"] != "ready":
            svc.update_status(
                cast(str, id),
                "ready",
                chunk_count=cast(int, doc.get("chunk_count", 0)),
            )
        return "", 204
