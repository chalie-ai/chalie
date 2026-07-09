"""Abstract base for every outbound response DTO.

Subclass :class:`Response` to declare the response contract for an endpoint.
``extra="forbid"`` rejects unknown keys outright; every ``datetime`` field
serializes as ISO-8601 UTC (``+00:00``), the exact format the frontend consumes,
so handlers never reformat dates themselves.
"""

from __future__ import annotations

from collections.abc import Sequence

from api.dto.base import DTO


class Response(DTO):
    """Abstract base for every outbound response DTO.

    This is the ONLY place response envelopes may be built. Endpoint handlers
    must use one of the three envelope builders below — never hand-construct
    the response shape.

    - :meth:`single` — a single resource instance.
    - :meth:`listing` — a paginated collection of resources.
    - :meth:`failure` — an error payload.
    """

    def single(self) -> dict[str, object]:
        """Wrap this DTO as the ``result`` of a successful single-resource envelope."""
        return {
            "success": True,
            "result": self.model_dump(mode="json"),
        }

    @classmethod
    def listing(
        cls,
        items: Sequence["Response"],
        page: int,
        limit: int,
        total: int,
    ) -> dict[str, object]:
        """Wrap a paginated collection of DTOs as a successful listing envelope."""
        return {
            "success": True,
            "result": [item.model_dump(mode="json") for item in items],
            "pagination": {"page": page, "limit": limit, "total": total},
        }

    @staticmethod
    def failure(message: str) -> dict[str, object]:
        """Build an error envelope with the given message."""
        return {
            "success": False,
            "result": [],
            "error": message,
        }
