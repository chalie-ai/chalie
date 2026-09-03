"""Memory search action — read-only search across the memory graph + map.

A single retrieval route fans the query out through ``MemoryRecallService``
which fuses Graph FTS (subject) + Map vector (situational cues) into one result.
Returns ranked hit DTOs; never raises — a miss surfaces as an empty listing.
"""

from __future__ import annotations

import logging
from typing import cast

from flask import request
from flask.typing import ResponseReturnValue

from api.action import Action
from api.endpoint import DocumentedResponse
from exceptions import EndpointError
from api.response.memory import MemoryHitResponse
from services.memory_recall_service import MemoryRecallService
from services.time_utils import parse_utc

logger = logging.getLogger(__name__)


class MemorySearch(Action):
    """Action searching the memory graph + map and returning ranked, merged hits."""

    # id is ignored — there is no per-id resource here, only a query-driven
    # search, so this handler can never 404.
    response_dto = {"get": DocumentedResponse(MemoryHitResponse, listing=True, not_found=False)}

    def get(self, id: int | str) -> ResponseReturnValue:
        """Search the memory graph + map and return ranked, merged hits.

        Accepts a required `q` query parameter (non-empty after stripping
        whitespace), matching the legacy MemorySearchQuery DTO validation.
        """
        raw_q = request.args.get("q")
        if raw_q is None:
            raise EndpointError("q is required")
        q = raw_q.strip()
        if not q:
            raise EndpointError("q must not be empty")

        results: list[MemoryHitResponse] = []

        try:
            recall_result = MemoryRecallService().recall(q)
            graph = recall_result.get("graph", [])
            for hit in graph:
                results.append(
                    MemoryHitResponse(
                        type="fact",
                        content=f"{cast(str, hit.get('subject', ''))}: {cast(str, hit.get('contents', ''))}",
                        score=1.0,
                        created_at=parse_utc(cast(str, hit.get("last_updated_at", ""))),
                    )
                )
            map_hits = recall_result.get("map", [])
            for hit in map_hits:
                results.append(
                    MemoryHitResponse(
                        type="episode",
                        content=cast(str, hit.get("contents", "")),
                        score=1.0,
                        created_at=parse_utc(cast(str, hit.get("generated_at", ""))),
                    )
                )
        except Exception as exc:
            logger.warning("[Memory] Recall failed: %s", exc)

        results.sort(key=lambda hit: hit.score, reverse=True)
        total = len(results)
        return MemoryHitResponse.listing(results, page=1, limit=max(total, 1), total=total)
